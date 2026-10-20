"""
Codex CLI language model: chat completions through a local `codex exec`
process, authenticated by the user's existing `codex login` (ChatGPT plan)
instead of an API key. See ADR-0XX-codex-cli-provider.md.

Source content and chat messages reach this process verbatim, so every call
runs Codex with its shell tool, plugins, MCP servers and user config switched
off, in a read-only sandbox rooted at an empty temporary directory.
"""

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Union

from esperanto import LanguageModel
from esperanto.common_types import (
    ChatCompletion,
    ChatCompletionChunk,
    Choice,
    Message,
    Model,
    Tool,
)
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from loguru import logger

from open_notebook.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ExternalServiceError,
)
from open_notebook.utils.text_utils import extract_text_content

PROVIDER_NAME = "codex"

_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "plugins",
    "browser_use",
    "computer_use",
    "image_generation",
    "memories",
    "multi_agent",
    "hooks",
    "goals",
)

_SECRET_ENV_MARKERS = ("KEY", "SECRET", "PASSWORD", "TOKEN")

_ROLE_LABELS = {
    "system": "System",
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "ai": "Assistant",
    "tool": "Tool result",
}

_PREAMBLE = (
    "You are the text-generation backend of a research notebook app, not a "
    "coding agent. You have no tools. Answer directly from the conversation below."
)


def _codex_binary() -> str:
    path = shutil.which(os.environ.get("CODEX_CLI_PATH", "codex"))
    if not path:
        raise ConfigurationError(
            "Codex CLI not found. Install it and run `codex login`, or set "
            "CODEX_CLI_PATH to the codex executable."
        )
    return path


def _timeout_seconds() -> float:
    return float(os.environ.get("ESPERANTO_LLM_TIMEOUT") or 180)


def _child_env() -> Dict[str, str]:
    """The app's own secrets (provider keys, DB password) never reach Codex."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(marker in k.upper() for marker in _SECRET_ENV_MARKERS)
    }


def build_exec_args(
    codex_bin: str, model_name: str, workdir: str, output_file: str
) -> List[str]:
    args = [
        codex_bin,
        "exec",
        "--ignore-user-config",
        "--ignore-rules",
        "--ephemeral",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "-C",
        workdir,
        "-o",
        output_file,
        "-c",
        'web_search="disabled"',
    ]
    for feature in _DISABLED_FEATURES:
        args += ["--disable", feature]
    if model_name:
        args += ["-m", model_name]
    args.append("-")
    return args


def build_prompt(messages: List[Dict[str, Any]], json_mode: bool = False) -> str:
    """Flatten role-tagged chat messages into the single prompt `codex exec` takes."""
    parts = [_PREAMBLE]
    for msg in messages:
        label = _ROLE_LABELS.get(str(msg.get("role", "user")), "User")
        text = extract_text_content(msg.get("content")).strip()
        if text:
            parts.append(f"### {label}\n{text}")
    instruction = "Write the assistant's next reply. Output only the reply text."
    if json_mode:
        instruction += " The reply must be a single valid JSON object."
    parts.append(instruction)
    return "\n\n".join(parts)


def _raise_for_failure(returncode: int, stderr: str) -> None:
    tail = "\n".join(stderr.strip().splitlines()[-5:])
    lower = stderr.lower()
    if "not logged in" in lower or "401" in lower or "unauthorized" in lower:
        raise AuthenticationError(
            "Codex CLI is not logged in. Run `codex login` on the API host."
        )
    raise ExternalServiceError(f"Codex CLI failed (exit {returncode}): {tail}")


def run_codex_sync(prompt: str, model_name: str) -> str:
    codex_bin = _codex_binary()
    timeout = _timeout_seconds()
    with tempfile.TemporaryDirectory(prefix="on-codex-") as workdir:
        output_file = str(Path(workdir) / "reply.txt")
        try:
            result = subprocess.run(
                build_exec_args(codex_bin, model_name, workdir, output_file),
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=_child_env(),
            )
        except subprocess.TimeoutExpired as e:
            raise ExternalServiceError(
                f"Codex CLI timed out after {timeout:.0f}s"
            ) from e
        if result.returncode != 0:
            _raise_for_failure(result.returncode, result.stderr)
        return Path(output_file).read_text(encoding="utf-8").strip()


async def run_codex(prompt: str, model_name: str) -> str:
    codex_bin = _codex_binary()
    timeout = _timeout_seconds()
    with tempfile.TemporaryDirectory(prefix="on-codex-") as workdir:
        output_file = str(Path(workdir) / "reply.txt")
        proc = await asyncio.create_subprocess_exec(
            *build_exec_args(codex_bin, model_name, workdir, output_file),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=_child_env(),
        )
        try:
            _, stderr = await asyncio.wait_for(
                proc.communicate(prompt.encode("utf-8")), timeout
            )
        except asyncio.TimeoutError as e:
            proc.kill()
            await proc.wait()
            raise ExternalServiceError(
                f"Codex CLI timed out after {timeout:.0f}s"
            ) from e
        if proc.returncode != 0:
            _raise_for_failure(proc.returncode or 1, stderr.decode(errors="replace"))
        return Path(output_file).read_text(encoding="utf-8").strip()


async def _run_codex_command(*args: str) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        _codex_binary(),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_child_env(),
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(), 30)
    return (
        proc.returncode or 0,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


async def check_codex_login() -> tuple[bool, str]:
    try:
        code, stdout, stderr = await _run_codex_command("login", "status")
    except ConfigurationError as e:
        return False, str(e)
    status = (stdout or stderr).strip().splitlines()
    message = status[0] if status else "Unknown Codex login status"
    return code == 0 and "logged in" in message.lower(), message


async def list_codex_models() -> List[str]:
    """Model slugs the Codex CLI offers in its picker (`codex debug models`)."""
    try:
        code, stdout, stderr = await _run_codex_command("debug", "models")
        if code != 0:
            raise ExternalServiceError(stderr.strip()[-200:])
        catalog = json.loads(stdout).get("models", [])
    except Exception as e:
        logger.warning(f"Failed to list Codex models: {e}")
        return []
    return [
        m["slug"] for m in catalog if m.get("slug") and m.get("visibility") == "list"
    ]


def _langchain_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    return [{"role": m.type, "content": m.content} for m in messages]


class ChatCodexCLI(BaseChatModel):
    """LangChain chat model backed by `codex exec`."""

    model_name: str = ""
    json_mode: bool = False

    @property
    def _llm_type(self) -> str:
        return "codex-cli"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = build_prompt(_langchain_to_dicts(messages), self.json_mode)
        text = run_codex_sync(prompt, self.model_name)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = build_prompt(_langchain_to_dicts(messages), self.json_mode)
        text = await run_codex(prompt, self.model_name)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


@dataclass
class CodexLanguageModel(LanguageModel):
    """Esperanto-compatible wrapper so ModelManager callers treat Codex like any provider.

    Sampling settings (temperature, max_tokens, top_p) and tool calling are
    accepted for interface compatibility but ignored: `codex exec` exposes neither.
    """

    @property
    def provider(self) -> str:
        return PROVIDER_NAME

    def _get_default_model(self) -> str:
        return ""

    def _get_models(self) -> List[Model]:
        return []

    def _json_mode(self) -> bool:
        return isinstance(self.structured, dict) and self.structured.get("type") in (
            "json",
            "json_object",
        )

    def _completion(self, text: str) -> ChatCompletion:
        return ChatCompletion(
            id=str(uuid.uuid4()),
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            model=self.model_name or PROVIDER_NAME,
            provider=PROVIDER_NAME,
        )

    def chat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletion, Generator[ChatCompletionChunk, None, None]]:
        prompt = build_prompt(messages, self._json_mode())
        return self._completion(run_codex_sync(prompt, self.model_name or ""))

    async def achat_complete(
        self,
        messages: List[Dict[str, Any]],
        stream: Optional[bool] = None,
        tools: Optional[List[Tool]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        parallel_tool_calls: Optional[bool] = None,
        validate_tool_calls: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Union[ChatCompletion, AsyncGenerator[ChatCompletionChunk, None]]:
        prompt = build_prompt(messages, self._json_mode())
        return self._completion(await run_codex(prompt, self.model_name or ""))

    def to_langchain(self) -> ChatCodexCLI:
        return ChatCodexCLI(
            model_name=self.model_name or "", json_mode=self._json_mode()
        )
