import stat
import sys
from unittest.mock import AsyncMock

import pytest
from esperanto.common_types import ChatCompletion
from langchain_core.messages import HumanMessage, SystemMessage

from open_notebook.ai import codex_cli
from open_notebook.ai.codex_cli import (
    ChatCodexCLI,
    CodexLanguageModel,
    build_exec_args,
    build_prompt,
)
from open_notebook.exceptions import AuthenticationError, ConfigurationError

FAKE_CODEX = """#!{python}
import sys
args = sys.argv[1:]
prompt = sys.stdin.read()
if args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT")
    sys.exit(0)
if args[:2] == ["debug", "models"]:
    print('{{"models": [{{"slug": "gpt-a", "visibility": "list"}}, {{"slug": "hidden", "visibility": "hide"}}]}}')
    sys.exit(0)
{body}
"""

ECHO_BODY = """
out = args[args.index("-o") + 1]
model = args[args.index("-m") + 1] if "-m" in args else "default"
with open(out, "w") as f:
    f.write(f"model={model} prompt_chars={len(prompt)}")
"""


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    def install(body: str = ECHO_BODY):
        path = tmp_path / "codex"
        path.write_text(FAKE_CODEX.format(python=sys.executable, body=body))
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("CODEX_CLI_PATH", str(path))
        return path

    return install


class TestExecArgs:
    def test_locks_down_codex(self):
        args = build_exec_args("codex", "gpt-a", "/tmp/w", "/tmp/w/out.txt")
        for flag in (
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
        ):
            assert flag in args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("-C") + 1] == "/tmp/w"
        disabled = {args[i + 1] for i, a in enumerate(args) if a == "--disable"}
        assert {"shell_tool", "unified_exec", "plugins", "apps"} <= disabled
        assert not any("dangerously" in a or a == "--yolo" for a in args)
        assert args[-1] == "-"

    def test_model_flag_only_when_named(self):
        assert "-m" not in build_exec_args("codex", "", "/w", "/w/o")
        args = build_exec_args("codex", "gpt-a", "/w", "/w/o")
        assert args[args.index("-m") + 1] == "gpt-a"


class TestPrompt:
    def test_labels_roles_and_keeps_order(self):
        prompt = build_prompt(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello"},
                {"role": "user", "content": [{"type": "text", "text": "Again"}]},
            ]
        )
        assert prompt.index("### System\nBe brief.") < prompt.index("### User\nHi")
        assert "### Assistant\nHello" in prompt
        assert "### User\nAgain" in prompt
        assert "JSON" not in prompt

    def test_json_mode_instruction(self):
        assert "valid JSON object" in build_prompt(
            [{"role": "user", "content": "x"}], True
        )


def test_child_env_drops_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPEN_NOTEBOOK_ENCRYPTION_KEY", "x")
    monkeypatch.setenv("SURREAL_PASSWORD", "x")
    monkeypatch.setenv("HOME", "/home/u")
    env = codex_cli._child_env()
    assert "OPENAI_API_KEY" not in env
    assert "OPEN_NOTEBOOK_ENCRYPTION_KEY" not in env
    assert "SURREAL_PASSWORD" not in env
    assert env["HOME"] == "/home/u"


def test_missing_binary_is_configuration_error(monkeypatch):
    monkeypatch.setenv("CODEX_CLI_PATH", "/nonexistent/codex")
    with pytest.raises(ConfigurationError):
        codex_cli.run_codex_sync("hi", "")


def test_sync_run_returns_reply(fake_codex):
    fake_codex()
    assert codex_cli.run_codex_sync("hello", "gpt-a") == "model=gpt-a prompt_chars=5"


@pytest.mark.asyncio
async def test_async_run_returns_reply(fake_codex):
    fake_codex()
    assert await codex_cli.run_codex("hello", "") == "model=default prompt_chars=5"


@pytest.mark.asyncio
async def test_auth_failure_maps_to_authentication_error(fake_codex):
    fake_codex('sys.stderr.write("401 Unauthorized\\n"); sys.exit(1)')
    with pytest.raises(AuthenticationError):
        await codex_cli.run_codex("hello", "gpt-a")


@pytest.mark.asyncio
async def test_login_status_and_model_listing(fake_codex):
    fake_codex()
    assert await codex_cli.check_codex_login() == (True, "Logged in using ChatGPT")
    assert await codex_cli.list_codex_models() == ["gpt-a"]


@pytest.mark.asyncio
async def test_esperanto_wrapper_and_langchain(fake_codex):
    fake_codex()
    model = CodexLanguageModel(
        model_name="gpt-a", config={"structured": {"type": "json"}}
    )
    completion = await model.achat_complete([{"role": "user", "content": "Hi"}])
    assert isinstance(completion, ChatCompletion)
    assert completion.content.startswith("model=gpt-a")

    lc = model.to_langchain()
    assert isinstance(lc, ChatCodexCLI) and lc.json_mode
    reply = lc.invoke([SystemMessage(content="sys"), HumanMessage(content="Hi")])
    assert reply.text.startswith("model=gpt-a")
    assert (await lc.ainvoke("Hi")).text.startswith("model=gpt-a")


@pytest.mark.asyncio
async def test_model_manager_builds_codex_model_without_credentials(monkeypatch):
    from open_notebook.ai import models as models_module

    record = models_module.Model(name="gpt-a", provider="codex", type="language")
    monkeypatch.setattr(models_module.Model, "get", AsyncMock(return_value=record))
    provision = AsyncMock()
    monkeypatch.setattr(
        "open_notebook.ai.key_provider.provision_provider_keys", provision
    )

    model = await models_module.ModelManager().get_model("model:x", max_tokens=10)

    assert isinstance(model, CodexLanguageModel)
    assert model.model_name == "gpt-a"
    provision.assert_not_awaited()


@pytest.mark.asyncio
async def test_model_manager_rejects_non_language_codex(monkeypatch):
    from open_notebook.ai import models as models_module

    record = models_module.Model(name="x", provider="codex", type="embedding")
    monkeypatch.setattr(models_module.Model, "get", AsyncMock(return_value=record))
    with pytest.raises(ConfigurationError):
        await models_module.ModelManager().get_model("model:x")
