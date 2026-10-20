# ADR-0XX: Codex CLI provider runs `codex exec` outside Esperanto

- **Status**: Proposed
- **Date**: 2026-10

## Context

Users with a ChatGPT plan but no OpenAI API key want to use the models their Codex CLI login already gives them. That login is OAuth to ChatGPT, not an API key, so Esperanto (which every other provider goes through) has no way to use it. Calling the ChatGPT backend directly with the tokens in `~/.codex/auth.json` would work, but that API isn't offered to third-party apps and can change or be restricted at any time.

## Decision

**Add a `codex` language-only provider that runs the official `codex exec` CLI as a subprocess, bypassing Esperanto.** `ModelManager.get_model()` returns a `CodexLanguageModel` (an Esperanto `LanguageModel` subclass) for `provider == "codex"`. Its `to_langchain()` returns a `ChatCodexCLI` LangChain chat model, so `provision_langchain_model()` and every graph work unchanged. Messages are flattened into one role-labelled prompt on stdin, and the reply is read from `--output-last-message`.

Source text reaches Codex verbatim, so a document could carry a prompt injection. Every call therefore runs with:

- `--ignore-user-config` and `--ignore-rules`: no user MCP servers, profiles, hooks or exec rules;
- `--disable shell_tool --disable unified_exec` plus apps, plugins, browser/computer use, memories, multi-agent, hooks and goals; `web_search="disabled"`;
- `--sandbox read-only`, an empty temporary working directory and `--ephemeral`;
- an environment without any variable whose name contains `KEY`, `SECRET`, `PASSWORD` or `TOKEN`.

The credential record is only a placeholder that makes the provider show up in Manage → Models. Its connection test runs `codex login status`, and discovery lists the `visibility: list` entries of `codex debug models`.

## Alternatives considered

- **Call the ChatGPT backend with the Codex OAuth tokens**: faster and no CLI needed, but not a supported API for third-party apps; rejected.
- **A local OpenAI-compatible proxy in front of the Codex login**: needs no code changes, but carries the same terms-of-service risk and means trusting third-party code with the login; rejected.
- **Register the class in Esperanto's `AIFactory._provider_modules`**: a private attribute, so it would break silently on an Esperanto upgrade; rejected in favor of an explicit branch next to the `anthropic_compatible` one.

## Consequences

- Each call starts a Codex process: a few seconds of overhead and roughly 10k tokens of Codex base instructions per call against the plan's limits. No token streaming.
- `temperature`, `max_tokens` and `top_p` are ignored, and tool calling isn't available. JSON mode is a prompt instruction, not enforced.
- The API and worker must run where `codex` is installed and logged in, normally a from-source setup (`make start-all`). The Docker image doesn't ship Codex.
- Embeddings, text-to-speech and speech-to-text still need another provider (Ollama works without a key). Podcasts build their own clients through `podcast_creator`, so Codex models can't drive podcasts.
- A future Codex release that renames or removes a disabled feature flag will make `codex exec` fail loudly. `tests/test_codex_cli.py` pins the flag set.
