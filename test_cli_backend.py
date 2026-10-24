"""Unit tests for authenticated CLI adapters and the MCP boundary."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cli_backend import CliBackend, parse_cli_output, sanitized_environment
from mcp_server import _BearerApp, _json_type, _tool_schema, _validate_tool_paths


class _FakeProcess:
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0,
                 timeout: bool = False) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.timeout = timeout
        self.calls = 0
        self.prompt = ""
        self.pid = 1234

    def communicate(self, prompt: str = "", timeout: int | None = None) -> tuple[str, str]:
        self.prompt = prompt
        self.calls += 1
        if self.timeout and self.calls == 1:
            import subprocess
            raise subprocess.TimeoutExpired("fake-cli", timeout)
        return self.stdout, self.stderr

    def kill(self) -> None:
        self.returncode = -9


class _Tool:
    name = "read_file"
    description = "Legge un file della repo."
    inputs = {
        "path": {"type": "string", "description": "Percorso relativo."},
        "end_line": {"type": "integer", "description": "Riga finale.", "nullable": True},
    }

    def __init__(self) -> None:
        self.ctx = SimpleNamespace(safe_path=self._safe_path)

    @staticmethod
    def _safe_path(path: str) -> str:
        if path.startswith("..") or path.startswith("/"):
            raise ValueError("percorso fuori dalla repo")
        return path

    def forward(self, path: str, end_line: int | None = None) -> str:
        return path


class CliParserTests(unittest.TestCase):
    def test_cli_instructions_do_not_require_final_answer_tool(self) -> None:
        from prompts import build_cli_instructions
        prompt = build_cli_instructions("theory", "repo", "stats", "map")
        self.assertNotIn("-> final_answer. Riserva l'ultimo passo a final_answer.", prompt)
        self.assertIn("Rispondi direttamente", prompt)

    def test_four_output_formats(self) -> None:
        samples = {
            "claude_cli": ('Claude', '{"type":"result","result":"Claude answer","usage":{"input_tokens":4,"output_tokens":5}}'),
            "codex_cli": ('Codex', '\n'.join([
                '{"type":"item.completed","item":{"type":"agent_message","text":"Codex answer"}}',
                '{"type":"turn.completed","usage":{"input_tokens":7,"output_tokens":8}}',
            ])),
            "mistral_vibe_cli": ('Vibe', '[{"role":"assistant","content":"Vibe answer"}]'),
            "antigravity_cli": ('Antigravity', '{"response":"Antigravity answer","usage":{"prompt_tokens":2,"completion_tokens":3}}'),
        }
        for provider, (expected, raw) in samples.items():
            with self.subTest(provider=provider):
                text, incoming, outgoing, _ = parse_cli_output(provider, raw)
                self.assertIn(expected, text)
                if provider == "claude_cli":
                    self.assertEqual((incoming, outgoing), (4, 5))
                if provider == "codex_cli":
                    self.assertEqual((incoming, outgoing), (7, 8))
                if provider == "antigravity_cli":
                    self.assertEqual((incoming, outgoing), (2, 3))

    def test_environment_removes_api_keys_but_keeps_cli_login(self) -> None:
        env = sanitized_environment({
            "OPENAI_API_KEY": "secret",
            "MISTRAL_API_TOKEN": "secret",
            "CLAUDE_CODE_OAUTH_TOKEN": "cli-session",
            "PATH": "bin",
        })
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("MISTRAL_API_TOKEN", env)
        self.assertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "cli-session")
        self.assertEqual(env["PATH"], "bin")

    def test_subprocess_success_passes_task_on_stdin_and_sanitizes_environment(self) -> None:
        proc = _FakeProcess(stdout='{"type":"item.completed","item":{"type":"agent_message","text":"Done"}}')
        with patch("cli_backend.shutil.which", return_value="fake-codex"), \
             patch("cli_backend.subprocess.Popen", return_value=proc) as popen, \
             patch.dict(os.environ, {"OPENAI_API_KEY": "secret", "CODEX_API_KEY": "secret"}):
            backend = CliBackend("codex_cli", {"command": "codex"})
            result = backend.complete("system rules", "inspect files")
        self.assertTrue(result.ok)
        self.assertEqual(result.text, "Done")
        self.assertIn("inspect files", proc.prompt)
        child_env = popen.call_args.kwargs["env"]
        self.assertNotIn("OPENAI_API_KEY", child_env)
        self.assertNotIn("CODEX_API_KEY", child_env)

    def test_timeout_terminates_process_tree(self) -> None:
        proc = _FakeProcess(timeout=True)
        group_kill = patch("cli_backend.os.killpg", create=True)
        with patch("cli_backend.shutil.which", return_value="fake-codex"), \
             patch("cli_backend.subprocess.Popen", return_value=proc), \
             patch("cli_backend.subprocess.run") as taskkill, \
             group_kill as killpg:
            backend = CliBackend("codex_cli", {"command": "codex"}, timeout=1)
            result = backend.complete("rules", "task")
        self.assertFalse(result.ok)
        self.assertIn("Timeout", result.error)
        if os.name == "nt":
            taskkill.assert_called_once()
        else:
            killpg.assert_called_once()

    def test_nonzero_exit_surfaces_authentication_error(self) -> None:
        proc = _FakeProcess(stderr="not logged in", returncode=1)
        with patch("cli_backend.shutil.which", return_value="fake-claude"), \
             patch("cli_backend.subprocess.Popen", return_value=proc):
            result = CliBackend("claude_cli", {"command": "claude"}).complete("rules", "task")
        self.assertFalse(result.ok)
        self.assertIn("Autenticazione o quota CLI", result.error)

    def test_each_profile_has_scoped_temporary_mcp_config(self) -> None:
        profiles = {
            "claude_cli": "claude",
            "codex_cli": "codex",
            "mistral_vibe_cli": "vibe",
            "antigravity_cli": "agy",
        }
        with patch("cli_backend.shutil.which", side_effect=lambda command: command):
            for provider, executable in profiles.items():
                with self.subTest(provider=provider), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp)
                    backend = CliBackend(provider, {"command": executable})
                    command = backend._config(root, "http://127.0.0.1:1234/mcp", "temporary-token",
                                              ["read_file", "list_dir"])
                    if provider == "claude_cli":
                        self.assertIn("mcp__codemaster__read_file", command)
                        data = json.loads((root / "claude-mcp.json").read_text(encoding="utf-8"))
                        self.assertEqual(data["mcpServers"]["codemaster"]["headers"]["Authorization"],
                                         "Bearer temporary-token")
                    elif provider == "codex_cli":
                        self.assertIn("--disable", command)
                        self.assertIn("shell_tool", command)
                        command_text = " ".join(command)
                        self.assertIn("CODEMASTER_MCP_TOKEN", command_text)
                        self.assertIn("http://127.0.0.1:1234/mcp", command_text)
                    elif provider == "mistral_vibe_cli":
                        config = (root / ".vibe" / "config.toml").read_text(encoding="utf-8")
                        self.assertIn('enabled_tools = ["codemaster_read_file", "codemaster_list_dir"]', config)
                        self.assertIn('api_key_env = "CODEMASTER_MCP_TOKEN"', config)
                    else:
                        config = json.loads((root / ".agents" / "mcp_config.json").read_text(encoding="utf-8"))
                        self.assertEqual(config["mcpServers"]["codemaster"]["enabledTools"], ["read_file", "list_dir"])
                        agent = (root / ".agents" / "agents" / "codemaster.md").read_text(encoding="utf-8")
                        self.assertIn("mcp_codemaster_read_file", agent)
                        self.assertIn("commandExecutionPolicy: off", agent)


class MCPBoundaryTests(unittest.TestCase):
    def test_json_schema_types_preserve_arrays_and_objects(self) -> None:
        self.assertEqual(_json_type("array"), "array")
        self.assertEqual(_json_type("object"), "object")

    def test_tool_schema_matches_inputs_and_signature(self) -> None:
        schema = _tool_schema(_Tool())
        self.assertEqual(set(schema["properties"]), {"path", "end_line"})
        self.assertEqual(schema["required"], ["path"])
        self.assertEqual(schema["properties"]["end_line"]["type"], ["integer", "null"])
        self.assertFalse(schema["additionalProperties"])

    def test_invalid_repo_path_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _validate_tool_paths(_Tool(), {"path": "../outside.txt"})

    def test_wrong_bearer_gets_401(self) -> None:
        async def run() -> list[dict]:
            sent: list[dict] = []
            async def app(_scope, _receive, _send):
                raise AssertionError("request passed the bearer guard")
            async def send(message):
                sent.append(message)
            wrapped = _BearerApp(app, "expected")
            await wrapped({"type": "http", "headers": [(b"authorization", b"Bearer wrong")]}, None, send)
            return sent
        messages = asyncio.run(run())
        self.assertEqual(messages[0]["status"], 401)

    def test_optional_tools_follow_existing_flags(self) -> None:
        from agent import RepoAgent
        tools = {name: SimpleNamespace(name=name) for name in
                 ("read_file", "run_command", "web_lookup")}
        disabled = SimpleNamespace(cfg=SimpleNamespace(allow_exec=False, allow_web=False), _all_tools=tools)
        names = {tool.name for tool in RepoAgent._tools_for(disabled, "practical")}
        self.assertNotIn("run_command", names)
        self.assertNotIn("web_lookup", names)
        enabled = SimpleNamespace(cfg=SimpleNamespace(allow_exec=True, allow_web=True), _all_tools=tools)
        names = {tool.name for tool in RepoAgent._tools_for(enabled, "practical")}
        self.assertIn("run_command", names)
        self.assertIn("web_lookup", names)


class AgentProviderTests(unittest.TestCase):
    def test_api_model_factory_stays_on_api_path(self) -> None:
        from agent import ResilientModelWrapper, build_model
        model = SimpleNamespace(model_id="api/model")
        config = SimpleNamespace(
            providers={"api_alias": {"model_id": "api/model"}},
            max_output_tokens=100, fallback_providers=(), llm_max_retries=1,
            llm_retry_base_delay=0, llm_switch_after=1, llm_cooldown_s=1,
        )
        with patch("agent._make_raw_model", return_value=model):
            built = build_model("api_alias", cfg=config, fallbacks=[])
        self.assertIsInstance(built, ResilientModelWrapper)
        self.assertIs(built._inner, model)

    def test_cli_alias_is_rejected_as_api_fallback(self) -> None:
        from agent import build_model
        config = SimpleNamespace(
            providers={"cli_alias": {"type": "cli", "command": "codex"}},
            max_output_tokens=100, fallback_providers=(), llm_max_retries=1,
            llm_retry_base_delay=0, llm_switch_after=1, llm_cooldown_s=1,
        )
        with self.assertRaisesRegex(ValueError, "solo provider API"):
            build_model("api_alias", cfg=config, fallbacks=["cli_alias"])

    def test_create_agent_rejects_cli_fallback_before_repository_access(self) -> None:
        from agent import create_agent
        from config import Config
        config = Config(providers={
            "api_alias": {"model_id": "api/model"},
            "cli_alias": {"type": "cli", "command": "codex"},
        })
        with self.assertRaisesRegex(ValueError, "fallback deve usare alias API"):
            create_agent("missing-repo", provider="api_alias", cfg=config, fallbacks=["cli_alias"])

    def test_cli_failure_uses_only_configured_api_fallback(self) -> None:
        from agent import AskStats, RepoAgent
        from cli_backend import CliResult

        class Progress:
            def reset(self) -> None:
                self.t0 = 1.0
                self.step = self.in_tok = self.out_tok = 0

            def finish(self) -> None:
                pass

        class APIAgent:
            def __init__(self) -> None:
                self.model = SimpleNamespace(model_id="api/model", rate_limit_hits=0)
                self.tools = {}
                self.memory = SimpleNamespace(steps=[])

            def run(self, *_args, **_kwargs) -> str:
                return "API fallback response"

        class CLIBackend:
            model_id = "codex"

            def agentic(self, *_args, **_kwargs) -> CliResult:
                return CliResult(error="not logged in")

        repo = RepoAgent.__new__(RepoAgent)
        repo._ask_lock = __import__("threading").RLock()
        repo.mode, repo.pinned = "theory", True
        repo.agent, repo.fallback_agent = None, APIAgent()
        repo.fallback_provider, repo.provider = "gemini_3.8", "codex_cli"
        repo.cli_backend, repo.backend = CLIBackend(), "cli"
        repo._mcp_server = SimpleNamespace(take_trace=lambda: [])
        repo._local_mcp = lambda _tools: ("http://127.0.0.1:1234/mcp", "token")
        repo._tools_for = lambda _mode, _question=None: []
        repo._try_fast_path = lambda _question, _history: None
        repo._compose_task = lambda _question, _history: "task"
        repo._cli_instructions = lambda _mode: "rules"
        repo._log_telemetry = lambda _question: None
        repo._cache_key = lambda _question, _history: None
        repo._ctx = SimpleNamespace(file_text=lambda _path: "")
        repo.index = SimpleNamespace(files={})
        repo.progress = Progress()
        repo.last = AskStats()

        answer, trace = repo.ask("Explain the architecture")
        self.assertEqual(answer, "API fallback response")
        self.assertEqual(repo.last.backend, "api")
        self.assertEqual(repo.last.provider, "gemini_3.8")
        self.assertTrue(repo.last.fallback)
        self.assertIn("fallback=api:gemini_3.8", trace)


if __name__ == "__main__":
    unittest.main()
