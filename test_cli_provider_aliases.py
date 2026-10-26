"""Regression tests for configured and documented CLI provider names."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cli_backend import CliBackend


PROVIDERS = {
    "claude": {"type": "cli", "command": "claude"},
    "codex": {"type": "cli", "command": "codex"},
    "vibe": {"type": "cli", "command": "vibe"},
    "agy": {"type": "cli", "command": "agy"},
}
ALIASES = {
    "claude_cli": "claude",
    "codex_cli": "codex",
    "mistral_vibe_cli": "vibe",
    "antigravity_cli": "agy",
}


class _FakeProcess:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = 0
        self.prompt = ""

    def communicate(self, prompt: str, timeout: int) -> tuple[str, str]:
        self.prompt = prompt
        return self.stdout, self.stderr


class CliProviderAliasTests(unittest.TestCase):
    def test_configured_short_aliases_select_cli_profiles(self) -> None:
        samples = {
            "claude": ("claude", "--strict-mcp-config", '{"type":"result","result":"Claude answer"}'),
            "codex": ("codex", "exec", '{"type":"item.completed","item":{"type":"agent_message","text":"Codex answer"}}'),
            "vibe": ("vibe", "--prompt", '[{"role":"assistant","content":"Vibe answer"}]'),
            "agy": ("agy", "--input-format", '{"response":"Antigravity answer"}'),
        }
        with patch("cli_backend.shutil.which", side_effect=lambda command: f"fake-{command}"), \
             patch("cli_backend.subprocess.Popen") as popen:
            for provider, (executable, expected_arg, output) in samples.items():
                with self.subTest(provider=provider):
                    popen.return_value = _FakeProcess(output)
                    backend = CliBackend(provider, {"command": executable})
                    self.assertTrue(backend.complete("system rules", "inspect files").ok)
                    self.assertEqual(backend.provider, provider)
                    self.assertIn(expected_arg, popen.call_args.args[0])

    def test_run_cli_maps_documented_aliases_to_configured_profiles(self) -> None:
        import run_cli

        agent = SimpleNamespace(index=SimpleNamespace(
            repo_root=SimpleNamespace(name="repo"), stats=lambda: "stats",
        ))
        with patch("run_cli.load_providers_config", return_value=PROVIDERS), \
             patch("run_cli.Config", return_value=SimpleNamespace()), \
             patch("run_cli.create_agent", return_value=agent) as create_agent, \
             patch("run_cli.os.system"), \
             patch("builtins.input", side_effect=EOFError):
            for alias, configured_name in ALIASES.items():
                with self.subTest(alias=alias), patch("sys.argv", ["run_cli.py", "repo", "--provider", alias]):
                    run_cli.main()
                    self.assertEqual(create_agent.call_args.kwargs["provider"], configured_name)

    def test_run_chat_maps_documented_aliases_to_configured_profiles(self) -> None:
        import run_chat

        with patch("run_chat.load_providers_config", return_value=PROVIDERS), \
             patch("run_chat.Config", return_value=SimpleNamespace()), \
             patch("run_chat.create_agent", return_value=SimpleNamespace()) as create_agent, \
             patch("run_chat.launch_chat"):
            for alias, configured_name in ALIASES.items():
                with self.subTest(alias=alias), patch("sys.argv", ["run_chat.py", "repo", "--provider", alias]):
                    run_chat.main()
                    self.assertEqual(create_agent.call_args.kwargs["provider"], configured_name)


if __name__ == "__main__":
    unittest.main()
