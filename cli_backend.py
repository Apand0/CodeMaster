"""Subprocess adapters for locally authenticated coding-agent CLIs."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLI_PROVIDER_ALIASES = {
    "claude": "claude_cli",
    "codex": "codex_cli",
    "vibe": "mistral_vibe_cli",
    "agy": "antigravity_cli",
}
CLI_PROVIDER_PROFILES = {alias: profile for profile, alias in CLI_PROVIDER_ALIASES.items()}


@dataclass
class CliResult:
    text: str = ""
    ok: bool = False
    error: str = ""
    returncode: int | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    turns: int = 0


def _text_from(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_text_from(v) for v in value) if part)
    if isinstance(value, dict):
        for key in ("result", "response", "text", "output", "content", "message"):
            if key in value:
                content = _text_from(value[key])
                if content:
                    return content
        return ""
    return ""


def _usage(obj: Any) -> tuple[int, int]:
    if isinstance(obj, dict):
        usage = obj.get("usage") or obj.get("token_usage") or obj.get("tokens") or obj
        if isinstance(usage, dict):
            incoming = usage.get("input_tokens", usage.get("prompt_tokens", usage.get("input", 0)))
            outgoing = usage.get("output_tokens", usage.get("completion_tokens", usage.get("output", 0)))
            try:
                return max(0, int(incoming or 0)), max(0, int(outgoing or 0))
            except (TypeError, ValueError):
                pass
    return 0, 0


def _json_values(raw: str) -> list[Any]:
    decoder = json.JSONDecoder()
    values: list[Any] = []
    offset = 0
    while offset < len(raw):
        match = re.search(r"[\[{]", raw[offset:])
        if match is None:
            break
        start = offset + match.start()
        try:
            value, end = decoder.raw_decode(raw, start)
            values.append(value)
            offset = end
        except ValueError:
            offset = start + 1
    return values


def parse_cli_output(provider: str, raw: str) -> tuple[str, int, int, int]:
    """Normalize final text and usage from each supported CLI's machine format."""
    stripped = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw).strip()
    values = _json_values(stripped)
    if provider == "codex_cli":
        messages: list[str] = []
        in_tok = out_tok = turns = 0
        for value in values:
            if not isinstance(value, dict):
                continue
            event = value.get("type") or value.get("event")
            item = value.get("item") or {}
            if event in ("item.completed", "item_completed") and isinstance(item, dict):
                if item.get("type") in ("agent_message", "message"):
                    message = _text_from(item.get("text") or item.get("content"))
                    if message:
                        messages.append(message)
            elif event in ("turn.completed", "turn_completed"):
                turns += 1
                a, b = _usage(value.get("usage") or value)
                in_tok += a
                out_tok += b
                message = _text_from(value)
                if message and message not in messages:
                    messages.append(message)
        return (messages[-1] if messages else ""), in_tok, out_tok, turns

    text = ""
    in_tok = out_tok = turns = 0
    for value in values:
        if isinstance(value, list):
            for message in value:
                if isinstance(message, dict) and message.get("role") in ("assistant", "model"):
                    candidate = _text_from(message.get("content") or message.get("text"))
                    if candidate:
                        text = candidate
                        turns += 1
                a, b = _usage(message)
                in_tok += a
                out_tok += b
        elif isinstance(value, dict):
            messages = value.get("messages")
            if isinstance(messages, list):
                candidates = [_text_from(message.get("content") or message.get("text"))
                              for message in messages if isinstance(message, dict)
                              and message.get("role") in ("assistant", "model")]
                candidate = candidates[-1] if candidates else ""
                turns += len(candidates)
            else:
                candidate = _text_from(value)
            if candidate:
                text = candidate
                turns = max(turns, int(value.get("turns", 0) or 0))
            a, b = _usage(value)
            in_tok += a
            out_tok += b
            turns += int(value.get("num_turns", 0) or 0)
    return text or (stripped if not values else ""), in_tok, out_tok, turns


def sanitized_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    """Remove API-key variables so an installed CLI uses its own login session."""
    env = dict(os.environ if source is None else source)
    for key in list(env):
        upper = key.upper()
        if (upper.endswith(("_API_KEY", "_API_TOKEN")) or
            upper in {"API_KEY", "API_TOKEN", "ANTHROPIC_AUTH_TOKEN"} or
            (upper.endswith("_AUTH_TOKEN") and not upper.endswith("_OAUTH_TOKEN"))):
            env.pop(key, None)
    env.pop("CODEMASTER_MCP_TOKEN", None)
    return env


class CliBackend:
    def __init__(self, provider: str, profile: dict[str, Any], model_id: str | None = None,
                 timeout: int = 300, max_turns: int = 10) -> None:
        self.provider = provider
        self._cli_provider = CLI_PROVIDER_ALIASES.get(provider, provider)
        self.profile = profile
        self.command = str(profile["command"])
        self.model_id = model_id
        self.timeout = max(1, timeout)
        self.max_turns = max(1, max_turns)
        self.executable = shutil.which(self.command)
        if not self.executable:
            raise FileNotFoundError(
                f"CLI '{self.command}' non trovata. Installa la CLI e completa il login prima di usare {provider}."
            )

    @property
    def label(self) -> str:
        return f"cli:{self.provider}"

    def _config(self, root: Path, url: str, token: str, tools: list[str]) -> list[str]:
        command = [self.executable or self.command]
        if self._cli_provider == "claude_cli":
            config = root / "claude-mcp.json"
            servers = ({"codemaster": {"type": "http", "url": url,
                        "headers": {"Authorization": f"Bearer {token}"}}} if url else {})
            config.write_text(json.dumps({"mcpServers": servers}, ensure_ascii=False), encoding="utf-8")
            command += ["-p", "--output-format", "json", "--strict-mcp-config", "--mcp-config", str(config),
                        "--setting-sources", "", "--tools", "", "--max-turns", str(self.max_turns)]
            if self.model_id:
                command += ["--model", self.model_id]
            if tools:
                command += ["--allowedTools", *[f"mcp__codemaster__{name}" for name in tools]]
        elif self._cli_provider == "codex_cli":
            command += ["exec", "--json", "--sandbox", "read-only",
                        "--cd", str(root), "--ignore-user-config",
                        "--disable", "shell_tool", "--disable", "unified_exec",
                        "--disable", "browser_use", "--disable", "browser_use_external"]
            if url:
                command += ["-c", f'mcp_servers.codemaster.url="{url}"',
                            "-c", 'mcp_servers.codemaster.bearer_token_env_var="CODEMASTER_MCP_TOKEN"']
            if self.model_id:
                command += ["--model", self.model_id]
            command.append("-")
        elif self._cli_provider == "mistral_vibe_cli":
            allowed = [f"codemaster_{name}" for name in tools]
            sections = ['default_agent = "auto-approve"', "enable_update_checks = false"]
            if allowed:
                sections.append("enabled_tools = [" + ", ".join(json.dumps(name) for name in allowed) + "]")
            else:
                sections.append('disabled_tools = ["*"]')
            if url:
                sections += ["", "[[mcp_servers]]", 'name = "codemaster"',
                             'transport = "streamable-http"', f"url = {json.dumps(url)}",
                             f'headers = {{ Authorization = "Bearer {token}" }}']
            for name in allowed:
                sections.extend(["", f"[tools.{name}]", 'permission = "always"'])
            (root / ".vibe").mkdir()
            (root / ".vibe" / "config.toml").write_text("\n".join(sections) + "\n", encoding="utf-8")
            command += ["--prompt", "--max-turns", str(self.max_turns), "--output", "json", "--trust"]
            if self.model_id:
                command += ["--model", self.model_id]
            command += [arg for name in allowed for arg in ("--enabled-tools", name)]
        elif self._cli_provider == "antigravity_cli":
            agents = root / ".agents"
            agents.mkdir(exist_ok=True)
            plugins_dir = agents / "plugins" / "codemaster"
            plugins_dir.mkdir(parents=True, exist_ok=True)
            (plugins_dir / "plugin.json").write_text(json.dumps({"name": "codemaster"}, ensure_ascii=False), encoding="utf-8")

            servers = ({"codemaster": {"url": url,
                        "headers": {"Authorization": f"Bearer {token}"},
                        "enabledTools": tools}} if url else {})
            config = {"mcpServers": servers}
            (agents / "mcp_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            (plugins_dir / "mcp_config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

            instructions = (
                "# Regole CodeMaster\n"
                "Usa esclusivamente gli strumenti MCP di CodeMaster esposti per consultare la repository.\n"
                "Non accedere a file locali non autorizzati e non invocare comandi shell esterni.\n"
            )
            (root / "AGENTS.md").write_text(instructions, encoding="utf-8")
            rules_dir = plugins_dir / "rules"
            rules_dir.mkdir(parents=True, exist_ok=True)
            (rules_dir / "AGENTS.md").write_text(instructions, encoding="utf-8")

            command += ["-p", "--input-format", "text", "--output-format", "json",
                        "--print-timeout", f"{self.timeout}s", "--sandbox",
                        "--dangerously-skip-permissions"]
            if self.model_id:
                command += ["--model", self.model_id]
        else:
            raise ValueError(f"Provider CLI non supportato: {self.provider}")
        return command

    def _run(self, system: str, task: str, mcp_url: str = "", token: str = "",
             tools: list[str] | None = None, timeout: int | None = None) -> CliResult:
        with tempfile.TemporaryDirectory(prefix="codemaster-cli-", ignore_cleanup_errors=True) as temp:
            root = Path(temp)
            try:
                command = self._config(root, mcp_url, token, tools or [])
            except (OSError, ValueError) as exc:
                return CliResult(error=f"Configurazione CLI non valida: {exc}")
            env = sanitized_environment()
            if token:
                env["CODEMASTER_MCP_TOKEN"] = token
            prompt = task if self._cli_provider == "claude_cli" else f"ISTRUZIONI DI SISTEMA:\n{system}\n\nTASK:\n{task}"
            if self._cli_provider == "claude_cli":
                try:
                    at = command.index("--max-turns")
                    command[at:at] = ["--append-system-prompt", system]
                except ValueError:
                    pass
            kwargs: dict[str, Any] = {"stdin": subprocess.PIPE, "stdout": subprocess.PIPE,
                                      "stderr": subprocess.PIPE, "text": True,
                                      "encoding": "utf-8", "errors": "replace", "env": env,
                                      "cwd": str(root), "shell": False}
            if os.name == "nt":
                kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                if str(command[0]).lower().endswith((".cmd", ".bat")):
                    command = ["cmd.exe", "/d", "/c", *command]
            else:
                kwargs["start_new_session"] = True
            eff_timeout = timeout or self.timeout
            try:
                process = subprocess.Popen(command, **kwargs)
                try:
                    stdout, stderr = process.communicate(prompt, timeout=eff_timeout)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except OSError:
                            process.kill()
                    stdout, stderr = process.communicate()
                    return CliResult(error=f"Timeout CLI dopo {eff_timeout} secondi.",
                                     returncode=process.returncode)
            except (OSError, ValueError) as exc:
                return CliResult(error=f"Impossibile avviare {self.command}: {exc}")

            text, input_tokens, output_tokens, turns = parse_cli_output(self._cli_provider, stdout or "")
            raw_values = _json_values(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", stdout or "").strip())
            has_json_error = any(
                isinstance(v, dict) and (v.get("is_error") is True or v.get("type") in ("error", "item.error"))
                for v in raw_values
            )
            if has_json_error:
                detail = (text or stderr or stdout or "errore restituito dalla CLI").strip()
                detail = re.sub(r"\s+", " ", detail)[:500]
                if re.search(r"unauthori[sz]ed|not logged|log ?in|authentication|quota|rate.?limit|credits?|balance", detail, re.I):
                    detail = f"Autenticazione o quota CLI: {detail}"
                return CliResult(error=f"Errore CLI {self.command}: {detail}", returncode=process.returncode or 1)
            if process.returncode != 0:
                detail = (stderr or text or stdout or "nessun dettaglio dalla CLI").strip()
                detail = re.sub(r"\s+", " ", detail)[:500]
                if re.search(r"unauthori[sz]ed|not logged|log ?in|authentication|quota|rate.?limit|credits?|balance", detail, re.I):
                    detail = f"Autenticazione o quota CLI: {detail}"
                return CliResult(error=f"{self.command} è terminata con codice {process.returncode}: {detail}",
                                  returncode=process.returncode)
            if not text.strip():
                detail = re.sub(r"\s+", " ", (stderr or stdout or "").strip())[:400]
                return CliResult(error=f"{self.command} non ha restituito una risposta utilizzabile. {detail}", returncode=0)
            return CliResult(text=text.strip(), ok=True, returncode=0, in_tokens=input_tokens,
                             out_tokens=output_tokens, turns=turns)

    def complete(self, system: str, task: str, timeout: int | None = None) -> CliResult:
        eff_timeout = timeout if timeout is not None else self.timeout
        return self._run(system, task, timeout=eff_timeout)

    def agentic(self, system: str, task: str, url: str, token: str, tools: list[str],
                timeout: int | None = None) -> CliResult:
        return self._run(system, task, url, token, tools, timeout=timeout)
