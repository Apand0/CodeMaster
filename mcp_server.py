"""MCP bridge that exposes only the repository tools selected for one question."""
from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import secrets
import socket
import threading
import time
from typing import Any


def _json_type(value: Any) -> Any:
    if isinstance(value, list):
        return [_json_type(item) for item in value]
    return {"integer": "integer", "number": "number", "boolean": "boolean",
            "array": "array", "object": "object"}.get(value, "string")


def _tool_schema(tool: Any) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(tool.forward)
    for name, spec in tool.inputs.items():
        entry: dict[str, Any] = {
            "type": _json_type(spec.get("type", "string")),
            "description": str(spec.get("description", "")),
        }
        if spec.get("nullable"):
            entry["type"] = [entry["type"], "null"]
        props[name] = entry
        param = signature.parameters.get(name)
        if param is not None and param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def _validate_tool_paths(tool: Any, arguments: dict[str, Any]) -> None:
    context = getattr(tool, "ctx", None)
    if context is None:
        return
    for key, value in arguments.items():
        if key == "path" and value:
            context.safe_path(str(value))


class _BearerApp:
    """ASGI bearer guard; no network interface other than the loopback listener is opened."""

    def __init__(self, app: Any, token: str) -> None:
        self.app = app
        self.token = token.encode("ascii")

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers", []))
        auth = headers.get(b"authorization", b"")
        supplied = auth[7:].strip() if auth.lower().startswith(b"bearer ") else b""
        if not hmac.compare_digest(supplied, self.token):
            body = b"Unauthorized"
            await send({"type": "http.response.start", "status": 401, "headers": [(b"content-type", b"text/plain"), (b"content-length", str(len(body)).encode())]})
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)


class LocalMCPServer:
    """Long-lived, authenticated streamable-HTTP server bound to an ephemeral loopback port."""

    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(32)
        self.url = ""
        self._tools: dict[str, Any] = {}
        self._trace: list[str] = []
        self._lock = threading.RLock()
        self._server: Any = None
        self._thread: threading.Thread | None = None
        self._socket: socket.socket | None = None

    def set_tools(self, tools: list[Any]) -> None:
        with self._lock:
            self._tools = {tool.name: tool for tool in tools}
            self._trace = []

    def take_trace(self) -> list[str]:
        with self._lock:
            result, self._trace = self._trace, []
            return result

    def start(self) -> tuple[str, str]:
        if self._thread and self._thread.is_alive():
            return self.url, self.token
        try:
            import uvicorn
            import mcp.types as types
            from mcp.server.lowlevel import Server
        except ImportError as exc:
            raise RuntimeError("Installa la dipendenza MCP con: pip install -r requirements.txt") from exc

        async def list_tools(_ctx: Any, _params: Any) -> Any:
            with self._lock:
                tools = list(self._tools.values())
            return types.ListToolsResult(tools=[
                types.Tool(name=tool.name, description=tool.description, input_schema=_tool_schema(tool))
                for tool in tools
            ])

        async def call_tool(_ctx: Any, params: Any) -> Any:
            name = params.name
            with self._lock:
                tool = self._tools.get(name)
            if tool is None:
                return types.CallToolResult(content=[types.TextContent(text=f"Tool non disponibile: {name}")], is_error=True)
            arguments = params.arguments or {}
            try:
                if not isinstance(arguments, dict):
                    raise TypeError("Gli argomenti devono essere un oggetto JSON.")
                bound = inspect.signature(tool.forward).bind(**arguments)
                bound.apply_defaults()
                _validate_tool_paths(tool, arguments)
                result = await asyncio.to_thread(tool, **bound.arguments)
                output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)
                summary = ", ".join(f"{key}={str(value)[:60]!r}" for key, value in arguments.items())
                with self._lock:
                    self._trace.append(f"{name}({summary})")
                return types.CallToolResult(content=[types.TextContent(text=output)])
            except Exception as exc:
                return types.CallToolResult(content=[types.TextContent(text=f"Errore tool {name}: {type(exc).__name__}: {str(exc)[:500]}")], is_error=True)

        # Constructor registration keeps the typed handlers aligned with the installed MCP SDK.
        server = Server("codemaster", version="1.0", on_list_tools=list_tools, on_call_tool=call_tool)
        app = server.streamable_http_app(host="127.0.0.1", streamable_http_path="/mcp",
                                         json_response=True, stateless_http=True)
        protected = _BearerApp(app, self.token)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        sock.setblocking(False)
        port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/mcp"
        config = uvicorn.Config(protected, log_level="critical", access_log=False, lifespan="on")
        self._server = uvicorn.Server(config)
        self._socket = sock
        self._thread = threading.Thread(target=self._server.run, kwargs={"sockets": [sock]}, name="codemaster-mcp", daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 8
        while not self._server.started and self._thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not self._server.started:
            self.close()
            raise RuntimeError("Il server MCP locale non è riuscito ad avviarsi.")
        return self.url, self.token

    def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
