import json
import os
import subprocess
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional


class Context7Client:
    """Minimal MCP HTTP client for the Context7 documentation server."""

    MCP_URL = "https://mcp.context7.com/mcp"

    def __init__(self, timeout: int = 30):
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._http = requests.Session()

    def _initialize(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "agentic-datagen", "version": "1.0"},
            },
        }
        resp = self._http.post(
            self.MCP_URL,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        self._session_id = resp.headers.get("Mcp-Session-Id")

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        if not self._session_id:
            self._initialize()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        payload = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        }
        resp = self._http.post(
            self.MCP_URL, json=payload, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()

        body = resp.text
        if body.startswith("data:"):
            for line in body.splitlines():
                if line.startswith("data:"):
                    body = line[len("data:"):].strip()
                    break

        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return body

        result = data.get("result", {})
        content = result.get("content", [])
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(parts) if parts else str(result)
        return str(result)


class ToolRegistry:
    """
    Registry of available tools for the agentic system.

    File-system operations run on the host workspace_dir (which is bind-mounted
    into the sandbox container at /workspace).  Shell commands are routed through
    the DockerSandbox when one is attached; otherwise fall back to subprocess.
    """

    def __init__(
        self,
        workspace_dir: Path,
        config: Optional[Dict[str, Any]] = None,
        sandbox=None,
    ):
        self.workspace_dir = workspace_dir
        self.config = config or {}
        self.sandbox = sandbox
        self._context7: Optional[Context7Client] = None

        self.tools = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "edit_file": self.edit_file,
            "list_directory": self.list_directory,
            "search_code": self.search_code,
            "run_command": self.run_command,
            "web_search": self.web_search,
            "resolve_library_id": self.resolve_library_id,
            "get_library_docs": self.get_library_docs,
        }

    def attach_sandbox(self, sandbox) -> None:
        """Attach a DockerSandbox after construction (called by AgentSession)."""
        self.sandbox = sandbox

    @property
    def context7(self) -> Context7Client:
        if self._context7 is None:
            timeout = self.config.get("api", {}).get("timeout", 30)
            self._context7 = Context7Client(timeout=timeout)
        return self._context7

    # ── tool definitions ──────────────────────────────────────────────────────

    def get_tool_definitions(self, enabled_tools: List[str]) -> List[Dict[str, Any]]:
        definitions = []

        if "read_file" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path relative to workspace root"}
                        },
                        "required": ["file_path"],
                    },
                },
            })

        if "write_file" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in the workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path relative to workspace root"},
                            "content": {"type": "string", "description": "Content to write"},
                        },
                        "required": ["file_path", "content"],
                    },
                },
            })

        if "edit_file" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "edit_file",
                    "description": "Edit a file by replacing old_text with new_text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path relative to workspace root"},
                            "old_text": {"type": "string", "description": "Exact text to replace"},
                            "new_text": {"type": "string", "description": "Replacement text"},
                        },
                        "required": ["file_path", "old_text", "new_text"],
                    },
                },
            })

        if "list_directory" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a workspace path",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "dir_path": {"type": "string", "description": "Directory path relative to workspace root (empty for root)"}
                        },
                        "required": [],
                    },
                },
            })

        if "search_code" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search for text patterns in workspace files",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string", "description": "Text pattern to search for"},
                            "file_pattern": {"type": "string", "description": "Optional file glob (e.g. '*.py')"},
                        },
                        "required": ["pattern"],
                    },
                },
            })

        if "run_command" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": (
                        "Execute a shell command inside the isolated sandbox container. "
                        "Working directory is /workspace. Stdout and stderr are returned."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute"},
                            "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
                        },
                        "required": ["command"],
                    },
                },
            })

        if "web_search" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                },
            })

        if "resolve_library_id" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "resolve_library_id",
                    "description": (
                        "Resolve a library name to its Context7-compatible ID. "
                        "Call this before get_library_docs."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "library_name": {"type": "string", "description": "Library name (e.g. 'fastapi', 'react')"}
                        },
                        "required": ["library_name"],
                    },
                },
            })

        if "get_library_docs" in enabled_tools:
            definitions.append({
                "type": "function",
                "function": {
                    "name": "get_library_docs",
                    "description": "Fetch current docs for a library via Context7. Use resolve_library_id first.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "context7_compatible_library_id": {"type": "string", "description": "Context7 library ID"},
                            "topic": {"type": "string", "description": "Optional topic to focus on"},
                            "tokens": {"type": "integer", "description": "Max tokens to return (default: 5000)"},
                        },
                        "required": ["context7_compatible_library_id"],
                    },
                },
            })

        return definitions

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            result = self.tools[tool_name](**arguments)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ── filesystem tools ──────────────────────────────────────────────────────

    def _safe_path(self, file_path: str) -> Path:
        full = (self.workspace_dir / file_path).resolve()
        if not str(full).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")
        return full

    def read_file(self, file_path: str) -> str:
        full = self._safe_path(file_path)
        if not full.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        return full.read_text(encoding="utf-8")

    def write_file(self, file_path: str, content: str) -> str:
        full = self._safe_path(file_path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {file_path}"

    def edit_file(self, file_path: str, old_text: str, new_text: str) -> str:
        content = self.read_file(file_path)
        if old_text not in content:
            raise ValueError(f"Text not found in {file_path}: {old_text[:50]!r}")
        self.write_file(file_path, content.replace(old_text, new_text, 1))
        return f"Edited {file_path}"

    def list_directory(self, dir_path: str = "") -> List[str]:
        full = (self.workspace_dir / dir_path).resolve() if dir_path else self.workspace_dir.resolve()
        if not str(full).startswith(str(self.workspace_dir.resolve())):
            raise PermissionError("Access denied: path outside workspace")
        if not full.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        items = []
        for item in sorted(full.iterdir()):
            rel = item.relative_to(self.workspace_dir)
            items.append(f"{rel}/" if item.is_dir() else f"{rel} ({item.stat().st_size} bytes)")
        return items

    def search_code(self, pattern: str, file_pattern: Optional[str] = None) -> List[Dict[str, Any]]:
        files = (
            list(self.workspace_dir.glob(f"**/{file_pattern}"))
            if file_pattern
            else [f for f in self.workspace_dir.rglob("*") if f.is_file()]
        )
        results = []
        for fp in files:
            try:
                for lineno, line in enumerate(fp.read_text(encoding="utf-8").splitlines(), 1):
                    if pattern.lower() in line.lower():
                        results.append({
                            "file": str(fp.relative_to(self.workspace_dir)),
                            "line": lineno,
                            "content": line.strip(),
                        })
            except Exception:
                continue
        return results[:50]

    # ── run_command — sandbox if available, subprocess fallback ───────────────

    def run_command(self, command: str, timeout: Optional[int] = None) -> str:
        if self.sandbox is not None:
            return self.sandbox.exec(command, timeout=timeout)

        # Fallback: plain subprocess (no Docker)
        effective_timeout = timeout or 30
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output.strip() or "Command executed successfully (no output)"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {effective_timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

    # ── web search ────────────────────────────────────────────────────────────

    def web_search(self, query: str) -> str:
        searxng_url = self.config.get("api", {}).get("searxng_url") or os.getenv(
            "SEARXNG_URL", "http://localhost:your-searxng-port"
        )
        try:
            resp = requests.get(
                f"{searxng_url}/search",
                params={"q": query, "format": "json"},
                timeout=10,
            )
            resp.raise_for_status()
            results = [
                f"Title: {r.get('title')}\nURL: {r.get('url')}\nSnippet: {r.get('content')}\n"
                for r in resp.json().get("results", [])[:5]
            ]
            return "\n".join(results) if results else "No results found."
        except Exception as e:
            return f"Error performing web search: {e}"

    # ── Context7 MCP ──────────────────────────────────────────────────────────

    def resolve_library_id(self, library_name: str) -> str:
        return self.context7.call_tool("resolve-library-id", {"libraryName": library_name})

    def get_library_docs(
        self,
        context7_compatible_library_id: str,
        topic: Optional[str] = None,
        tokens: Optional[int] = None,
    ) -> str:
        args: Dict[str, Any] = {
            "context7CompatibleLibraryId": context7_compatible_library_id
        }
        if topic:
            args["topic"] = topic
        if tokens:
            args["tokens"] = tokens
        return self.context7.call_tool("get-library-docs", args)
