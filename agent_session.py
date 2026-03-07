"""
agent_session.py — Manages a single agentic session for one prompt.

Key additions over the original:
  1. DockerSandbox lifecycle — a container is started before the first LLM call
     and stopped when session.close() is called.  The sandbox is injected into
     ToolRegistry so run_command routes through it automatically.

  2. Text-based tool call parser — many models on OpenRouter (Minimax, Aurora …)
     do not return native function-calling JSON; instead they embed tool calls
     inline in the assistant's text content.  _parse_tool_calls_from_text()
     detects and normalises them so the rest of the loop works identically
     regardless of which format the model uses.

     Supported text formats
     ──────────────────────
     a) XML tags (most common on non-OpenAI models):
        <tool_call>{"name": "read_file", "arguments": {"file_path": "main.py"}}</tool_call>

     b) Fenced JSON block labelled tool_call / function_call:
        ```tool_call
        {"name": "write_file", "arguments": {"file_path": "out.py", "content": "..."}}
        ```

     c) Bare JSON objects that look like tool invocations (fallback heuristic):
        {"name": "run_command", "arguments": {"command": "python main.py"}}
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tools import ToolRegistry

logger = logging.getLogger("agentic_datagen.session")


# ── tool-call text parsers ────────────────────────────────────────────────────

def _try_parse_json_tool(raw: str) -> Optional[Dict[str, Any]]:
    """
    Try to interpret a JSON string as a tool call.
    Accepts two shapes:
      {"name": ..., "arguments": {...}}
      {"tool": ..., "parameters": {...}}          ← some model variants
    Returns an OpenAI-style tool_call dict or None.
    """
    try:
        obj = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(obj, dict):
        return None

    name = obj.get("name") or obj.get("tool") or obj.get("function")
    args = obj.get("arguments") or obj.get("parameters") or obj.get("args") or {}

    if not name or not isinstance(args, dict):
        return None

    return {
        "id": f"call_text_{hash(raw) & 0xFFFFFF:06x}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(args),
        },
    }


def _parse_tool_calls_from_text(text: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Extract inline tool calls from an assistant text response.

    Returns:
        tool_calls  — list of OpenAI-style tool_call dicts (may be empty)
        clean_text  — the text with tool_call blocks stripped out
    """
    if not text:
        return [], text

    tool_calls: List[Dict[str, Any]] = []
    consumed_spans: List[Tuple[int, int]] = []

    # ── strategy A: <tool_call>…</tool_call> XML blocks ──────────────────────
    xml_pattern = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL | re.IGNORECASE
    )
    for m in xml_pattern.finditer(text):
        tc = _try_parse_json_tool(m.group(1))
        if tc:
            tool_calls.append(tc)
            consumed_spans.append((m.start(), m.end()))

    # ── strategy B: fenced code blocks labelled tool_call / function_call ────
    fence_pattern = re.compile(
        r"```(?:tool_call|function_call|tool)\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )
    for m in fence_pattern.finditer(text):
        # Skip if already captured by XML strategy
        if any(s <= m.start() < e for s, e in consumed_spans):
            continue
        tc = _try_parse_json_tool(m.group(1))
        if tc:
            tool_calls.append(tc)
            consumed_spans.append((m.start(), m.end()))

    # ── strategy C: bare JSON objects that match the tool-call shape ──────────
    # Only runs if nothing found yet (avoids false positives in normal JSON output)
    if not tool_calls:
        bare_pattern = re.compile(r"\{[^{}]*\"(?:name|tool)\"[^{}]*\}", re.DOTALL)
        for m in bare_pattern.finditer(text):
            tc = _try_parse_json_tool(m.group(0))
            if tc:
                tool_calls.append(tc)
                consumed_spans.append((m.start(), m.end()))

    # Strip consumed spans from text (right-to-left to preserve indices)
    clean = text
    for start, end in sorted(consumed_spans, reverse=True):
        clean = clean[:start] + clean[end:]
    clean = clean.strip()

    return tool_calls, clean


# ── session ───────────────────────────────────────────────────────────────────

class AgentSession:
    """Manages a single agentic session for one prompt."""

    def __init__(
        self,
        prompt: str,
        workspace_dir: Path,
        api_config: Dict[str, Any],
        agent_config: Dict[str, Any],
        session_id: str,
    ):
        self.prompt = prompt
        self.workspace_dir = workspace_dir
        self.api_config = api_config
        self.agent_config = agent_config
        self.session_id = session_id

        # Sandbox (started lazily on first run() call)
        self._sandbox = None
        self._sandbox_enabled = agent_config.get("sandbox", {}).get("enabled", True)

        # Tool registry — sandbox will be injected before first LLM call
        self.tool_registry = ToolRegistry(
            workspace_dir,
            config={"api": api_config, "sandbox": agent_config.get("sandbox", {})},
        )

        self.conversation_history: List[Dict[str, Any]] = []
        self.tool_calls_log: List[Dict[str, Any]] = []
        self.http_session = self._create_http_session()

    # ── HTTP session ──────────────────────────────────────────────────────────

    def _create_http_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=self.api_config.get("max_retries", 3),
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    # ── sandbox lifecycle ─────────────────────────────────────────────────────

    def _start_sandbox(self):
        """Start a Docker sandbox and attach it to the tool registry."""
        if not self._sandbox_enabled:
            logger.debug("Session %s: sandbox disabled, using subprocess fallback", self.session_id)
            return
        try:
            from sandbox import DockerSandbox
            self._sandbox = DockerSandbox(
                workspace_dir=self.workspace_dir,
                session_id=self.session_id,
                config={"api": self.api_config, "sandbox": self.agent_config.get("sandbox", {})},
            )
            self.tool_registry.attach_sandbox(self._sandbox)
            logger.info("Session %s: sandbox started (%s)", self.session_id, self._sandbox.container_id[:12])
        except Exception as e:
            logger.warning(
                "Session %s: could not start Docker sandbox (%s) — falling back to subprocess",
                self.session_id, e,
            )
            self._sandbox = None

    def _stop_sandbox(self):
        if self._sandbox is not None:
            self._sandbox.stop()
            self._sandbox = None

    # ── main run loop ─────────────────────────────────────────────────────────

    def run(self) -> Dict[str, Any]:
        """Run the agentic session and return the complete trajectory."""
        self._start_sandbox()

        system_prompt = self.agent_config.get("system_prompt") or (
            "You are a helpful coding assistant with access to file operations and "
            "code analysis tools.\nComplete the user's task thoroughly and efficiently.\n"
            "When given a coding task, create working code files in the workspace."
        )

        max_turns = self.agent_config.get("max_turns") or 50
        enabled_tools = self.agent_config.get("tools_enabled", [])

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.prompt},
        ]

        turn_count = 0
        final_response = None
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0

        while turn_count < max_turns:
            turn_count += 1

            try:
                response = self._call_llm(messages, enabled_tools)
                pt, ct, tc = self._extract_usage(response)
                total_prompt_tokens += pt
                total_completion_tokens += ct
                total_cost += tc
            except Exception as e:
                return self._error_result(
                    f"LLM call failed: {e}", messages, turn_count,
                    total_prompt_tokens, total_completion_tokens, total_cost,
                )

            assistant_message = response.get("choices", [{}])[0].get("message", {})
            if not assistant_message:
                break

            # ── extract reasoning content ─────────────────────────────────────
            reasoning_content = ""
            reasoning_details = assistant_message.get("reasoning_details", [])
            if isinstance(reasoning_details, list):
                for detail in reasoning_details:
                    if isinstance(detail, dict) and detail.get("type") == "reasoning.text":
                        reasoning_content += detail.get("text", "")
            elif isinstance(reasoning_details, str):
                reasoning_content = reasoning_details
            if not reasoning_content:
                reasoning_content = assistant_message.get("reasoning", "")

            if reasoning_content:
                original = assistant_message.get("content") or ""
                assistant_message["content"] = f"<think>{reasoning_content}</think>\n{original}"

            raw_content: str = assistant_message.get("content") or ""

            # ── resolve tool calls ────────────────────────────────────────────
            # Priority: native API tool_calls > text-embedded tool calls
            native_tool_calls: List[Dict[str, Any]] = assistant_message.get("tool_calls") or []

            if native_tool_calls:
                tool_calls = native_tool_calls
                clean_content = raw_content
            else:
                tool_calls, clean_content = _parse_tool_calls_from_text(raw_content)
                if tool_calls:
                    logger.debug(
                        "Session %s turn %d: parsed %d tool call(s) from text",
                        self.session_id, turn_count, len(tool_calls),
                    )

            # ── build clean assistant message for history ─────────────────────
            clean_message: Dict[str, Any] = {
                "role": assistant_message.get("role", "assistant"),
                "content": clean_content if tool_calls else raw_content,
            }
            if tool_calls:
                clean_message["tool_calls"] = tool_calls
            messages.append(clean_message)

            # ── no tool calls → session complete ──────────────────────────────
            if not tool_calls:
                final_response = raw_content
                break

            # ── dispatch tool calls ───────────────────────────────────────────
            for tool_call in tool_calls:
                tool_name = tool_call.get("function", {}).get("name")
                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                tool_id = tool_call.get("id", f"call_{turn_count}")

                try:
                    tool_args = json.loads(tool_args_str)
                except json.JSONDecodeError:
                    tool_args = {}

                logger.debug(
                    "Session %s turn %d: calling %s(%s)",
                    self.session_id, turn_count, tool_name,
                    str(tool_args)[:80],
                )

                tool_result = self.tool_registry.execute_tool(tool_name, tool_args)

                self.tool_calls_log.append({
                    "turn": turn_count,
                    "tool": tool_name,
                    "arguments": tool_args,
                    "result": tool_result,
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result),
                })

        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "turns": turn_count,
            "conversation": messages,
            "tool_calls": self.tool_calls_log,
            "final_response": final_response,
            "completed": final_response is not None,
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
                "cost": total_cost,
            },
        }

    # ── helpers ───────────────────────────────────────────────────────────────

    def _error_result(
        self,
        error: str,
        messages: List[Dict[str, Any]],
        turn_count: int,
        pt: int, ct: int, cost: float,
    ) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "prompt": self.prompt,
            "error": error,
            "turns": turn_count,
            "conversation": messages,
            "tool_calls": self.tool_calls_log,
            "final_response": None,
            "completed": False,
            "usage": {
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": pt + ct,
                "cost": cost,
            },
        }

    def _call_llm(self, messages: List[Dict[str, Any]], enabled_tools: List[str]) -> Dict[str, Any]:
        api_key = self.api_config.get("api_key")
        base_url = self.api_config.get("base_url")
        model = self.api_config.get("model")
        timeout = self.api_config.get("timeout", 120)
        reasoning_effort = self.api_config.get("reasoning_effort")
        temperature = self.api_config.get("temperature")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        body: Dict[str, Any] = {"model": model, "messages": messages}

        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}
        if temperature is not None:
            body["temperature"] = temperature

        if enabled_tools:
            tool_defs = self.tool_registry.get_tool_definitions(enabled_tools)
            if tool_defs:
                body["tools"] = tool_defs
                body["tool_choice"] = "auto"

        response = self.http_session.post(
            base_url, headers=headers, json=body, timeout=timeout
        )

        if response.status_code != 200:
            raise RuntimeError(f"API error {response.status_code}: {response.text[:500]}")

        payload = response.json()
        payload["_headers"] = dict(response.headers)
        return payload

    def _extract_usage(self, response: Dict[str, Any]) -> Tuple[int, int, float]:
        usage = response.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
        completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0

        cost_candidates = (
            response.get("cost"),
            response.get("total_cost"),
            usage.get("cost"),
            usage.get("total_cost"),
            usage.get("total_price"),
        )
        turn_cost = next((float(v) for v in cost_candidates if v is not None), 0.0)

        headers = response.get("_headers") or {}
        if (prompt_tokens == 0 and completion_tokens == 0) or turn_cost == 0.0:
            header_usage = headers.get("x-openrouter-usage")
            if header_usage:
                try:
                    parsed = json.loads(header_usage)
                    prompt_tokens = prompt_tokens or parsed.get("prompt_tokens", 0)
                    completion_tokens = completion_tokens or parsed.get("completion_tokens", 0)
                    if turn_cost == 0.0:
                        turn_cost = parsed.get("cost", 0.0) or parsed.get("total_cost", 0.0)
                except (TypeError, ValueError):
                    pass

            header_cost = headers.get("x-openrouter-cost")
            if header_cost and turn_cost == 0.0:
                try:
                    turn_cost = float(header_cost)
                except ValueError:
                    pass

        return int(prompt_tokens), int(completion_tokens), float(turn_cost)

    def close(self):
        """Clean up HTTP session and sandbox container."""
        self.http_session.close()
        self._stop_sandbox()
