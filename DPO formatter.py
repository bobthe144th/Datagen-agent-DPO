import json
from typing import Any, Dict, List, Optional


class Formatter:
    """Format agentic sessions to proper format — SFT and DPO modes."""

    @staticmethod
    def format_session(session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format a session into SFT structure."""
        conversation = session_data.get("conversation", [])
        tool_calls = session_data.get("tool_calls", [])
        usage = session_data.get("usage", {})

        formatted_messages = []
        for msg in conversation:
            role = msg.get("role")
            content = msg.get("content", "")
            formatted_msg = {"role": role, "content": content}
            if role == "assistant" and "tool_calls" in msg:
                formatted_msg["tool_calls"] = msg["tool_calls"]
            if role == "tool":
                formatted_msg["tool_call_id"] = msg.get("tool_call_id")
                formatted_msg["name"] = msg.get("name")
            formatted_messages.append(formatted_msg)

        return {
            "prompt": session_data.get("prompt"),
            "messages": formatted_messages,
            "metadata": {
                "session_id": session_data.get("session_id"),
                "turns": session_data.get("turns"),
                "completed": session_data.get("completed", False),
                "tool_calls_count": len(tool_calls),
                "error": session_data.get("error"),
            },
            "usage": usage,
        }

    @staticmethod
    def format_dpo_pair(
        prompt: str,
        chosen_data: Dict[str, Any],
        rejected_data: Dict[str, Any],
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Format two sessions into a DPO pair.

        Output schema:
          {
            "prompt":   str,
            "chosen":   [messages],   # decisive, committed trajectory
            "rejected": [messages],   # hesitant, deliberating trajectory
            "tools":    [...],
            "metadata": {
              "chosen_turns":         int,
              "rejected_turns":       int,
              "chosen_tool_calls":    int,
              "rejected_tool_calls":  int,
              "chosen_completed":     bool,
              "rejected_completed":   bool,
            },
            "usage": {
              "chosen":   {...},
              "rejected": {...},
              "total_tokens": int,
              "cost":         float,
            }
          }
        """

        def extract_messages(session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
            messages = []
            for msg in session_data.get("conversation", []):
                role = msg.get("role")
                content = msg.get("content", "")
                formatted = {"role": role, "content": content}
                if role == "assistant" and "tool_calls" in msg:
                    formatted["tool_calls"] = msg["tool_calls"]
                if role == "tool":
                    formatted["tool_call_id"] = msg.get("tool_call_id")
                    formatted["name"] = msg.get("name")
                messages.append(formatted)
            return messages

        chosen_messages = extract_messages(chosen_data)
        rejected_messages = extract_messages(rejected_data)

        if not chosen_messages or not rejected_messages:
            return None

        chosen_usage = chosen_data.get("usage", {})
        rejected_usage = rejected_data.get("usage", {})

        return {
            "prompt": prompt,
            "chosen": chosen_messages,
            "rejected": rejected_messages,
            "tools": tool_definitions or [],
            "metadata": {
                "chosen_session_id": chosen_data.get("session_id"),
                "rejected_session_id": rejected_data.get("session_id"),
                "chosen_turns": chosen_data.get("turns", 0),
                "rejected_turns": rejected_data.get("turns", 0),
                "chosen_tool_calls": len(chosen_data.get("tool_calls", [])),
                "rejected_tool_calls": len(rejected_data.get("tool_calls", [])),
                "chosen_completed": chosen_data.get("completed", False),
                "rejected_completed": rejected_data.get("completed", False),
            },
            "usage": {
                "chosen": chosen_usage,
                "rejected": rejected_usage,
                "total_tokens": (
                    chosen_usage.get("total_tokens", 0)
                    + rejected_usage.get("total_tokens", 0)
                ),
                "cost": (
                    chosen_usage.get("cost", 0.0)
                    + rejected_usage.get("cost", 0.0)
                ),
            },
        }

    @staticmethod
    def validate_entry(entry: Dict[str, Any]) -> bool:
        """Validate SFT entry structure."""
        if not isinstance(entry, dict):
            return False
        if "messages" not in entry:
            return False
        messages = entry["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return False
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg:
                return False
        return True

    @staticmethod
    def validate_dpo_entry(entry: Dict[str, Any]) -> bool:
        """Validate DPO pair structure."""
        if not isinstance(entry, dict):
            return False
        for key in ("prompt", "chosen", "rejected"):
            if key not in entry:
                return False
        for key in ("chosen", "rejected"):
            msgs = entry[key]
            if not isinstance(msgs, list) or len(msgs) == 0:
                return False
            for msg in msgs:
                if not isinstance(msg, dict) or "role" not in msg:
                    return False
        return True

    @staticmethod
    def to_jsonl_line(entry: Dict[str, Any]) -> str:
        return json.dumps(entry, ensure_ascii=False)
