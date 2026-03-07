import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml

from agent_session import AgentSession
from formatter import Formatter
from system_prompts import get_prompt, ELITE

# ─────────────────────────────────────────────────────────────────────────────
# DPO system prompts (kept here for DPO mode compatibility)
# ─────────────────────────────────────────────────────────────────────────────

CHOSEN_SYSTEM_PROMPT = ELITE  # decisive, commits to tool calls immediately

REJECTED_SYSTEM_PROMPT = """\
You are a coding assistant. You help with coding tasks.

When approaching a task:
- First, think carefully about whether you actually need to use a tool or if you can reason about it directly.
- Consider whether reading the file is really necessary or if you can infer the structure.
- Ask yourself: should I use read_file here, or would it be better to first think through the approach?
- Sometimes it's better to explain your plan before taking action.
- Make sure to reason through all the implications before calling any tools.
- It's important to be thorough in your thinking before committing to an action.\
"""


def _resolve_system_prompt(agent_config: Dict[str, Any]) -> str:
    """
    Resolve the system prompt from agent config.

    Priority:
      1. agent.system_prompt  (inline string — takes precedence)
      2. agent.system_prompt_name  (key into system_prompts.PROMPT_REGISTRY)
      3. Hardcoded default (ELITE)
    """
    inline = agent_config.get("system_prompt")
    if inline:
        return inline

    name = agent_config.get("system_prompt_name")
    if name:
        return get_prompt(name)

    return ELITE


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class AgenticDatasetGenerator:
    """Main orchestrator for agentic dataset generation — SFT and DPO modes."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.formatter = Formatter()

        self.api_key = self._get_api_key()
        self.config["api"]["api_key"] = self.api_key

        self.base_workspace_dir = Path(self.config["workspace"]["base_dir"])
        self.base_workspace_dir.mkdir(parents=True, exist_ok=True)

        self.output_file = Path(self.config["output"]["dataset_file"])
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        if not self.config.get("output", {}).get("append_mode", True):
            if self.output_file.exists():
                self.output_file.unlink()

        error_output_path = self.config.get("output", {}).get("error_dataset_file")
        self.error_output_file = None
        if error_output_path:
            self.error_output_file = Path(error_output_path)
            self.error_output_file.parent.mkdir(parents=True, exist_ok=True)

        self.dpo_mode = self.config.get("dpo", {}).get("enabled", False)

        self.enabled_tools = self.config["agent"].get("tools_enabled", [])
        from tools import ToolRegistry
        temp_registry = ToolRegistry(Path("."), self.config)
        self.tool_definitions = temp_registry.get_tool_definitions(self.enabled_tools)

        # Resolve system prompt once at construction time
        self._system_prompt = _resolve_system_prompt(self.config["agent"])
        self.logger.info(
            "System prompt: %s",
            self.config["agent"].get("system_prompt_name", "inline/default"),
        )

    # ── config / logging / api key ────────────────────────────────────────────

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> logging.Logger:
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        logger = logging.getLogger("agentic_datagen")
        logger.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        if log_config.get("console", True):
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        if "log_file" in log_config:
            fh = logging.FileHandler(log_config["log_file"])
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        return logger

    def _get_api_key(self) -> str:
        api_config = self.config["api"]

        if api_config.get("api_key"):
            k = api_config["api_key"]
            self.logger.info("Using API key from config: %s...%s", k[:4], k[-4:])
            return k

        env_var = api_config.get("api_key_env", "OPENROUTER_API_KEY")
        key = os.getenv(env_var)

        if not key:
            from dotenv import dotenv_values
            key = dotenv_values(".env").get(env_var)

        if not key:
            old_env = Path("old/.env")
            if old_env.exists():
                from dotenv import dotenv_values
                key = dotenv_values(old_env).get(env_var)

        if not key:
            raise ValueError(
                f"Missing API key. Set {env_var} or provide api_key in config."
            )

        self.logger.info("Using API key from %s: %s...%s", env_var, key[:4], key[-4:])
        return key

    # ── prompt loading ────────────────────────────────────────────────────────

    def _load_prompts(self) -> List[str]:
        from utils import load_prompts
        cfg = self.config["prompts"]
        prompts = load_prompts(Path(cfg["source"]))
        if cfg.get("shuffle", False):
            import random
            random.shuffle(prompts)
        limit = cfg.get("limit")
        if limit and limit > 0:
            prompts = prompts[:limit]
        return prompts

    def _load_completed_prompts(self) -> Set[str]:
        completed: Set[str] = set()
        if not self.output_file.exists():
            return completed
        with self.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    p = entry.get("prompt")
                    if p:
                        completed.add(p.strip())
                except json.JSONDecodeError:
                    continue
        return completed

    # ── workspace helpers ─────────────────────────────────────────────────────

    def _create_workspace(self, session_id: str) -> Path:
        ws = self.base_workspace_dir / session_id
        ws.mkdir(parents=True, exist_ok=True)
        return ws

    def _cleanup_workspace(self, workspace_dir: Path):
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    # ── session runner ────────────────────────────────────────────────────────

    def _run_session(
        self,
        prompt: str,
        session_id: str,
        system_prompt: str,
        workspace_dir: Path,
        api_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        agent_config = dict(self.config["agent"])
        agent_config["system_prompt"] = system_prompt

        try:
            session = AgentSession(
                prompt=prompt,
                workspace_dir=workspace_dir,
                api_config=api_config or self.config["api"],
                agent_config=agent_config,
                session_id=session_id,
            )
            data = session.run()
            session.close()
            return data
        except Exception as e:
            self.logger.error("Session %s failed: %s", session_id, e, exc_info=True)
            return None

    # ── SFT mode ──────────────────────────────────────────────────────────────

    def _process_prompt(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        session_id = f"session_{index:06d}"
        workspace_dir = self._create_workspace(session_id)
        self.logger.info("Processing prompt %d: %s…", index, prompt[:80])

        agent_config = dict(self.config["agent"])
        agent_config["system_prompt"] = self._system_prompt   # ← resolved prompt

        try:
            session = AgentSession(
                prompt=prompt,
                workspace_dir=workspace_dir,
                api_config=self.config["api"],
                agent_config=agent_config,
                session_id=session_id,
            )
            session_data = session.run()
            session.close()

            is_error = "error" in session_data
            if is_error:
                self.logger.error("Session error: %s", session_data["error"])
                if not self.config["workspace"].get("preserve_on_error", True):
                    self._cleanup_workspace(workspace_dir)
            else:
                if self.config["workspace"].get("cleanup", True):
                    self._cleanup_workspace(workspace_dir)

            formatted = self.formatter.format_session(session_data)
            formatted["tools"] = self.tool_definitions
            formatted = {
                "prompt": formatted.get("prompt"),
                "tools": formatted.get("tools"),
                "messages": formatted.get("messages"),
                "metadata": formatted.get("metadata"),
                "usage": formatted.get("usage"),
            }

            if not self.formatter.validate_entry(formatted):
                self.logger.error("Entry validation failed for prompt %d", index)
                return None

            return formatted

        except Exception as e:
            self.logger.error("Error processing prompt %d: %s", index, e, exc_info=True)
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info("Preserving workspace: %s", workspace_dir)
            else:
                self._cleanup_workspace(workspace_dir)
            return None

    # ── DPO mode ──────────────────────────────────────────────────────────────

    def _process_prompt_dpo(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        dpo_config = self.config.get("dpo", {})
        chosen_system = dpo_config.get("chosen_system_prompt", CHOSEN_SYSTEM_PROMPT)
        rejected_system = dpo_config.get("rejected_system_prompt", REJECTED_SYSTEM_PROMPT)

        chosen_api = dict(self.config["api"])
        rejected_api = dict(self.config["api"])
        chosen_api["temperature"] = dpo_config.get("chosen_temperature", 0.6)
        rejected_api["temperature"] = dpo_config.get("rejected_temperature", 0.85)

        chosen_ws = self._create_workspace(f"dpo_{index:06d}_chosen")
        rejected_ws = self._create_workspace(f"dpo_{index:06d}_rejected")

        self.logger.info("DPO processing prompt %d: %s…", index, prompt[:80])

        try:
            chosen_data = self._run_session(prompt, f"dpo_{index:06d}_chosen", chosen_system, chosen_ws, chosen_api)
            rejected_data = self._run_session(prompt, f"dpo_{index:06d}_rejected", rejected_system, rejected_ws, rejected_api)

            if not chosen_data or not rejected_data:
                return None

            if "error" in chosen_data or "error" in rejected_data:
                self.logger.warning(
                    "DPO prompt %d: session error — chosen=%s rejected=%s",
                    index, chosen_data.get("error"), rejected_data.get("error"),
                )
                if self.config["workspace"].get("cleanup", True):
                    if "error" not in chosen_data:
                        self._cleanup_workspace(chosen_ws)
                    if "error" not in rejected_data:
                        self._cleanup_workspace(rejected_ws)
                return None

            pair = self.formatter.format_dpo_pair(
                prompt=prompt,
                chosen_data=chosen_data,
                rejected_data=rejected_data,
                tool_definitions=self.tool_definitions,
            )

            if not pair:
                self.logger.error("DPO pair formatting failed for prompt %d", index)
                return None

            if self.config["workspace"].get("cleanup", True):
                self._cleanup_workspace(chosen_ws)
                self._cleanup_workspace(rejected_ws)

            return pair

        except Exception as e:
            self.logger.error("DPO error for prompt %d: %s", index, e, exc_info=True)
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info("Preserving workspaces for prompt %d", index)
            else:
                self._cleanup_workspace(chosen_ws)
                self._cleanup_workspace(rejected_ws)
            return None

    # ── output helpers ────────────────────────────────────────────────────────

    def _append_to_dataset(self, entry: Dict[str, Any]):
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(self.formatter.to_jsonl_line(entry) + "\n")

    def _append_to_error_dataset(self, entry: Dict[str, Any]):
        if not self.error_output_file:
            return
        with self.error_output_file.open("a", encoding="utf-8") as f:
            f.write(self.formatter.to_jsonl_line(entry) + "\n")

    # ── main loop ─────────────────────────────────────────────────────────────

    def generate(self):
        from tqdm import tqdm

        mode = "DPO" if self.dpo_mode else "SFT"
        self.logger.info("Starting dataset generation — mode: %s", mode)

        prompts = self._load_prompts()
        self.logger.info("Loaded %d prompts", len(prompts))

        if self.config["processing"].get("resume", True):
            completed = self._load_completed_prompts()
            self.logger.info("Skipping %d already-completed prompts", len(completed))
            to_process = [(i, p) for i, p in enumerate(prompts) if p.strip() not in completed]
        else:
            to_process = list(enumerate(prompts))

        if not to_process:
            self.logger.info("Nothing to process.")
            return

        self.logger.info("Processing %d prompts", len(to_process))

        self.total_cost = 0.0
        self.total_tokens = 0
        process_fn = self._process_prompt_dpo if self.dpo_mode else self._process_prompt

        pbar = tqdm(total=len(to_process), desc=f"[{mode}] Generating")

        def _handle(entry):
            if not entry:
                pbar.update(1)
                return
            if self.dpo_mode:
                self._append_to_dataset(entry)
            else:
                is_error = entry.get("metadata", {}).get("error")
                turns = entry.get("metadata", {}).get("turns", 0)
                if is_error and turns < 2 and self.error_output_file:
                    self._append_to_error_dataset(entry)
                else:
                    self._append_to_dataset(entry)
            if "usage" in entry:
                self.total_cost += entry["usage"].get("cost", 0.0)
                self.total_tokens += entry["usage"].get("total_tokens", 0)
            pbar.set_postfix(cost=f"${self.total_cost:.4f}", tokens=f"{self.total_tokens:,}")
            pbar.update(1)

        concurrency = self.config["processing"].get("concurrency", 1)

        if concurrency <= 1:
            for index, prompt in to_process:
                _handle(process_fn(prompt, index))
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(process_fn, p, i): (i, p)
                    for i, p in to_process
                }
                for future in as_completed(futures):
                    try:
                        _handle(future.result())
                    except Exception as e:
                        self.logger.error("Future error: %s", e)
                        pbar.update(1)

        pbar.close()
        self.logger.info("%s generation complete", mode)
        self.logger.info("Total cost:   $%.4f", self.total_cost)
        self.logger.info("Total tokens: %s", f"{self.total_tokens:,}")
        self.logger.info("Output:       %s", self.output_file)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Generate agentic SFT or DPO datasets"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to config YAML")
    args = parser.parse_args()

    try:
        AgenticDatasetGenerator(args.config).generate()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
