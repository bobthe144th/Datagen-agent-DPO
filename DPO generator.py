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

# ─────────────────────────────────────────────────────────────────────────────
# DPO system prompts
# ─────────────────────────────────────────────────────────────────────────────

CHOSEN_SYSTEM_PROMPT = """You are an elite agentic coding assistant. You operate with decisive confidence.

Your rules:
- When you identify what needs to be done, call the tool IMMEDIATELY. No preamble.
- Never debate with yourself about whether to call a tool. If it will help, call it.
- Read files before editing them. Write files after generating code. Run commands to verify.
- If a tool call fails, diagnose the error and retry with a fix — do not give up.
- Think in brief <think> tags only when genuinely needed for complex reasoning, never for simple tool decisions.
- Produce clean, idiomatic, well-structured code with proper error handling.
- Complete the task fully. Do not stop after a single step."""

REJECTED_SYSTEM_PROMPT = """You are a coding assistant. You help with coding tasks.

When approaching a task:
- First, think carefully about whether you actually need to use a tool or if you can reason about it directly.
- Consider whether reading the file is really necessary or if you can infer the structure.
- Ask yourself: should I use read_file here, or would it be better to first think through the approach?
- Sometimes it's better to explain your plan before taking action.
- Make sure to reason through all the implications before calling any tools.
- It's important to be thorough in your thinking before committing to an action."""


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

        # DPO mode flag
        self.dpo_mode = self.config.get("dpo", {}).get("enabled", False)

        self.enabled_tools = self.config["agent"].get("tools_enabled", [])
        from tools import ToolRegistry
        temp_registry = ToolRegistry(Path("."), self.config)
        self.tool_definitions = temp_registry.get_tool_definitions(self.enabled_tools)

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
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        if "log_file" in log_config:
            file_handler = logging.FileHandler(log_config["log_file"])
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

    def _get_api_key(self) -> str:
        api_config = self.config["api"]

        if "api_key" in api_config and api_config["api_key"]:
            api_key = api_config["api_key"]
            self.logger.info(f"Using API Key from config: {api_key[:4]}...{api_key[-4:]}")
            return api_key

        env_var = api_config.get("api_key_env", "OPENROUTER_API_KEY")
        api_key = os.getenv(env_var)

        if not api_key:
            from dotenv import dotenv_values
            api_key = dotenv_values(".env").get(env_var)

        if not api_key:
            old_env = Path("old/.env")
            if old_env.exists():
                from dotenv import dotenv_values
                api_key = dotenv_values(old_env).get(env_var)

        if not api_key:
            raise ValueError(
                f"Missing API key. Provide 'api_key' in config or set {env_var} environment variable."
            )

        self.logger.info(f"Using API Key from {env_var}: {api_key[:4]}...{api_key[-4:]}")
        return api_key

    def _load_prompts(self) -> List[str]:
        from utils import load_prompts
        prompts_config = self.config["prompts"]
        source_path = Path(prompts_config["source"])
        prompts = load_prompts(source_path)
        if prompts_config.get("shuffle", False):
            import random
            random.shuffle(prompts)
        limit = prompts_config.get("limit")
        if limit and limit > 0:
            prompts = prompts[:limit]
        return prompts

    def _load_completed_prompts(self) -> Set[str]:
        completed = set()
        if not self.output_file.exists():
            return completed
        with self.output_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    prompt = entry.get("prompt")
                    if prompt:
                        completed.add(prompt.strip())
                except json.JSONDecodeError:
                    continue
        return completed

    def _create_workspace(self, session_id: str) -> Path:
        workspace_dir = self.base_workspace_dir / session_id
        workspace_dir.mkdir(parents=True, exist_ok=True)
        return workspace_dir

    def _cleanup_workspace(self, workspace_dir: Path):
        if workspace_dir.exists():
            shutil.rmtree(workspace_dir)

    def _run_session(
        self,
        prompt: str,
        session_id: str,
        system_prompt: str,
        workspace_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Run a single agent session with the given system prompt."""
        agent_config = dict(self.config["agent"])
        agent_config["system_prompt"] = system_prompt

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
            return session_data
        except Exception as e:
            self.logger.error(f"Session {session_id} failed: {e}", exc_info=True)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # SFT mode (original behaviour)
    # ─────────────────────────────────────────────────────────────────────────

    def _process_prompt(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        """Process a single prompt for SFT — original behaviour."""
        session_id = f"session_{index:06d}"
        workspace_dir = self._create_workspace(session_id)
        self.logger.info(f"Processing prompt {index}: {prompt[:80]}...")

        try:
            session = AgentSession(
                prompt=prompt,
                workspace_dir=workspace_dir,
                api_config=self.config["api"],
                agent_config=self.config["agent"],
                session_id=session_id,
            )
            session_data = session.run()
            session.close()

            is_error = "error" in session_data
            if is_error:
                self.logger.error(f"Session error: {session_data['error']}")
                if self.config["workspace"].get("preserve_on_error", True):
                    self.logger.info(f"Preserving workspace: {workspace_dir}")
                else:
                    self._cleanup_workspace(workspace_dir)

            formatted_entry = self.formatter.format_session(session_data)
            formatted_entry["tools"] = self.tool_definitions
            formatted_entry = {
                "prompt": formatted_entry.get("prompt"),
                "tools": formatted_entry.get("tools"),
                "messages": formatted_entry.get("messages"),
                "metadata": formatted_entry.get("metadata"),
                "usage": formatted_entry.get("usage"),
            }

            if not self.formatter.validate_entry(formatted_entry):
                self.logger.error("Entry validation failed")
                return None

            if self.config["workspace"].get("cleanup", True) and not is_error:
                self._cleanup_workspace(workspace_dir)

            return formatted_entry

        except Exception as e:
            self.logger.error(f"Error processing prompt: {e}", exc_info=True)
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info(f"Preserving workspace: {workspace_dir}")
            else:
                self._cleanup_workspace(workspace_dir)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # DPO mode
    # ─────────────────────────────────────────────────────────────────────────

    def _process_prompt_dpo(self, prompt: str, index: int) -> Optional[Dict[str, Any]]:
        """
        Process a single prompt for DPO.

        Runs two isolated sessions:
          - chosen:   decisive system prompt, low temperature
          - rejected: hesitant system prompt, higher temperature

        Each session gets its own workspace so tool calls don't interfere.
        Returns a DPO pair dict or None if either session fails.
        """
        dpo_config = self.config.get("dpo", {})
        chosen_system = dpo_config.get("chosen_system_prompt", CHOSEN_SYSTEM_PROMPT)
        rejected_system = dpo_config.get("rejected_system_prompt", REJECTED_SYSTEM_PROMPT)

        # Override temperatures per session by temporarily patching api_config
        chosen_api_config = dict(self.config["api"])
        rejected_api_config = dict(self.config["api"])
        chosen_api_config["temperature"] = dpo_config.get("chosen_temperature", 0.6)
        rejected_api_config["temperature"] = dpo_config.get("rejected_temperature", 0.85)

        chosen_workspace = self._create_workspace(f"dpo_{index:06d}_chosen")
        rejected_workspace = self._create_workspace(f"dpo_{index:06d}_rejected")

        self.logger.info(f"DPO processing prompt {index}: {prompt[:80]}...")

        try:
            # Run chosen session
            chosen_agent_config = dict(self.config["agent"])
            chosen_agent_config["system_prompt"] = chosen_system
            chosen_session = AgentSession(
                prompt=prompt,
                workspace_dir=chosen_workspace,
                api_config=chosen_api_config,
                agent_config=chosen_agent_config,
                session_id=f"dpo_{index:06d}_chosen",
            )
            chosen_data = chosen_session.run()
            chosen_session.close()

            # Run rejected session
            rejected_agent_config = dict(self.config["agent"])
            rejected_agent_config["system_prompt"] = rejected_system
            rejected_session = AgentSession(
                prompt=prompt,
                workspace_dir=rejected_workspace,
                api_config=rejected_api_config,
                agent_config=rejected_agent_config,
                session_id=f"dpo_{index:06d}_rejected",
            )
            rejected_data = rejected_session.run()
            rejected_session.close()

            # Both sessions must complete successfully
            if "error" in chosen_data or "error" in rejected_data:
                self.logger.warning(
                    f"DPO prompt {index}: session error — "
                    f"chosen_error={chosen_data.get('error')} "
                    f"rejected_error={rejected_data.get('error')}"
                )
                # Preserve workspaces on error
                if not self.config["workspace"].get("cleanup", True):
                    pass
                else:
                    if "error" not in chosen_data:
                        self._cleanup_workspace(chosen_workspace)
                    if "error" not in rejected_data:
                        self._cleanup_workspace(rejected_workspace)
                return None

            dpo_pair = self.formatter.format_dpo_pair(
                prompt=prompt,
                chosen_data=chosen_data,
                rejected_data=rejected_data,
                tool_definitions=self.tool_definitions,
            )

            if not dpo_pair:
                self.logger.error(f"DPO pair formatting failed for prompt {index}")
                return None

            # Cleanup workspaces
            if self.config["workspace"].get("cleanup", True):
                self._cleanup_workspace(chosen_workspace)
                self._cleanup_workspace(rejected_workspace)

            return dpo_pair

        except Exception as e:
            self.logger.error(f"DPO error for prompt {index}: {e}", exc_info=True)
            if self.config["workspace"].get("preserve_on_error", True):
                self.logger.info(f"Preserving workspaces for prompt {index}")
            else:
                self._cleanup_workspace(chosen_workspace)
                self._cleanup_workspace(rejected_workspace)
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Output helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _append_to_dataset(self, entry: Dict[str, Any]):
        jsonl_line = self.formatter.to_jsonl_line(entry)
        with self.output_file.open("a", encoding="utf-8") as f:
            f.write(jsonl_line + "\n")

    def _append_to_error_dataset(self, entry: Dict[str, Any]):
        if not self.error_output_file:
            return
        jsonl_line = self.formatter.to_jsonl_line(entry)
        with self.error_output_file.open("a", encoding="utf-8") as f:
            f.write(jsonl_line + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Main loop
    # ─────────────────────────────────────────────────────────────────────────

    def generate(self):
        from tqdm import tqdm

        mode = "DPO" if self.dpo_mode else "SFT"
        self.logger.info(f"Starting agentic dataset generation — mode: {mode}")

        prompts = self._load_prompts()
        self.logger.info(f"Loaded {len(prompts)} prompts")

        if self.config["processing"].get("resume", True):
            completed = self._load_completed_prompts()
            self.logger.info(f"Found {len(completed)} completed prompts")
            prompts_to_process = [
                (i, p) for i, p in enumerate(prompts) if p.strip() not in completed
            ]
        else:
            prompts_to_process = list(enumerate(prompts))

        if not prompts_to_process:
            self.logger.info("No prompts to process")
            return

        self.logger.info(f"Processing {len(prompts_to_process)} prompts")

        concurrency = self.config["processing"].get("concurrency", 1)
        self.total_cost = 0.0
        self.total_tokens = 0

        # In DPO mode each prompt generates 2x the API calls — note that in the bar
        pbar = tqdm(
            total=len(prompts_to_process),
            desc=f"Generating {mode} Dataset",
        )

        def update_pbar(entry):
            if entry and "usage" in entry:
                self.total_cost += entry["usage"].get("cost", 0.0)
                self.total_tokens += entry["usage"].get("total_tokens", 0)
            pbar.set_postfix({
                "cost": f"${self.total_cost:.4f}",
                "tokens": f"{self.total_tokens:,}",
            })
            pbar.update(1)

        # Select processing function based on mode
        process_fn = self._process_prompt_dpo if self.dpo_mode else self._process_prompt

        if concurrency <= 1:
            for index, prompt in prompts_to_process:
                entry = process_fn(prompt, index)
                if entry:
                    if self.dpo_mode:
                        # DPO pairs always go to main dataset — no error routing
                        self._append_to_dataset(entry)
                    else:
                        is_error = entry.get("metadata", {}).get("error")
                        turns = entry.get("metadata", {}).get("turns", 0)
                        if is_error and turns < 2 and self.error_output_file:
                            self._append_to_error_dataset(entry)
                        else:
                            self._append_to_dataset(entry)
                    update_pbar(entry)
                else:
                    pbar.update(1)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {
                    executor.submit(process_fn, prompt, index): (index, prompt)
                    for index, prompt in prompts_to_process
                }
                for future in as_completed(futures):
                    try:
                        entry = future.result()
                        if entry:
                            if self.dpo_mode:
                                self._append_to_dataset(entry)
                            else:
                                is_error = entry.get("metadata", {}).get("error")
                                turns = entry.get("metadata", {}).get("turns", 0)
                                if is_error and turns < 2 and self.error_output_file:
                                    self._append_to_error_dataset(entry)
                                else:
                                    self._append_to_dataset(entry)
                            update_pbar(entry)
                    except Exception as e:
                        self.logger.error(f"Error in future: {e}")
                        pbar.update(1)

        pbar.close()
        self.logger.info(f"{mode} dataset generation complete")
        self.logger.info(f"Total Cost: ${self.total_cost:.4f}")
        self.logger.info(f"Total Tokens: {self.total_tokens:,}")
        self.logger.info(f"Output saved to: {self.output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate agentic SFT or DPO datasets with tool-calling capabilities"
    )
    parser.add_argument("-c", "--config", required=True, help="Path to configuration YAML file")
    args = parser.parse_args()

    try:
        generator = AgenticDatasetGenerator(args.config)
        generator.generate()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
