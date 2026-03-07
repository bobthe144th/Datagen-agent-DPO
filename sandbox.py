"""
sandbox.py — Docker-based isolated execution environment.

Each AgentSession gets its own container:
  - The host workspace_dir is bind-mounted at /workspace inside the container
  - Commands run via `docker exec` so file state persists across tool calls
  - The container is removed automatically when the session ends

Requirements:
  - Docker daemon running and accessible to the current user
  - Default image: python:3.12-slim  (override via config: sandbox.image)

Config keys (under `sandbox:` in YAML):
  image:           Docker image to use            (default: python:3.12-slim)
  extra_packages:  List of apt packages to install at startup
  env:             Dict of extra env vars injected into every exec call
  timeout:         Default command timeout in seconds (default: 30)
  memory:          Container memory limit, e.g. "512m"  (default: none)
  cpus:            CPU quota, e.g. "1.0"                (default: none)
"""

import json
import logging
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentic_datagen.sandbox")


class SandboxError(RuntimeError):
    pass


class DockerSandbox:
    """
    Thin wrapper around a Docker container used as a per-session sandbox.

    The host workspace directory is mounted read-write at /workspace so that
    file-system tool calls (read_file, write_file …) operate on the same tree
    as shell commands — no copying required.
    """

    WORKSPACE_MOUNT = "/workspace"

    def __init__(
        self,
        workspace_dir: Path,
        session_id: str,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.workspace_dir = workspace_dir.resolve()
        self.session_id = session_id
        self.config = config or {}

        sandbox_cfg = self.config.get("sandbox", {})
        self.image = sandbox_cfg.get("image", "python:3.12-slim")
        self.default_timeout = int(sandbox_cfg.get("timeout", 30))
        self.extra_env: Dict[str, str] = sandbox_cfg.get("env", {})
        self.memory: Optional[str] = sandbox_cfg.get("memory")
        self.cpus: Optional[str] = sandbox_cfg.get("cpus")
        self.extra_packages: List[str] = sandbox_cfg.get("extra_packages", [])

        self.container_id: Optional[str] = None
        self._start()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _start(self):
        """Pull image if needed and start the container."""
        container_name = f"agentdatagen_{self.session_id}_{uuid.uuid4().hex[:8]}"

        cmd = [
            "docker", "run",
            "--detach",
            "--rm",                                    # auto-remove on stop
            "--name", container_name,
            "--workdir", self.WORKSPACE_MOUNT,
            "--volume", f"{self.workspace_dir}:{self.WORKSPACE_MOUNT}",
            "--network", "none",                       # no outbound network
        ]

        if self.memory:
            cmd += ["--memory", self.memory]
        if self.cpus:
            cmd += ["--cpus", self.cpus]

        for key, val in self.extra_env.items():
            cmd += ["--env", f"{key}={val}"]

        cmd += [self.image, "tail", "-f", "/dev/null"]   # keep container alive

        logger.debug("Starting sandbox container: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise SandboxError(
                f"Failed to start sandbox container: {result.stderr.strip()}"
            )

        self.container_id = result.stdout.strip()
        logger.info("Sandbox started: container=%s", self.container_id[:12])

        if self.extra_packages:
            self._install_packages(self.extra_packages)

    def _install_packages(self, packages: List[str]):
        pkg_list = " ".join(packages)
        logger.info("Installing packages: %s", pkg_list)
        out = self.exec(
            f"apt-get update -qq && apt-get install -y --no-install-recommends {pkg_list}",
            timeout=120,
        )
        logger.debug("Package install output: %s", out[:200])

    def stop(self):
        """Stop and remove the container (idempotent)."""
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "stop", self.container_id],
            capture_output=True,
            timeout=15,
        )
        self.container_id = None
        logger.info("Sandbox stopped for session %s", self.session_id)

    # ── command execution ─────────────────────────────────────────────────────

    def exec(self, command: str, timeout: Optional[int] = None) -> str:
        """
        Run a shell command inside the container and return combined stdout+stderr.

        The command always runs in /workspace (the mounted workspace dir).
        Environment variables set in config.sandbox.env are injected.
        """
        if not self.container_id:
            raise SandboxError("Sandbox container is not running")

        effective_timeout = timeout or self.default_timeout

        # Build env flags
        env_flags: List[str] = []
        for key, val in self.extra_env.items():
            env_flags += ["--env", f"{key}={val}"]

        exec_cmd = (
            ["docker", "exec"]
            + env_flags
            + [self.container_id, "bash", "-c", command]
        )

        logger.debug("Sandbox exec: %s", command[:120])

        try:
            result = subprocess.run(
                exec_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {effective_timeout}s"
        except Exception as e:
            return f"Error executing command: {e}"

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        return output.strip() or "Command executed successfully (no output)"

    # ── context manager support ───────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.stop()
