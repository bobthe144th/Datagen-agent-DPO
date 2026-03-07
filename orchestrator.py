"""
orchestrator.py — Run multiple model configs in parallel.

Each config gets its own AgenticDatasetGenerator running in a dedicated
thread.  Progress bars stack vertically (one per model).  A shared lock
serialises JSONL writes within each generator's output file so concurrent
sessions don't corrupt them.

Usage
─────
    python orchestrator.py -c config.minimax.yaml -c config.aurora.yaml

    # Or point at a multi-model manifest (see config.multi.yaml):
    python orchestrator.py --multi config.multi.yaml

The orchestrator adds no overhead beyond what each generator already does —
it simply runs them side-by-side in threads so both models burn through
prompts simultaneously instead of sequentially.
"""

import argparse
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import yaml

logger = logging.getLogger("agentic_datagen.orchestrator")


# ─────────────────────────────────────────────────────────────────────────────
# Worker
# ─────────────────────────────────────────────────────────────────────────────

def _run_generator(config_path: str, label: str) -> dict:
    """
    Instantiate and run one AgenticDatasetGenerator.
    Returns a summary dict; never raises (errors are captured).
    """
    from generator import AgenticDatasetGenerator

    log = logging.getLogger(f"agentic_datagen.{label}")
    log.info("[%s] starting — config: %s", label, config_path)

    try:
        gen = AgenticDatasetGenerator(config_path)
        gen.generate()
        return {
            "label": label,
            "config": config_path,
            "status": "done",
            "cost": getattr(gen, "total_cost", 0.0),
            "tokens": getattr(gen, "total_tokens", 0),
        }
    except Exception as exc:
        log.error("[%s] fatal error: %s", label, exc, exc_info=True)
        return {
            "label": label,
            "config": config_path,
            "status": "error",
            "error": str(exc),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class MultiModelOrchestrator:
    """
    Runs multiple AgenticDatasetGenerators concurrently, one thread per model.

    Parameters
    ----------
    config_paths : list of str
        Paths to individual generator YAML configs.
    labels : list of str | None
        Human-readable labels (defaults to config filename stems).
    """

    def __init__(self, config_paths: List[str], labels: List[str] | None = None):
        if not config_paths:
            raise ValueError("At least one config path is required.")

        self.config_paths = config_paths
        self.labels = labels or [Path(p).stem for p in config_paths]

        if len(self.labels) != len(self.config_paths):
            raise ValueError("labels and config_paths must have the same length.")

    def run(self) -> List[dict]:
        """
        Launch all generators in parallel and block until all finish.
        Returns a list of summary dicts, one per model.
        """
        n = len(self.config_paths)
        logger.info("Launching %d model(s) in parallel: %s", n, self.labels)

        results: List[dict] = []

        with ThreadPoolExecutor(max_workers=n, thread_name_prefix="model") as pool:
            futures = {
                pool.submit(_run_generator, cfg, lbl): lbl
                for cfg, lbl in zip(self.config_paths, self.labels)
            }
            for future in as_completed(futures):
                lbl = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"label": lbl, "status": "error", "error": str(exc)}
                results.append(result)
                _log_result(result)

        _print_summary(results)
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-model manifest loader
# ─────────────────────────────────────────────────────────────────────────────

def load_multi_manifest(manifest_path: str) -> tuple[List[str], List[str]]:
    """
    Parse a multi-model manifest YAML.

    Expected shape:
        models:
          - label: minimax
            config: config.minimax.yaml
          - label: aurora
            config: config.aurora.yaml
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = data.get("models", [])
    if not models:
        raise ValueError(f"No 'models' entries found in {manifest_path}")

    configs = [str(Path(manifest_path).parent / m["config"]) for m in models]
    labels = [m.get("label", Path(m["config"]).stem) for m in models]
    return configs, labels


# ─────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ─────────────────────────────────────────────────────────────────────────────

def _log_result(result: dict):
    if result["status"] == "done":
        logger.info(
            "[%s] ✓ done — cost $%.4f  tokens %s",
            result["label"],
            result.get("cost", 0.0),
            f"{result.get('tokens', 0):,}",
        )
    else:
        logger.error("[%s] ✗ error: %s", result["label"], result.get("error", "unknown"))


def _print_summary(results: List[dict]):
    print("\n" + "═" * 60)
    print("  MULTI-MODEL RUN SUMMARY")
    print("═" * 60)
    total_cost = 0.0
    total_tokens = 0
    for r in results:
        status = "✓" if r["status"] == "done" else "✗"
        cost = r.get("cost", 0.0)
        tokens = r.get("tokens", 0)
        total_cost += cost
        total_tokens += tokens
        line = f"  {status}  {r['label']:<20}  ${cost:.4f}   {tokens:>12,} tokens"
        if r["status"] == "error":
            line += f"   ERROR: {r.get('error', '')[:40]}"
        print(line)
    print("─" * 60)
    print(f"  {'TOTAL':<22}  ${total_cost:.4f}   {total_tokens:>12,} tokens")
    print("═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-35s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    _setup_logging()

    parser = argparse.ArgumentParser(
        description="Run multiple agentic dataset generators in parallel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # two configs side-by-side
  python orchestrator.py -c config.minimax.yaml -c config.aurora.yaml

  # manifest file
  python orchestrator.py --multi config.multi.yaml
        """,
    )
    parser.add_argument(
        "-c", "--config",
        dest="configs",
        action="append",
        metavar="CONFIG",
        help="Path to a generator config YAML (repeat for each model).",
    )
    parser.add_argument(
        "--multi",
        metavar="MANIFEST",
        help="Path to a multi-model manifest YAML.",
    )
    parser.add_argument(
        "--label",
        dest="labels",
        action="append",
        metavar="LABEL",
        help="Optional label for each -c config (same order, repeat to match).",
    )
    args = parser.parse_args()

    if args.multi:
        configs, labels = load_multi_manifest(args.multi)
    elif args.configs:
        configs = args.configs
        labels = args.labels or None
    else:
        parser.error("Provide at least one -c/--config or a --multi manifest.")

    orchestrator = MultiModelOrchestrator(configs, labels)
    results = orchestrator.run()

    # Exit non-zero if any model errored
    if any(r["status"] == "error" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
