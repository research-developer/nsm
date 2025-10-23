"""
Baseline tracking system for parallel experiments.

Maintains a JSONL file with indexed references to each branch's metrics,
preventing overwrites and enabling comparison across worktrees.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import subprocess


class BaselineTracker:
    """
    Track baseline metrics across branches and experiments.

    Usage:
        tracker = BaselineTracker()
        tracker.record_baseline(
            experiment="physics_safety_factor",
            metrics={"accuracy": 0.55, "q_neural": 2.3},
            config={...}
        )

        # Compare to baseline
        baseline = tracker.get_baseline(branch="main")
        improvement = metrics["accuracy"] - baseline["metrics"]["accuracy"]
    """

    def __init__(self, baselines_file: Optional[str] = None):
        """
        Initialize tracker with JSONL file path.

        Args:
            baselines_file: Path to JSONL file (default: from env or repo root)
        """
        if baselines_file is None:
            # Try environment variable first
            baselines_file = os.getenv("NSM_BASELINES_FILE")

            if baselines_file is None:
                # Fall back to repo root
                repo_root = os.getenv("NSM_REPO_ROOT", os.getcwd())
                baselines_file = os.path.join(repo_root, "baselines.jsonl")

        self.baselines_file = Path(baselines_file)

        # Create file if it doesn't exist
        if not self.baselines_file.exists():
            self.baselines_file.parent.mkdir(parents=True, exist_ok=True)
            self.baselines_file.touch()

    def _get_git_info(self) -> Dict[str, str]:
        """Get current git branch and commit."""
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()

            return {"branch": branch, "commit": commit}
        except:
            return {"branch": "unknown", "commit": "unknown"}

    def record_baseline(
        self,
        experiment: str,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        notes: str = "",
        branch: Optional[str] = None,
        commit: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Record a new baseline entry.

        Args:
            experiment: Experiment name/identifier
            metrics: Dictionary of metric values
            config: Experiment configuration
            notes: Optional notes about this run
            branch: Git branch (auto-detected if None)
            commit: Git commit (auto-detected if None)

        Returns:
            The recorded baseline entry
        """
        # Get git info if not provided
        if branch is None or commit is None:
            git_info = self._get_git_info()
            branch = branch or git_info["branch"]
            commit = commit or git_info["commit"]

        # Create baseline entry
        entry = {
            "branch": branch,
            "commit": commit,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "experiment": experiment,
            "metrics": metrics,
            "config": config,
            "notes": notes
        }

        # Append to JSONL file
        with open(self.baselines_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        return entry

    def get_baseline(
        self,
        branch: Optional[str] = None,
        experiment: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get most recent baseline for branch/experiment.

        Args:
            branch: Filter by branch (current branch if None)
            experiment: Filter by experiment name

        Returns:
            Most recent matching baseline entry, or None
        """
        if branch is None:
            branch = self._get_git_info()["branch"]

        # Read all entries
        entries = self.load_all()

        # Filter
        filtered = [
            e for e in entries
            if (branch is None or e["branch"] == branch) and
               (experiment is None or e["experiment"] == experiment)
        ]

        # Return most recent
        if filtered:
            return filtered[-1]
        return None

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all baseline entries."""
        if not self.baselines_file.exists():
            return []

        entries = []
        with open(self.baselines_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries

    def compare(
        self,
        metrics: Dict[str, float],
        baseline_branch: str = "main",
        baseline_experiment: Optional[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare metrics to baseline.

        Args:
            metrics: Current metrics
            baseline_branch: Branch to compare against
            baseline_experiment: Specific experiment to compare against

        Returns:
            Dictionary with 'baseline', 'current', 'delta', 'percent_change'
        """
        baseline = self.get_baseline(
            branch=baseline_branch,
            experiment=baseline_experiment
        )

        if baseline is None:
            raise ValueError(f"No baseline found for branch={baseline_branch}, experiment={baseline_experiment}")

        baseline_metrics = baseline["metrics"]

        # Compute deltas
        delta = {}
        percent_change = {}

        for key in metrics:
            if key in baseline_metrics:
                baseline_val = baseline_metrics[key]
                current_val = metrics[key]

                if baseline_val is not None and current_val is not None:
                    delta[key] = current_val - baseline_val

                    if baseline_val != 0:
                        percent_change[key] = (delta[key] / abs(baseline_val)) * 100
                    else:
                        percent_change[key] = float('inf') if delta[key] > 0 else 0

        return {
            "baseline": baseline_metrics,
            "current": metrics,
            "delta": delta,
            "percent_change": percent_change
        }


# Export
__all__ = ['BaselineTracker']
