from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLedger:
    """How many runs updated the state, and the last run that was refused.

    Kept apart from the state file so a refused run can record itself
    without touching the state."""

    successful_runs: int = 0
    last_snapshot_count: int | None = None
    last_failed_run: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "successful_runs": self.successful_runs,
            "last_snapshot_count": self.last_snapshot_count,
            "last_failed_run": self.last_failed_run,
        }


def ledger_path(state_dir: Path, slug: str) -> Path:
    return state_dir / f"{slug}-runs.json"


def load_ledger(path: Path) -> RunLedger:
    if not path.exists():
        return RunLedger()
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunLedger(
        successful_runs=int(data.get("successful_runs", 0)),
        last_snapshot_count=data.get("last_snapshot_count"),
        last_failed_run=data.get("last_failed_run"),
    )


def save_ledger(path: Path, ledger: RunLedger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ledger.to_dict(), indent=2) + "\n", encoding="utf-8")


def record_success(ledger: RunLedger, snapshot_count: int | None) -> RunLedger:
    count = snapshot_count if snapshot_count is not None else ledger.last_snapshot_count
    return replace(ledger, successful_runs=ledger.successful_runs + 1, last_snapshot_count=count)


def record_failure(ledger: RunLedger, run_date: str, reason: str) -> RunLedger:
    return replace(ledger, last_failed_run={"date": run_date, "reason": reason})
