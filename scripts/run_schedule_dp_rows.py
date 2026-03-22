from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mdp_schedule_dp
from schedule import ScheduleMDP


LOG_DIR = ROOT / "logs" / "schedule_dp_repro"
STATUS_FILE = LOG_DIR / "status.tsv"
CURRENT_JOB_FILE = LOG_DIR / "current_job.txt"
RESULTS_JSONL = LOG_DIR / "results.jsonl"

ROWS = [
    (2, 4),
    (2, 5),
    (3, 4),
    (3, 5),
]

DISTS = ["uniform", "asymmetric"]

REG_NUM_SAMPLES = 20
REG_NUM_ITERS = 100
REG_START_SEED = 0
REG_LR = 0.01
REG_STRENGTH = 0.01


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_files() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text("timestamp\tphase\tdist\tmethod\tn\tm\texit_code\n", encoding="utf-8")
    CURRENT_JOB_FILE.write_text("", encoding="utf-8")
    RESULTS_JSONL.write_text("", encoding="utf-8")


def _make_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "total_seconds"):
        return value.total_seconds()
    if hasattr(value, "item"):
        return value.item()
    return value


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_make_jsonable(payload), ensure_ascii=True) + "\n")


def write_status(phase: str, dist: str, method: str, n: int, m: int, exit_code: str = "") -> None:
    with STATUS_FILE.open("a", encoding="utf-8") as fh:
        fh.write(f"{utc_now()}\t{phase}\t{dist}\t{method}\t{n}\t{m}\t{exit_code}\n")


def run_regdp(n: int, m: int, dist: str) -> dict[str, Any]:
    result, _ = mdp_schedule_dp.runtrial(
        num_agents=n,
        num_items=m,
        num_samples=REG_NUM_SAMPLES,
        num_training_iters=REG_NUM_ITERS,
        seed=REG_START_SEED,
        mdp_factory=ScheduleMDP,
        lr=REG_LR,
        reg_strength=REG_STRENGTH,
        dist_type=dist,
        optimize_weights=(dist == "asymmetric"),
        gamma=1.0,
    )
    return result


def run_job(dist: str, n: int, m: int) -> None:
    method = "regdp"
    write_status("START", dist, method, n, m)
    CURRENT_JOB_FILE.write_text(
        f"dist={dist} method={method} n={n} m={m} started_at={utc_now()}\n",
        encoding="utf-8",
    )

    result = run_regdp(n, m, dist)
    append_jsonl(RESULTS_JSONL, result)
    write_status("END", dist, method, n, m, "0")
    CURRENT_JOB_FILE.write_text(
        f"dist={dist} method={method} n={n} m={m} finished_at={utc_now()} exit_code=0\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_files()
    for dist in DISTS:
        for n, m in ROWS:
            run_job(dist, n, m)
    CURRENT_JOB_FILE.write_text("", encoding="utf-8")


if __name__ == "__main__":
    main()
