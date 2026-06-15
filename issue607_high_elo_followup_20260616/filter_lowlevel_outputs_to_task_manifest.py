"""Filter low-level best-move outputs to a smaller task manifest.

Use this when a broader collection was already run and a later task manifest
selects only a subset of those nodes. Existing level CSV rows are copied so the
collector can resume without recalculating completed hints.
"""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = SCRIPT_DIR / "reports" / "full_loss_bucket_lowlevel_20260616"
DEFAULT_TASK_MANIFEST = SCRIPT_DIR / "reports" / "loss_bucket_testset_recent5_first_20260616" / "full_supplement_task_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "full_supplement_lowlevel_20260616"


def parse_levels(text: str) -> List[int]:
    levels: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            levels.extend(range(int(start_s), int(end_s) + 1))
        else:
            levels.append(int(part))
    return sorted(set(levels))


def read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter existing low-level CSV outputs to a selected task manifest.")
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--levels", type=parse_levels, default=parse_levels("1-10"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {args.output_dir}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    task_rows = read_csv(args.task_manifest)
    selected_task_uids = {row["task_uid"] for row in task_rows}
    write_csv(args.output_dir / "task_manifest.csv", task_rows[0].keys(), task_rows)

    level_summary = []
    for level in args.levels:
        source_path = args.source_dir / f"level{level}_bestmoves.csv"
        output_path = args.output_dir / f"level{level}_bestmoves.csv"
        if not source_path.exists():
            level_summary.append({"level": level, "source_rows": 0, "copied_rows": 0, "missing_rows": len(selected_task_uids), "source_exists": 0})
            continue
        rows = read_csv(source_path)
        fieldnames = rows[0].keys() if rows else task_rows[0].keys()
        copied = [row for row in rows if row.get("task_uid") in selected_task_uids]
        write_csv(output_path, fieldnames, copied)
        copied_ids = {row.get("task_uid") for row in copied}
        level_summary.append(
            {
                "level": level,
                "source_rows": len(rows),
                "copied_rows": len(copied),
                "missing_rows": len(selected_task_uids - copied_ids),
                "source_exists": 1,
            }
        )

    write_csv(args.output_dir / "filter_summary.csv", level_summary[0].keys(), level_summary)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(args.source_dir.resolve()),
        "task_manifest": str(args.task_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "levels": args.levels,
        "task_count": len(task_rows),
        "note": "Run analyze_recent5_lowlevel_match_rates.py with --task-manifest-input pointing to this output task_manifest.csv to fill missing rows.",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", **config}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
