"""Filter candidate child-loss rows to selected task_uids."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_CSV = SCRIPT_DIR / "reports" / "recent5_candidate_child_losses_level18" / "candidate_child_losses_level18.csv"
DEFAULT_TASK_MANIFEST = SCRIPT_DIR / "reports" / "loss_bucket_testset_recent5_first_20260616" / "recent5_selected_task_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "recent5_selected_candidate_child_losses_level18_20260616"


def split_semicolon(value: str) -> List[str]:
    return [part for part in str(value or "").split(";") if part]


def read_task_uids(path: Path) -> set:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["task_uid"] for row in csv.DictReader(f)}


def write_rows(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter candidate child-loss CSV rows to a selected task manifest.")
    parser.add_argument("--source-csv", type=Path, default=DEFAULT_SOURCE_CSV)
    parser.add_argument("--task-manifest", type=Path, default=DEFAULT_TASK_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {args.output_dir}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = read_task_uids(args.task_manifest)

    source_rows = 0
    kept_rows: List[dict] = []
    with args.source_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for row in reader:
            source_rows += 1
            row_task_uids = set(split_semicolon(row.get("source_task_uids", "")))
            if row_task_uids & selected:
                kept_rows.append(row)

    output_csv = args.output_dir / "candidate_child_losses_level18.csv"
    kept_count = write_rows(output_csv, fieldnames, kept_rows)
    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_csv": str(args.source_csv.resolve()),
        "task_manifest": str(args.task_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "output_csv": str(output_csv.resolve()),
        "selected_task_count": len(selected),
        "source_rows": source_rows,
        "kept_rows": kept_count,
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", **config}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
