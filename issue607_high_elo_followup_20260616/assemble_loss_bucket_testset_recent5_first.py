"""Build a loss-bucket test set, preferring already-computed recent5 nodes.

The output is a task manifest for evaluation/collection. Each loss bucket is
filled from recent5 rows that already have level1-10 recommendations and
level18 candidate child losses. Only bucket shortfalls are filled from a
pre-sampled full-server manifest.
"""

import argparse
import csv
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_RECENT5_TASKS = SCRIPT_DIR / "reports" / "recent5_lowlevel_match_rates_threads5" / "task_manifest.csv"
DEFAULT_RECENT5_CHILD_LOSSES = SCRIPT_DIR / "reports" / "recent5_candidate_child_losses_level18" / "candidate_child_losses_level18.csv"
DEFAULT_FULL_TASKS = SCRIPT_DIR / "reports" / "full_loss_bucket_sample_20260616" / "task_manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "loss_bucket_testset_recent5_first_20260616"

BUCKETS: List[Tuple[str, Optional[float], Optional[float]]] = [
    ("negative", None, 0.0),
    ("0", 0.0, 0.0),
    ("0-2", 0.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-8", 5.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10-18", 10.0, 18.0),
    ("18-28", 18.0, 28.0),
    ("28+", 28.0, None),
]

BASE_FIELDNAMES = [
    "task_uid",
    "task_index",
    "source_dataset",
    "source_task_index",
    "year",
    "game_no",
    "tournament",
    "global_ply",
    "chunk_index",
    "chunk_range",
    "empties_before",
    "selected_player",
    "selected_player_index",
    "player_color",
    "opponent_player",
    "black_player",
    "white_player",
    "actual_move",
    "target_elo",
    "raw_name",
    "board_str",
    "transcript_prefix",
    "loss_bucket",
    "cached_parent_best_move",
    "cached_parent_best_score",
    "cached_actual_move_score",
    "cached_actual_child_loss",
    "cached_parent_depth",
    "cached_parent_probability",
]


def parse_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    return float(text)


def row_passes_min_elo(row: dict, min_elo: int) -> bool:
    if min_elo <= 0:
        return True
    elo = parse_float(row.get("target_elo"))
    return elo is not None and elo >= min_elo


def bucket_for_loss(loss: float) -> str:
    if loss < 0:
        return "negative"
    if loss == 0:
        return "0"
    for label, low, high in BUCKETS:
        if label in {"negative", "0"}:
            continue
        if low is not None and loss < low:
            continue
        if high is not None and loss >= high:
            continue
        return label
    raise ValueError(f"unbucketed loss: {loss}")


def split_semicolon(value: str) -> List[str]:
    return [part for part in str(value or "").split(";") if part]


def load_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_output_row(row: dict, source_dataset: str, loss_bucket: str) -> dict:
    out = {name: row.get(name, "") for name in BASE_FIELDNAMES}
    out["source_dataset"] = source_dataset
    out["loss_bucket"] = loss_bucket
    out["source_task_index"] = row.get("source_task_index", row.get("task_index", ""))
    return out


def build_recent5_rows(task_manifest: Path, child_loss_csv: Path, min_elo: int) -> List[dict]:
    tasks = {row["task_uid"]: row for row in load_csv(task_manifest) if row_passes_min_elo(row, min_elo)}
    rows_by_bucket: Dict[str, List[dict]] = defaultdict(list)

    with child_loss_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for loss_row in reader:
            if loss_row.get("status", "ok") != "ok":
                continue
            sources = set(split_semicolon(loss_row.get("candidate_sources", "")))
            if "human" not in sources:
                continue
            child_loss = parse_float(loss_row.get("child_loss"))
            if child_loss is None:
                continue
            loss_bucket = bucket_for_loss(child_loss)
            for task_uid in split_semicolon(loss_row.get("source_task_uids", "")):
                task = tasks.get(task_uid)
                if task is None:
                    continue
                out = as_output_row(task, "recent5", loss_bucket)
                out["cached_parent_best_move"] = loss_row.get("parent_best_move", "")
                out["cached_parent_best_score"] = loss_row.get("parent_best_score", "")
                out["cached_actual_move_score"] = loss_row.get("candidate_score", "")
                out["cached_actual_child_loss"] = loss_row.get("child_loss", "")
                out["cached_parent_depth"] = loss_row.get("parent_depth", "")
                out["cached_parent_probability"] = loss_row.get("parent_probability", "")
                rows_by_bucket[loss_bucket].append(out)

    deduped: List[dict] = []
    seen = set()
    for _, _, _ in BUCKETS:
        pass
    for label, _, _ in BUCKETS:
        for row in rows_by_bucket[label]:
            key = row["task_uid"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
    return deduped


def group_by_bucket(rows: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = {label: [] for label, _, _ in BUCKETS}
    for row in rows:
        bucket = row.get("loss_bucket") or bucket_for_loss(float(row["cached_actual_child_loss"]))
        grouped[bucket].append(row)
    return grouped


def select_rows(
    recent5_rows: List[dict],
    full_rows: List[dict],
    max_per_bucket: int,
    seed: int,
    min_elo: int,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)
    recent5_by_bucket = group_by_bucket(recent5_rows)
    full_by_bucket = group_by_bucket(
        as_output_row(row, "full_supplement", row["loss_bucket"])
        for row in full_rows
        if row_passes_min_elo(row, min_elo)
    )

    selected_recent5: List[dict] = []
    selected_full: List[dict] = []
    summary_rows: List[dict] = []
    used_boards = set()

    for label, _, _ in BUCKETS:
        recent_candidates = recent5_by_bucket[label][:]
        rng.shuffle(recent_candidates)
        recent_selected = recent_candidates[:max_per_bucket]
        for row in recent_selected:
            used_boards.add(row["board_str"])
        selected_recent5.extend(recent_selected)

        needed = max_per_bucket - len(recent_selected)
        full_candidates = [row for row in full_by_bucket[label] if row.get("board_str") not in used_boards]
        rng.shuffle(full_candidates)
        full_selected = full_candidates[:needed]
        for row in full_selected:
            used_boards.add(row["board_str"])
        selected_full.extend(full_selected)

        summary_rows.append(
            {
                "loss_bucket": label,
                "recent5_available": len(recent5_by_bucket[label]),
                "recent5_selected": len(recent_selected),
                "full_available": len(full_by_bucket[label]),
                "full_selected": len(full_selected),
                "total_selected": len(recent_selected) + len(full_selected),
                "target_per_bucket": max_per_bucket,
                "shortfall": max(0, max_per_bucket - len(recent_selected) - len(full_selected)),
            }
        )

    combined = selected_recent5 + selected_full
    combined.sort(key=lambda r: (r["loss_bucket"], r["source_dataset"], int(r["source_task_index"] or 0), r["task_uid"]))
    selected_recent5.sort(key=lambda r: (r["loss_bucket"], int(r["source_task_index"] or 0), r["task_uid"]))
    selected_full.sort(key=lambda r: (r["loss_bucket"], int(r["source_task_index"] or 0), r["task_uid"]))
    return combined, selected_recent5, selected_full, summary_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble a loss-bucket test set, preferring already-computed recent5 rows.")
    parser.add_argument("--recent5-task-manifest", type=Path, default=DEFAULT_RECENT5_TASKS)
    parser.add_argument("--recent5-child-loss-csv", type=Path, default=DEFAULT_RECENT5_CHILD_LOSSES)
    parser.add_argument("--full-task-manifest", type=Path, default=DEFAULT_FULL_TASKS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-per-bucket", type=int, default=1000)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--seed", type=int, default=596)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {args.output_dir}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recent5_rows = build_recent5_rows(args.recent5_task_manifest, args.recent5_child_loss_csv, args.min_elo)
    full_rows = load_csv(args.full_task_manifest)
    combined, recent5_selected, full_selected, summary_rows = select_rows(
        recent5_rows=recent5_rows,
        full_rows=full_rows,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
        min_elo=args.min_elo,
    )

    write_csv(args.output_dir / "task_manifest.csv", BASE_FIELDNAMES, combined)
    write_csv(args.output_dir / "recent5_selected_task_manifest.csv", BASE_FIELDNAMES, recent5_selected)
    write_csv(args.output_dir / "full_supplement_task_manifest.csv", BASE_FIELDNAMES, full_selected)
    write_csv(args.output_dir / "bucket_source_summary.csv", summary_rows[0].keys(), summary_rows)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "recent5_task_manifest": str(args.recent5_task_manifest.resolve()),
        "recent5_child_loss_csv": str(args.recent5_child_loss_csv.resolve()),
        "full_task_manifest": str(args.full_task_manifest.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "max_per_bucket": args.max_per_bucket,
        "min_elo_inclusive": args.min_elo,
        "seed": args.seed,
        "recent5_candidate_rows": len(recent5_rows),
        "full_candidate_rows": len(full_rows),
        "selected_rows": len(combined),
        "recent5_selected_rows": len(recent5_selected),
        "full_selected_rows": len(full_selected),
        "note": "recent5 rows already have level1-10 recommendations and level18 candidate child losses; only full_supplement rows need additional collection.",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "done", **config}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
