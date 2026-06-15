"""Sample full-server THOR nodes by existing level18 human child-loss buckets.

This script only selects nodes and writes a task manifest. It does not call
Egaroucid. The full-server detail file already contains level18 best/human
scores, so those values are copied as cache columns for later collection.
"""

import argparse
import csv
import json
import random
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORIES_DIR = SCRIPT_DIR.parent.parent
WORKSPACE_DIR = REPOSITORIES_DIR.parent

DEFAULT_DETAIL_CSV = WORKSPACE_DIR / "IMPORTANT_FULL_SERVER_DATA" / "full_dataset" / "thor_all_rawdata_detail.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reports" / "full_loss_bucket_sample"
DEFAULT_ELO_DIR = SCRIPT_DIR.parent / "elo"

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

TASK_FIELDNAMES = [
    "task_uid",
    "task_index",
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


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^A-Z0-9]+", "", text.upper())


def playerlist_to_wthor(name: str) -> str:
    tokens = name.strip().split()
    i = len(tokens)
    while i > 0 and tokens[i - 1].isupper():
        i -= 1
    if i == len(tokens):
        return name.title()
    given = " ".join(tokens[:i]).title()
    surname = " ".join(tokens[i:]).title()
    return f"{surname} {given}".strip()


def name_variants(name: str) -> List[str]:
    raw = " ".join(str(name or "").strip().split())
    if not raw:
        return []
    forms = {raw, raw.title(), playerlist_to_wthor(raw)}
    for form in list(forms):
        tokens = form.split()
        if len(tokens) >= 2:
            forms.add(" ".join(reversed(tokens)))
    return [normalize_name(form) for form in forms if normalize_name(form)]


def load_yearly_elo_maps(elo_dir: Path) -> Dict[int, Dict[str, Tuple[str, int]]]:
    maps: Dict[int, Dict[str, Tuple[str, int]]] = {}
    for path in sorted(elo_dir.glob("*.txt")):
        if not path.stem.isdigit():
            continue
        year = int(path.stem)
        year_map: Dict[str, Tuple[str, int]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = [part.strip() for part in line.split("\t")]
            if len(parts) < 7 or not parts[2].isdigit():
                continue
            raw_name = parts[5]
            elo = int(parts[2])
            for variant in name_variants(raw_name):
                year_map.setdefault(variant, (raw_name, elo))
        maps[year] = year_map
    return maps


def lookup_player_elo(yearly_elo_maps: Dict[int, Dict[str, Tuple[str, int]]], year: int, player_name: str) -> Tuple[str, Optional[int]]:
    year_map = yearly_elo_maps.get(year)
    if not year_map:
        return "", None
    for variant in name_variants(player_name):
        hit = year_map.get(variant)
        if hit:
            return hit
    return "", None


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


def make_task_row(row: dict, loss_bucket: str, child_loss: float, raw_name: str, target_elo: int) -> dict:
    source_task_index = row["task_index"]
    player_name = row.get("player_name", "")
    return {
        "task_uid": f"full_{source_task_index}",
        "task_index": source_task_index,
        "source_task_index": source_task_index,
        "year": row.get("year", ""),
        "game_no": row.get("game_no", ""),
        "tournament": row.get("tournament", ""),
        "global_ply": row.get("global_ply", ""),
        "chunk_index": row.get("chunk_index", ""),
        "chunk_range": row.get("chunk_range", ""),
        "empties_before": row.get("empties_before", ""),
        "selected_player": player_name,
        "selected_player_index": player_name,
        "player_color": row.get("player_color", ""),
        "opponent_player": row.get("opponent_player", ""),
        "black_player": row.get("black_player", ""),
        "white_player": row.get("white_player", ""),
        "actual_move": str(row.get("move", "")).lower(),
        "target_elo": target_elo,
        "raw_name": raw_name,
        "board_str": row.get("board_str", ""),
        "transcript_prefix": "",
        "loss_bucket": loss_bucket,
        "cached_parent_best_move": str(row.get("best_move", "")).lower(),
        "cached_parent_best_score": row.get("best_score", ""),
        "cached_actual_move_score": row.get("human_score", ""),
        "cached_actual_child_loss": int(child_loss) if float(child_loss).is_integer() else child_loss,
        "cached_parent_depth": row.get("alt_depth", ""),
        "cached_parent_probability": row.get("alt_probability", ""),
    }


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample full-server nodes by existing level18 human child-loss buckets.")
    parser.add_argument("--detail-csv", type=Path, default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--elo-dir", type=Path, default=DEFAULT_ELO_DIR)
    parser.add_argument("--min-elo", type=int, default=0)
    parser.add_argument("--max-per-bucket", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=596)
    parser.add_argument("--max-source-rows", type=int, default=0, help="Optional smoke/debug cap while scanning the full detail CSV.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_per_bucket <= 0:
        raise ValueError("--max-per-bucket must be positive")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output dir already exists: {args.output_dir}. Use --overwrite to replace it.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    seen_counts: Dict[str, int] = {label: 0 for label, _, _ in BUCKETS}
    sampled: Dict[str, List[dict]] = {label: [] for label, _, _ in BUCKETS}
    bad_rows = 0
    elo_missing_rows = 0
    elo_below_threshold_rows = 0
    scanned_rows = 0
    yearly_elo_maps = load_yearly_elo_maps(args.elo_dir) if args.min_elo > 0 else {}

    with args.detail_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scanned_rows += 1
            try:
                raw_name = str(row.get("player_name", ""))
                target_elo = 0
                if args.min_elo > 0:
                    raw_name, matched_elo = lookup_player_elo(yearly_elo_maps, int(row.get("year", "0")), row.get("player_name", ""))
                    if matched_elo is None:
                        elo_missing_rows += 1
                        if args.max_source_rows > 0 and scanned_rows >= args.max_source_rows:
                            break
                        continue
                    if matched_elo < args.min_elo:
                        elo_below_threshold_rows += 1
                        if args.max_source_rows > 0 and scanned_rows >= args.max_source_rows:
                            break
                        continue
                    target_elo = matched_elo
                raw_loss = parse_float(row.get("raw_stone_loss"))
                if raw_loss is None:
                    best_score = parse_float(row.get("best_score"))
                    human_score = parse_float(row.get("human_score"))
                    if best_score is None or human_score is None:
                        raise ValueError("missing loss columns")
                    raw_loss = best_score - human_score
                loss_bucket = bucket_for_loss(raw_loss)
                task_row = make_task_row(row, loss_bucket, raw_loss, raw_name, target_elo)
            except Exception:
                bad_rows += 1
                if args.max_source_rows > 0 and scanned_rows >= args.max_source_rows:
                    break
                continue

            seen_counts[loss_bucket] += 1
            bucket_rows = sampled[loss_bucket]
            if len(bucket_rows) < args.max_per_bucket:
                bucket_rows.append(task_row)
            else:
                replace_at = rng.randrange(seen_counts[loss_bucket])
                if replace_at < args.max_per_bucket:
                    bucket_rows[replace_at] = task_row

            if args.max_source_rows > 0 and scanned_rows >= args.max_source_rows:
                break

    task_rows: List[dict] = []
    for label, _, _ in BUCKETS:
        task_rows.extend(sampled[label])
    task_rows.sort(key=lambda r: int(r["task_index"]))

    write_csv(args.output_dir / "task_manifest.csv", TASK_FIELDNAMES, task_rows)

    summary_rows = []
    total_seen = sum(seen_counts.values())
    for label, _, _ in BUCKETS:
        count = seen_counts[label]
        summary_rows.append(
            {
                "loss_bucket": label,
                "source_count": count,
                "source_percent": round(count / total_seen, 8) if total_seen else 0.0,
                "sampled_count": len(sampled[label]),
                "max_per_bucket": args.max_per_bucket,
            }
        )
    write_csv(args.output_dir / "bucket_summary.csv", summary_rows[0].keys(), summary_rows)

    run_config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "detail_csv": str(args.detail_csv.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "max_per_bucket": args.max_per_bucket,
        "elo_dir": str(args.elo_dir.resolve()),
        "min_elo_inclusive": args.min_elo,
        "seed": args.seed,
        "max_source_rows": args.max_source_rows,
        "scanned_rows": scanned_rows,
        "bad_rows": bad_rows,
        "elo_missing_rows": elo_missing_rows,
        "elo_below_threshold_rows": elo_below_threshold_rows,
        "sampled_rows": len(task_rows),
        "buckets": [label for label, _, _ in BUCKETS],
        "note": "Loss buckets use existing full-server level18 raw_stone_loss/best_score/human_score; no console calls are made here.",
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "done", **run_config}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
