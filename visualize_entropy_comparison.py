import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, render_template, request
from transformers import AutoTokenizer
MODEL_MAPPING = {
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "quietstar": "mistralai/Mistral-7B-v0.1",
}

from loguru import logger


def load_jsonl(path: str, max_n: int) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            data.append(json.loads(line))
    return data


def read_jsonl_line(path: str, index: int) -> Optional[Dict[str, Any]]:
    if index < 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == index:
                return json.loads(line)
    return None


def count_jsonl_lines(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        return 0


def discover_jsonl_files(directory: str) -> List[str]:
    import glob
    pattern = os.path.join(directory, "entropies_*.jsonl")
    files = sorted(glob.glob(pattern))
    return files


def build_union_segments(text: str,
                         left_offsets: List[Tuple[int, int]], left_ent: List[float],
                         right_offsets: List[Tuple[int, int]], right_ent: List[float]) -> Tuple[List[Dict[str, Any]], float]:
    # Build boundaries from both sides
    boundaries: List[int] = [0, len(text)]
    for s, e in left_offsets:
        boundaries.append(s)
        boundaries.append(e)
    for s, e in right_offsets:
        boundaries.append(s)
        boundaries.append(e)
    boundaries = sorted(sorted(set([b for b in boundaries if 0 <= b <= len(text)])))

    # Helper to map original per-offset values onto the union segments
    def value_for_span(start: int, end: int, offsets: List[Tuple[int, int]], ent: List[float]) -> float:
        # Find the token whose offset covers [start,end) midpoint
        if start >= end:
            return 0.0
        mid = (start + end) // 2
        for i, (s, e) in enumerate(offsets):
            if s <= mid < e and i < len(ent):
                return float(ent[i])
        return 0.0

    # Determine global max |diff| for opacity scaling
    segments: List[Dict[str, Any]] = []
    diffs_abs: List[float] = []
    for idx in range(len(boundaries) - 1):
        s = boundaries[idx]
        e = boundaries[idx + 1]
        piece = text[s:e]
        if piece == "":
            continue
        lh = value_for_span(s, e, left_offsets, left_ent)
        rh = value_for_span(s, e, right_offsets, right_ent)
        diff = lh - rh
        segments.append({
            "idx": len(segments),
            "start": s,
            "end": e,
            "text": piece,
            "left_h": lh,
            "right_h": rh,
            "diff": diff,
        })
        diffs_abs.append(abs(diff))

    max_abs_diff = max(diffs_abs) if diffs_abs else 0.0
    return segments, max_abs_diff


def apply_colors(segments: List[Dict[str, Any]], max_abs_diff: float) -> None:
    # Color scheme: red if left > right, green if left < right. Opacity scales with |diff|/max_abs_diff
    for seg in segments:
        diff = seg["diff"]
        if max_abs_diff > 0:
            alpha = 0.15 + 0.85 * min(abs(diff) / max_abs_diff, 1.0)
        else:
            alpha = 0.0

        if diff > 0:
            left_bg = f"rgba(255,0,0,{alpha})"
            right_bg = f"rgba(0,128,0,{alpha})"
        elif diff < 0:
            left_bg = f"rgba(0,128,0,{alpha})"
            right_bg = f"rgba(255,0,0,{alpha})"
        else:
            left_bg = ""
            right_bg = ""

        seg["left_bg"] = left_bg
        seg["right_bg"] = right_bg


def align_records(left_records: List[Dict[str, Any]], right_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    n = min(len(left_records), len(right_records))
    for i in range(n):
        L = left_records[i]
        R = right_records[i]
        text_L: str = L.get("text", "")
        text_R: str = R.get("text", "")
        # Only compare if the raw text matches; otherwise, skip
        if text_L != text_R:
            logger.warning(f"Mismatched text at index {i}; skipping")
            continue
        text = text_L
        left_offsets = [(int(s), int(e)) for s, e in (L.get("offsets") or [])]
        right_offsets = [(int(s), int(e)) for s, e in (R.get("offsets") or [])]
        left_ent = [float(x) for x in (L.get("entropy") or [])]
        right_ent = [float(x) for x in (R.get("entropy") or [])]

        # Truncate to same length as offsets length if needed
        if len(left_ent) > len(left_offsets):
            left_ent = left_ent[:len(left_offsets)]
        if len(right_ent) > len(right_offsets):
            right_ent = right_ent[:len(right_offsets)]

        segments, max_abs_diff = build_union_segments(text, left_offsets, left_ent, right_offsets, right_ent)
        apply_colors(segments, max_abs_diff)

        items.append({
            "key": f"pair-{i}",
            "text": text,
            "segments": segments,
        })
    return items


def ensure_offsets_and_trim(record: Optional[Dict[str, Any]], tokenizer: Optional[AutoTokenizer]) -> Tuple[List[Tuple[int, int]], List[float]]:
    if not record:
        return [], []
    text: str = record.get("text", "")
    offsets: List[Tuple[int, int]] = [(int(s), int(e)) for s, e in (record.get("offsets") or [])]
    ent: List[float] = [float(x) for x in (record.get("entropy") or [])]
    # If offsets missing or length mismatch, recompute using tokenizer
    if (not offsets) or (len(offsets) < len(ent)):
        if tokenizer is not None:
            enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    # If ent shorter than offsets, truncate offsets to ent length to avoid long zero tails
    if len(ent) < len(offsets):
        offsets = offsets[:len(ent)]
    return offsets, ent


def infer_model_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    stem = Path(path).stem
    parts = stem.split("_")
    if len(parts) >= 2:
        key = parts[1]
        return MODEL_MAPPING.get(key, None)
    return None


def create_app(left_path: Optional[str], right_path: Optional[str], max_n: int) -> Flask:
    app = Flask(__name__)

    # Discover available files each request so newly created files appear
    available_files = discover_jsonl_files(os.getcwd())
    default_left = left_path or (available_files[0] if available_files else "")
    default_right = right_path or (available_files[1] if len(available_files) > 1 else default_left)

    @app.route("/")
    def index():
        left = request.args.get("left", default_left)
        right = request.args.get("right", default_right)
        # Single-sample index-based navigation
        idx = int(request.args.get("idx", 0))

        # Load tokenizers lazily per request to ensure correct models for each file
        left_model = infer_model_from_path(left)
        right_model = infer_model_from_path(right)
        left_tok = AutoTokenizer.from_pretrained(left_model, use_fast=True) if left_model else None
        right_tok = AutoTokenizer.from_pretrained(right_model, use_fast=True) if right_model else None

        total_left = count_jsonl_lines(left) if left else 0
        total_right = count_jsonl_lines(right) if right else 0
        total = min(total_left, total_right)
        if total == 0:
            items: List[Dict[str, Any]] = []
        else:
            # Clamp idx
            if idx < 0:
                idx = 0
            if idx >= total:
                idx = total - 1
            L = read_jsonl_line(left, idx) if left else None
            R = read_jsonl_line(right, idx) if right else None
            # Ensure offsets present and aligned
            if L is not None:
                loff, lent = ensure_offsets_and_trim(L, left_tok)
                L = {**L, "offsets": loff, "entropy": lent}
            if R is not None:
                roff, rent = ensure_offsets_and_trim(R, right_tok)
                R = {**R, "offsets": roff, "entropy": rent}
            items = align_records([L] if L else [], [R] if R else [])

        return render_template(
            "entropy_comparison.html",
            items=items,
            left_path=left,
            right_path=right,
            idx=idx,
            total=total,
            available_files=available_files,
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize entropy comparison between two JSONL files")
    parser.add_argument("--left", type=str, required=False, default=None, help="Path to left JSONL (e.g., baseline)")
    parser.add_argument("--right", type=str, required=False, default=None, help="Path to right JSONL (e.g., new model)")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--n", type=int, default=20, help="Max samples to display")
    args = parser.parse_args()

    app = create_app(args.left, args.right, args.n)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()


