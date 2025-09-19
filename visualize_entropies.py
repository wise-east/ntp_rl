import argparse
import json
import os
import glob
from typing import List, Dict, Any, Optional
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for
from transformers import AutoTokenizer
from loguru import logger
import numpy as np

MODEL_MAPPING = {
    "Mistral-7B-v0.1": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
    "quietstar": "mistralai/Mistral-7B-v0.1",
}

def load_jsonl(path: str, max_n: int) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            data.append(json.loads(line))
    return data

def compute_correlation_between_entropy_and_accuracy(records: List[Dict[str, Any]], accuracies: List[int]) -> float:
    entropies = []
    for r in records:
        entropies.extend(r.get("entropy", []))
    # remove 0s from entropies  
    entropies = [e for e in entropies if e != 0]
    return np.corrcoef(entropies, accuracies)[0, 1]

def compute_correlation_between_entropy_and_index_position(records: List[Dict[str, Any]]) -> float:
    entropies = []
    for r in records:
        entropies.extend(r.get("entropy", []))
    return np.corrcoef(entropies, np.arange(len(entropies)))[0, 1]

def compute_overall_accuracy(records: List[Dict[str, Any]]) -> tuple[int, int]:
    correct = 0
    total = 0
    accuracies = []
    for r in records:
        gt = r.get("tokens") or []
        pred = r.get("pred_tokens") or []
        L = min(len(gt), len(pred))
        logger.info(f"gt: {len(gt)}, pred: {len(pred)}, L: {L}")
        # models such as qwen don't add a bos token, so there is no prediction for the first token. therefore right align the predictions.
        gt = gt[-L:]
        pred = pred[-L:]
        for i in range(L):
            if pred[i] == gt[i]:
                correct += 1
                accuracies.append(1)
            else:
                accuracies.append(0)
            total += 1
    return correct, total, accuracies


def compute_average_loss(records: List[Dict[str, Any]]):
    vals = [r.get("loss") for r in records if r.get("loss") is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def build_spans(record: Dict[str, Any], tokenizer, max_entropy: float) -> Dict[str, Any]:
    text: str = record.get("text", "")
    tokens: List[str] = record.get("tokens", [])
    entropies: List[float] = record.get("entropy", [])
    offsets: List[List[int]] = record.get("offsets", [])
    pred_tokens: List[str] = record.get("pred_tokens", [""] * len(entropies))
    sequential_thought_tokens: List[str] = record.get("sequential_thought_tokens", [])

    # If offsets are missing or mismatched, fall back to re-tokenization
    if not offsets or len(offsets) != len(entropies):
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])  # tokenized pieces

    spans = []
    nonzero_count = 0
    length = len(tokens)
    logger.info(f"length of tokens: {len(tokens)}, length of entropies: {len(entropies)}, length of offsets: {len(offsets)}, length of pred_tokens: {len(pred_tokens)}")
    
    if len(tokens) == len(pred_tokens) + 1:
        pred_tokens = ["DUMMY"] + pred_tokens
     
    for idx in range(length):
        start, end = offsets[idx]
        piece = text[start:end]
        h = float(entropies[idx])
        greedy = pred_tokens[idx] if idx < len(pred_tokens) else ""
        mismatch = (idx < len(tokens) and idx < len(pred_tokens) and tokens[idx] != pred_tokens[idx]) 
        # if the prediction is a dummy token, then the mismatch is always False 
        mismatch = mismatch and pred_tokens[idx] != "DUMMY"
        if h <= 0.0 or piece == "":
            spans.append((idx, piece, 0.0, 0.0, greedy, mismatch))
            continue
        nonzero_count += 1
        alpha = 0.15 + 0.85 * min(h / max_entropy if max_entropy > 0 else 0.0, 1.0)
        sequential_thoughts = sequential_thought_tokens[idx] if idx < len(sequential_thought_tokens) else ""
        
        spans.append((idx, piece, h, alpha, greedy, mismatch, sequential_thoughts))

    return {
        "tokens": tokens,
        "spans": spans,
        "nonzero_count": nonzero_count,
    }


def discover_jsonl_files(directory: str) -> List[str]:
    pattern = os.path.join(directory, "entropies_*.jsonl")
    files = sorted(glob.glob(pattern))
    return files


def create_app(jsonl_path: Optional[str], model_name: str, max_n: int) -> Flask:
    app = Flask(__name__)
    # Discover available files each request so newly created files appear
    available_files = discover_jsonl_files(os.getcwd())
    default_path = jsonl_path or (available_files[0] if available_files else "")
    if model_name is None:
        model_name = Path(default_path).stem.split("_")[1].split("_")[0] if default_path else None
        model_name = MODEL_MAPPING[model_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    @app.route("/")
    def index():
        path = request.args.get("file", default_path)
        n = int(request.args.get("n", max_n))
        model = request.args.get("model", model_name)

        # Re-load tokenizer if model changed
        nonlocal tokenizer
        if model != model_name:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        if not path:
            records = []
        else:
            records = load_jsonl(path, n)
        # Estimate max entropy from records to normalize
        max_entropy = 0.0
        for r in records:
            if r.get("entropy"):
                max_entropy = max(max_entropy, max([float(x) for x in r["entropy"]] or [0.0]))

        correct, total, accuracies = compute_overall_accuracy(records)
        entropy_acc_correlation = compute_correlation_between_entropy_and_accuracy(records, accuracies)
        entropy_index_correlation = compute_correlation_between_entropy_and_index_position(records)
        accuracy = (correct / total * 100.0) if total else None
        avg_loss = compute_average_loss(records)

        items = [build_spans(r, tokenizer, max_entropy) for r in records]
        
        return render_template(
            "entropy.html",
            items=items,
            jsonl_path=path,
            model_name=model,
            max_n=n,
            available_files=available_files,
            accuracy=accuracy,
            correct=correct,
            total=total,
            entropy_index_correlation=entropy_index_correlation,
            entropy_acc_correlation=entropy_acc_correlation,
            avg_loss=avg_loss,
        )
        
    return app 


def main():
    parser = argparse.ArgumentParser(description="Visualize token entropies as colored text")
    parser.add_argument("--file", type=str, required=False, default=None, help="Path to JSONL output from compute_entropy.py. If omitted, use first entropies_*.jsonl in CWD.")
    parser.add_argument("--model", type=str, required=False, default=None, help="Model name. If omitted, infer from jsonl path.")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--n", type=int, default=20, help="Max samples to display")
    args = parser.parse_args()

    app = create_app(args.file, args.model, args.n)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()


