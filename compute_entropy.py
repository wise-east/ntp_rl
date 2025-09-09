import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
from datasets import load_dataset
from loguru import logger

@dataclass
class Args:
    output_path: str
    split: str
    subset: str
    text_field: str
    max_length: int
    batch_size: int
    max_samples: Optional[int]
    model_name: str
    revision: Optional[str]
    dtype: str
    device: str
    padding: bool
    trust_remote_code: bool


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Compute per-token entropy with a model over C4 samples")
    parser.add_argument("--output", dest="output_path", type=str, required=True, help="Path to write JSONL results")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split (e.g., validation)")
    parser.add_argument("--subset", type=str, default="en", help="Dataset subset, e.g., en")
    parser.add_argument("--text-field", type=str, default="text", help="Field name with raw text")
    parser.add_argument("--max-length", type=int, default=4096, help="Truncate/pad sequences to this length")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size of texts")
    parser.add_argument("--max-samples", type=int, default=50, help="Max number of samples to process")
    parser.add_argument("--model", dest="model_name", type=str, default="mistralai/Mistral-7B-v0.1", help="HF model name")
    parser.add_argument("--revision", type=str, default=None, help="HF revision (commit/tag)")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"], help="Model dtype")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, or cuda:N")
    parser.add_argument("--no-padding", dest="padding", action="store_false", help="Disable right padding")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to HF loaders")
    parser.set_defaults(padding=True)
    ns = parser.parse_args()
    return Args(
        output_path=ns.output_path,
        split=ns.split,
        subset=ns.subset,
        text_field=ns.text_field,
        max_length=ns.max_length,
        batch_size=ns.batch_size,
        max_samples=ns.max_samples,
        model_name=ns.model_name,
        revision=ns.revision,
        dtype=ns.dtype,
        device=ns.device,
        padding=ns.padding,
        trust_remote_code=ns.trust_remote_code,
    )


def infer_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_arg)


def infer_dtype(dtype_arg: str):
    if dtype_arg == "auto":
        if torch.cuda.is_available():
            return torch.float16
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_arg]


def batched(iterable: Iterable, n: int) -> Iterable[List]:
    batch: List = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def compute_token_entropies(logits: torch.Tensor, attention_mask: torch.Tensor) -> List[List[float]]:
    # logits: [B, T, V]; attention_mask: [B, T]
    # Use log_softmax for numerical stability. Compute H = -sum p * log p.
    logits = logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)  # [B, T]
    entropy = torch.nan_to_num(entropy, nan=0.0, posinf=0.0, neginf=0.0)
    # Mask padding positions
    entropy = entropy * attention_mask
    # Convert to python lists per sequence, trimming positions that won't have a next-token prediction (last token)
    entropies: List[List[float]] = []
    for seq_ent, seq_mask in zip(entropy, attention_mask):
        nonpad_idx = (seq_mask > 0).nonzero(as_tuple=False).squeeze(-1)
        if nonpad_idx.numel() <= 1:
            entropies.append([])
            continue
        start = int(nonpad_idx[0].item())
        last = int(nonpad_idx[-1].item())
        # Predict next-token: exclude the last non-pad position
        truncated = seq_ent[start:last]
        entropies.append([float(x) for x in truncated.tolist()])
    return entropies


def main() -> None:
    args = parse_args()
    device = infer_device(args.device)
    dtype = infer_dtype(args.dtype)

    # Stream dataset
    ds = load_dataset("allenai/c4", args.subset, streaming=True, split=args.split)
    iterator = (row[args.text_field] for row in ds)
    if args.max_samples is not None:
        # Create a bounded iterator
        def take(it, k):
            for i, x in enumerate(it):
                if i >= k:
                    break
                yield x

        iterator = take(iterator, args.max_samples)

    # Lazy import transformers to avoid overhead if just parsing args
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    # Ensure pad token exists for batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # set padding side to right 
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.revision,
        dtype=dtype,
        device_map=None,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    model.eval()

    # mistral's max length is 8192
    max_length = min(args.max_length, 8192)   
    logger.info(f"Max length: {max_length}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as fout:
        for batch_texts in batched(iterator, args.batch_size):
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=("max_length" if not args.padding else True),
                truncation=True,
                max_length=max_length,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                # Shift logits to align with labels (predict token t from logits at t-1)
                logits = out.logits  # [B, T, V]

            # Greedy next-token predictions per position (before shifting alignment)
            pred_ids_all = torch.argmax(logits, dim=-1)  # [B, T]
            # Per-position negative log-likelihood for the next token
            log_probs_full = torch.log_softmax(logits.float(), dim=-1)  # [B, T, V]
            target_ids = input_ids[:, 1:]  # [B, T-1]
            log_probs_trunc = log_probs_full[:, :-1, :]  # [B, T-1, V]
            nll_trunc = -log_probs_trunc.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # [B, T-1]

            entropies = compute_token_entropies(logits, attention_mask)
            # Build per-sample outputs with tokens and offsets aligned to entropies.
            for b_idx, (text, ids, mask, seq_ent) in enumerate(zip(batch_texts, input_ids, attention_mask, entropies)):
                
                # Tokenize raw text to obtain offsets (no special tokens)
                enc_off = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
                pieces_ids = enc_off["input_ids"]
                pieces = tokenizer.convert_ids_to_tokens(pieces_ids)
                offsets = enc_off["offset_mapping"]

                # Predicted next token strings aligned to entropy positions (non-pad span, excluding last token)
                nonpad_idx = (mask > 0).nonzero(as_tuple=False).squeeze(-1)
                start = int(nonpad_idx[0].item())
                last = int(nonpad_idx[-1].item())
                pred_ids_seq = pred_ids_all[b_idx, start:last].tolist()  # length == len(seq_ent)
                pred_tokens_seq = tokenizer.convert_ids_to_tokens(pred_ids_seq)

                seq_len = len(seq_ent)
                pieces_len = len(pieces)

                # Use BOS presence to handle off-by-one:
                # - If tokenizer.bos_token is None (e.g., Qwen), predictions correspond to tokens[1:]
                #   so add 0 entropy to the first token
                if getattr(tokenizer, "bos_token", None) is None and pieces_len == seq_len + 1:
                    seq_ent = [0.0] + seq_ent

                tokens = pieces

                # Compute per-sample average loss over predictor positions (exclude padding)
                nll_seq = nll_trunc[b_idx]
                nll_predictors = nll_seq[start:last]
                sample_loss = float(nll_predictors.mean().item()) if nll_predictors.numel() > 0 else None

                record = {
                    "text": text,
                    "tokens": tokens,
                    "entropy": seq_ent,
                    "offsets": offsets,
                    "pred_tokens": pred_tokens_seq,
                    "loss": sample_loss,
                }
                fout.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()