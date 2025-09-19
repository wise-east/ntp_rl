# run inference with quietstar model on the c4 dataset 

# model name: ezelikman/quietstar-8-ahead

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from compute_entropy import compute_token_entropies, batched
import json
from loguru import logger
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--batch_idx", type=int, default=0)
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--temp", type=float, default=0.9)
parser.add_argument("--output", type=str, default="entropies_quietstar.jsonl")
parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1")
parser.add_argument("--max_samples", type=int, default=50)
parser.add_argument("--dataset", type=str, default="allenai/c4")
parser.add_argument("--split", type=str, default="validation")
parser.add_argument("--subset", type=str, default="en")
parser.add_argument("--text-field", dest="text_field", type=str, default="text")
parser.add_argument("--checkpoint", type=str, default="ezelikman/quietstar-8-ahead")
parser.add_argument("--n_ahead", type=int, default=8)
args = parser.parse_args()

output_path = args.output.replace(".jsonl", f"_{args.model.split('/')[-1]}_dataset_{args.dataset.split('/')[-1]}_split_{args.split}_subset_{args.subset}_textfield_{args.text_field}_maxsamples_{args.max_samples}.jsonl")

def quietstar_init(params):
    if params is None:
        params = {}
    else:
        params = params.params
    n_ahead = params.get("n_ahead", args.n_ahead if not args.baseline else 1)
    n_ahead_talk = 1
    use_start_thought_token = params.get("use_start_thought_token", True)
    use_end_thought_token = params.get("use_end_thought_token", True)
    include_policy_loss = params.get("include_policy_loss", True)
    gumbel_detach = params.get("gumbel_detach", True)
    merged_talk_heads = params.get("merged_talk_heads", True)
    residual_think_head = params.get("residual_think_head", False)
    optimize_lm_head_only_at_start = params.get("optimize_lm_head_only_at_start", False)
    print("Loading model")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto',
        max_thoughts=n_ahead + n_ahead_talk + 1,
        merged_talk_heads=merged_talk_heads,
        merged_lm_and_talk_heads=False,
        merged_lm_and_think_heads=True,
        use_concat_talk_head=True,
        use_shallow_think=True,
        use_shallow_talk=False,
        use_complex_think_head=False,
        use_complex_talk_head=True,
        use_weighted_talk_head=True,
    )
    print("Loaded model")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    tokenizer.padding_side = "right"
    tokenizer.pad_token_id = tokenizer.eos_token_id
    special_tokens_to_add = []
    if model.use_start_thought_token:
        special_tokens_to_add.append("<|startthought|>")
    if model.use_end_thought_token:
        special_tokens_to_add.append("<|endthought|>")
    if special_tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens_to_add})
        model.resize_token_embeddings(len(tokenizer))
    model.tokenizer = tokenizer
    model.gumbel_detach = gumbel_detach
    model.include_policy_loss = include_policy_loss
    model.use_end_thought_token = use_end_thought_token
    model.use_start_thought_token = use_start_thought_token
    model.n_ahead = n_ahead
    model.n_ahead_talk = n_ahead_talk
    model.n_passes = 1
    model.residual_think_head = residual_think_head
    if args.baseline:
        model.skip_residual = True
        model.cumulative_residual = False
        model.clever_residual = False
        model.base_residual = False
    model.optimize_lm_head_only_at_start = optimize_lm_head_only_at_start
    model.use_policy_loss = False
    model.rm_initialized = True
    model.first_run = False
    model.wandb_enabled = False
    model.config_params = params
    model.run_start = int(time.time())
    model.eval_mode = True
    model.eval()
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = quietstar_init(None)
ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
samples = [next(iter(ds))[args.text_field] for _ in range(args.max_samples)] 

model.eval()

with open(output_path, "w", encoding="utf-8") as f:
    with torch.no_grad():
        for batch_texts in tqdm(batched(samples, args.batch_size), total=(len(samples) + args.batch_size - 1)//args.batch_size):
            # Encode batch
            enc = model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )
            attention_mask = enc.attention_mask.to(model.device)
            input_ids = enc.input_ids.to(model.device)

            # Forward pass for batch
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            entropies_all = compute_token_entropies(output.logits, attention_mask)

            logits = output.logits
            # Exclude thought special tokens from next-token predictions
            start_id = model.tokenizer.convert_tokens_to_ids("<|startthought|>")
            end_id = model.tokenizer.convert_tokens_to_ids("<|endthought|>")
            if start_id is not None:
                logits[:, :, start_id] = -float("inf")
            if end_id is not None:
                logits[:, :, end_id] = -float("inf")

            # Greedy next-token predictions per position
            pred_ids_all = torch.argmax(logits, dim=-1)

            # Per-position negative log-likelihood (align with compute_entropy.py)
            log_probs_full = torch.log_softmax(logits.float(), dim=-1)
            target_ids = input_ids[:, 1:]
            log_probs_trunc = log_probs_full[:, :-1, :]
            nll_trunc = -log_probs_trunc.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            # Entropy per token (predictor positions)
            generated_thought_token_ids_all = output.sampled_token_history

            # Iterate per example in batch
            for b_idx, (text, ids, mask, seq_ent) in enumerate(zip(batch_texts, input_ids, attention_mask, entropies_all)):
                # Tokenize raw text for offsets and tokens (no special tokens)
                enc_off = model.tokenizer(text, return_offsets_mapping=False, add_special_tokens=False)
                pieces_ids = enc_off["input_ids"]
                tokens = model.tokenizer.convert_ids_to_tokens(pieces_ids)
                offsets = None

                # Determine non-pad span and slice aligned sequences
                nonpad_idx = (mask > 0).nonzero(as_tuple=False).squeeze(-1)
                start = int(nonpad_idx[0].item())
                last = int(nonpad_idx[-1].item())
                pred_ids_seq = pred_ids_all[b_idx, start:last].tolist()
                pred_tokens_seq = model.tokenizer.convert_ids_to_tokens(pred_ids_seq)

                seq_len = len(seq_ent)
                pieces_len = len(tokens)
                if getattr(model.tokenizer, "bos_token", None) is None and pieces_len == seq_len + 1:
                    seq_ent = [0.0] + seq_ent

                # Average loss over predictor positions
                nll_seq = nll_trunc[b_idx]
                nll_predictors = nll_seq[start:last]
                sample_loss = float(nll_predictors.mean().item()) if nll_predictors.numel() > 0 else None

                # Sequential thought tokens (per-sample)
                sequential_thought_tokens: list[str] = []
                if torch.is_tensor(generated_thought_token_ids_all) and generated_thought_token_ids_all.dim() == 3:
                    # expected [B, S, T]
                    per_pos = [
                        generated_thought_token_ids_all[b_idx, :, idx]
                        for idx in range(input_ids.shape[1])
                    ]
                    sequential_thought_tokens = [
                        model.tokenizer.decode(tok_ids, skip_special_tokens=False)
                        for tok_ids in per_pos
                    ][start:last]
                else: 
                    raise ValueError("generated_thought_token_ids_all is not a tensor of shape [B, S, T]: got shape {}".format(generated_thought_token_ids_all.shape))

                record = {
                    "text": text,
                    "tokens": tokens,
                    "entropy": seq_ent,
                    "offsets": offsets,
                    "pred_tokens": pred_tokens_seq,
                    "loss": sample_loss,
                    "sequential_thought_tokens": sequential_thought_tokens,
                }
                f.write(json.dumps(record) + "\n")
        
        