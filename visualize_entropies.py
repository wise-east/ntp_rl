import argparse
import json
import os
from typing import List, Dict, Any

from flask import Flask, render_template_string, request, redirect, url_for
from transformers import AutoTokenizer


HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Entropy Visualizer</title>
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }
      .sample { margin-bottom: 32px; }
      .token { padding: 1px 2px; border-radius: 2px; }
      .controls { margin-bottom: 16px; }
      .legend { margin-top: 8px; font-size: 12px; color: #555; }
      .grid { display: grid; grid-template-columns: 1fr; gap: 16px; }
      textarea { width: 100%; height: 100px; }
      .meta { color: #666; font-size: 12px; }
      .line { margin: 12px 0; line-height: 1.8; }
    </style>
  </head>
  <body>
    <h2>Entropy Visualizer</h2>
    <div class="controls">
      <form method="get" action="/">
        <label>JSONL File: <input type="text" name="file" value="{{ jsonl_path }}" size="60"></label>
        <label style="margin-left: 12px;">Max Samples: <input type="number" name="n" value="{{ max_n }}" min="1"></label>
        <label style="margin-left: 12px;">Model: <input type="text" name="model" value="{{ model_name }}" size="40"></label>
        <button type="submit">Load</button>
      </form>
      <div class="legend">Darker background = higher entropy. Zero-entropy and padded tokens are hidden.</div>
    </div>

    {% for item in items %}
      <div class="sample">
        <div class="meta">Sample {{ loop.index }} | Tokens: {{ item.tokens|length }} | Nonzero entropies: {{ item.nonzero_count }}</div>
        <div class="line">
          {% for span in item.spans %}
            {% if span[2] > 0 %}
              <span class="token" title="H={{ '%.3f' % span[2] }} | Greedy='{{ span[4] }}'" style="background-color: rgba(255, 0, 0, {{ span[3] }});">{{ span[1] }}</span>
            {% else %}
              <span>{{ span[1] }}</span>
            {% endif %}
          {% endfor %}
        </div>
      </div>
    {% endfor %}
  </body>
  </html>
"""


def load_jsonl(path: str, max_n: int) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            data.append(json.loads(line))
    return data


def build_spans(record: Dict[str, Any], tokenizer, max_entropy: float) -> Dict[str, Any]:
    text: str = record.get("text", "")
    tokens: List[str] = record.get("tokens", [])
    entropies: List[float] = record.get("entropy", [])
    offsets: List[List[int]] = record.get("offsets", [])
    pred_tokens: List[str] = record.get("pred_tokens", [""] * len(entropies))

    # If offsets are missing or mismatched, fall back to re-tokenization
    if not offsets or len(offsets) != len(entropies):
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
        tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"])  # tokenized pieces

    spans = []
    nonzero_count = 0
    length = min(len(tokens), len(entropies), len(offsets), len(pred_tokens))
    for idx in range(length):
        start, end = offsets[idx]
        piece = text[start:end]
        h = float(entropies[idx])
        greedy = pred_tokens[idx] if idx < len(pred_tokens) else ""
        if h <= 0.0 or piece == "":
            spans.append((idx, piece, 0.0, 0.0, greedy))
            continue
        nonzero_count += 1
        alpha = 0.15 + 0.85 * min(h / max_entropy if max_entropy > 0 else 0.0, 1.0)
        spans.append((idx, piece, h, alpha, greedy))

    return {
        "tokens": tokens,
        "spans": spans,
        "nonzero_count": nonzero_count,
    }


def create_app(jsonl_path: str, model_name: str, max_n: int) -> Flask:
    app = Flask(__name__)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    @app.route("/")
    def index():
        path = request.args.get("file", jsonl_path)
        n = int(request.args.get("n", max_n))
        model = request.args.get("model", model_name)

        # Re-load tokenizer if model changed
        nonlocal tokenizer
        if model != model_name:
            tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)

        records = load_jsonl(path, n)
        # Estimate max entropy from records to normalize
        max_entropy = 0.0
        for r in records:
            if r.get("entropy"):
                max_entropy = max(max_entropy, max([float(x) for x in r["entropy"]] or [0.0]))

        items = [build_spans(r, tokenizer, max_entropy) for r in records]
        return render_template_string(
            HTML_PAGE,
            items=items,
            jsonl_path=path,
            model_name=model,
            max_n=n,
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Visualize token entropies as colored text")
    parser.add_argument("--file", type=str, required=True, help="Path to JSONL output from compute_entropy.py")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-v0.1", help="Tokenizer model to align tokens")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--n", type=int, default=20, help="Max samples to display")
    args = parser.parse_args()

    app = create_app(args.file, args.model, args.n)
    app.run(host=args.host, port=args.port, debug=True)


if __name__ == "__main__":
    main()


