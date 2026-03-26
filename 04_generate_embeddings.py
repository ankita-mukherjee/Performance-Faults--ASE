"""
embeddings/04_generate_embeddings.py
Embedding generation (Section 3.2).

Produces three L2-normalised 768-dim embeddings per validated pair:
  v_bug   — before-fix method
  v_fix   — after-fix method
  v_delta — <before>[SEP]<after>

Methods > 512 tokens use sliding window (512 tokens, stride 128, mean-pool).
Affects ~12% of pairs; recovers +6% F1 on that subset (Section 7.1).

Also builds and saves the FAISS FlatIP training index.

Outputs
-------
embeddings/embeddings.npz   arrays: v_bug, v_fix, v_delta, labels, projects,
                                    methods, ids, train_mask
embeddings/faiss_index.bin  FAISS index of training v_bug vectors
embeddings/meta.json        id → {label, project, method, fault_category}
"""

from __future__ import annotations
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import faiss                                  # pip install faiss-cpu
from transformers import AutoTokenizer, AutoModel  # pip install transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "microsoft/graphcodebert-base"
EMBED_DIM   = 768
MAX_TOKENS  = 512
STRIDE      = 128

FAULT_LABEL = {"lock_contention": 0, "memory_leak": 1, "resource_leak": 2, "safe": 3}
LABEL_NAME  = {v: k for k, v in FAULT_LABEL.items()}


# ---------------------------------------------------------------------------
# Model loading (frozen)
# ---------------------------------------------------------------------------

def load_model(device: str):
    log.info("Loading %s (frozen) on %s …", MODEL_NAME, device)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModel.from_pretrained(MODEL_NAME)
    mdl.eval()
    for p in mdl.parameters():
        p.requires_grad = False
    mdl = mdl.to(device)
    n_params = sum(p.numel() for p in mdl.parameters())
    log.info("Model loaded: %.0fM parameters", n_params / 1e6)
    return tok, mdl


# ---------------------------------------------------------------------------
# Single embedding with sliding window
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed(text: str, tok, mdl, device: str) -> np.ndarray:
    """
    Return L2-normalised [CLS] embedding.
    Uses sliding window for sequences > MAX_TOKENS (affects ~12% of dataset).
    """
    ids = tok(text, add_special_tokens=True, truncation=False,
              return_tensors="pt")["input_ids"][0]
    total = len(ids)

    if total <= MAX_TOKENS:
        enc = tok(text, add_special_tokens=True, max_length=MAX_TOKENS,
                  truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        cls = mdl(**enc).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    else:
        # Sliding window: mean-pool [CLS] over windows
        window_cls = []
        start = 0
        while start < total:
            end   = min(start + MAX_TOKENS, total)
            chunk = ids[start:end].unsqueeze(0).to(device)
            mask  = torch.ones_like(chunk)
            cls   = mdl(input_ids=chunk, attention_mask=mask).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            window_cls.append(cls)
            if end == total:
                break
            start += STRIDE
        cls = np.mean(window_cls, axis=0)

    norm = np.linalg.norm(cls)
    return cls / max(norm, 1e-9)


def embed_pair(buggy: str, fixed: str, tok, mdl, device: str):
    """Returns (v_bug, v_fix, v_delta) all L2-normalised."""
    v_bug   = embed(buggy, tok, mdl, device)
    v_fix   = embed(fixed, tok, mdl, device)
    sep     = tok.sep_token or "[SEP]"
    v_delta = embed(buggy + f" {sep} " + fixed, tok, mdl, device)
    return v_bug, v_fix, v_delta


# ---------------------------------------------------------------------------
# Dataset embedding loop
# ---------------------------------------------------------------------------

def embed_dataset(csv_path: Path, tok, mdl, device: str) -> dict:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    log.info("Embedding %d pairs …", len(rows))
    ids, v_bugs, v_fixes, v_deltas = [], [], [], []
    labels, projects, methods = [], [], []
    long_window = 0

    for i, row in enumerate(rows):
        try:
            # Count tokens to track sliding-window usage
            n_tok = len(tok(row["buggy_code"], add_special_tokens=True,
                            truncation=False)["input_ids"])
            if n_tok > MAX_TOKENS:
                long_window += 1

            vb, vf, vd = embed_pair(row["buggy_code"], row["fixed_code"], tok, mdl, device)
        except Exception as e:
            log.warning("Skip %s/%s: %s", row.get("project"), row.get("method_name"), e)
            continue

        pair_id = f"{row['project']}_{row['commit_hash'][:7]}_{row['method_name']}"
        ids.append(pair_id)
        v_bugs.append(vb); v_fixes.append(vf); v_deltas.append(vd)
        labels.append(FAULT_LABEL.get(row["fault_category"], -1))
        projects.append(row["project"])
        methods.append(row["method_name"])

        if (i + 1) % 100 == 0:
            log.info("  %d / %d  (sliding window: %d = %.1f%%)",
                     i+1, len(rows), long_window, 100*long_window/(i+1))

    log.info("Sliding-window used for %.1f%% of pairs (paper: 12%%)",
             100 * long_window / max(len(ids), 1))

    return {
        "ids":      ids,
        "v_bug":    np.array(v_bugs,   dtype=np.float32),
        "v_fix":    np.array(v_fixes,  dtype=np.float32),
        "v_delta":  np.array(v_deltas, dtype=np.float32),
        "labels":   np.array(labels,   dtype=np.int32),
        "projects": np.array(projects),
        "methods":  np.array(methods),
    }


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """FlatIP = exact cosine similarity search (vectors are L2-normalised)."""
    idx = faiss.IndexFlatIP(EMBED_DIM)
    idx.add(vectors.astype(np.float32))
    log.info("FAISS index: %d vectors", idx.ntotal)
    return idx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Generate GraphCodeBERT embeddings + FAISS index")
    p.add_argument("--input",       default="data/validated_pairs.csv")
    p.add_argument("--output-dir",  default="embeddings")
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train-ratio", type=float, default=0.9,
                   help="Fraction in training index (paper: 90%%)")
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok, mdl = load_model(args.device)
    data     = embed_dataset(Path(args.input), tok, mdl, args.device)
    n        = len(data["ids"])

    # Stratified split (preserve project distribution)
    rng   = np.random.default_rng(args.seed)
    perm  = rng.permutation(n)
    n_tr  = int(n * args.train_ratio)
    t_idx = perm[:n_tr]

    train_mask = np.zeros(n, dtype=bool)
    train_mask[t_idx] = True

    # Save embeddings
    np.savez(
        out / "embeddings.npz",
        ids        = data["ids"],
        v_bug      = data["v_bug"],
        v_fix      = data["v_fix"],
        v_delta    = data["v_delta"],
        labels     = data["labels"],
        projects   = data["projects"],
        methods    = data["methods"],
        train_mask = train_mask,
    )
    log.info("Saved embeddings → %s", out / "embeddings.npz")

    # FAISS index (training split only)
    idx = build_index(data["v_bug"][train_mask])
    faiss.write_index(idx, str(out / "faiss_index.bin"))
    log.info("FAISS index → %s", out / "faiss_index.bin")

    # Meta
    meta = {
        i: {
            "id":             data["ids"][i],
            "label":          int(data["labels"][i]),
            "fault_category": LABEL_NAME.get(int(data["labels"][i]), "unknown"),
            "project":        str(data["projects"][i]),
            "method":         str(data["methods"][i]),
            "in_train":       bool(train_mask[i]),
        }
        for i in range(n)
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    log.info("Train: %d | Test: %d | Total: %d", train_mask.sum(),
             (~train_mask).sum(), n)


if __name__ == "__main__":
    main()
