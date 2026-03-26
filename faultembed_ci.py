"""
ci_integration/faultembed_ci.py
FaultEmbed CI/CD integration — screens Java methods in a pull request.

Usage:
  python ci_integration/faultembed_ci.py \\
      --diff   path/to/pr.diff \\
      --index  embeddings/faiss_index.bin \\
      --model  embeddings/faultembed_rf.pkl \\
      --meta   embeddings/meta.json \\
      --output ci_report.json

Exit codes:
  0 — no faults detected
  1 — one or more faults detected
  2 — error

Reports per-method verdict in JSON for CI pipeline consumption.
At 68 ms/method with zero API cost (Table 14).
"""

from __future__ import annotations
import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_NAME  = "microsoft/graphcodebert-base"
EMBED_DIM   = 768
K           = 5
TAU         = 0.4
N_CLASSES   = 4
MAX_TOKENS  = 512
STRIDE      = 128

LABEL_MAP = {0: "LOCK_CONTENTION", 1: "MEMORY_LEAK", 2: "RESOURCE_LEAK", 3: "SAFE"}
EXIT_FAULT = 1
EXIT_CLEAN = 0
EXIT_ERROR = 2


# ---------------------------------------------------------------------------
# Encoder (frozen, reused from 04_generate_embeddings)
# ---------------------------------------------------------------------------

_tok = _mdl = None


def get_encoder(device: str = "cpu"):
    global _tok, _mdl
    if _tok is None:
        log.info("Loading %s (frozen) …", MODEL_NAME)
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _mdl = AutoModel.from_pretrained(MODEL_NAME)
        _mdl.eval()
        for p in _mdl.parameters():
            p.requires_grad = False
        _mdl = _mdl.to(device)
    return _tok, _mdl


@torch.no_grad()
def encode(code: str, device: str = "cpu") -> np.ndarray:
    tok, mdl = get_encoder(device)
    ids = tok(code, add_special_tokens=True, truncation=False,
               return_tensors="pt")["input_ids"][0]

    if len(ids) <= MAX_TOKENS:
        enc = tok(code, add_special_tokens=True, max_length=MAX_TOKENS,
                  truncation=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        cls = mdl(**enc).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
    else:
        windows = []
        start = 0
        while start < len(ids):
            end   = min(start + MAX_TOKENS, len(ids))
            chunk = ids[start:end].unsqueeze(0).to(device)
            mask  = torch.ones_like(chunk)
            cls   = mdl(input_ids=chunk, attention_mask=mask).last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
            windows.append(cls)
            if end == len(ids): break
            start += STRIDE
        cls = np.mean(windows, axis=0)

    norm = np.linalg.norm(cls)
    return cls / max(norm, 1e-9)


# ---------------------------------------------------------------------------
# Feature construction — Algorithm 1
# ---------------------------------------------------------------------------

def build_feature(
    emb: np.ndarray,
    index: faiss.IndexFlatIP,
    train_labels: np.ndarray,
) -> np.ndarray:
    q = emb.reshape(1, -1).astype(np.float32)
    dists, idx = index.search(q, K)
    dists, idx = dists[0], idx[0]

    d_cat = np.zeros(N_CLASSES, dtype=np.float32)
    for ni in idx[idx >= 0]:
        lbl = int(train_labels[ni])
        if 0 <= lbl < N_CLASSES:
            d_cat[lbl] += 1
    d_cat /= max(d_cat.sum(), 1)

    valid = dists[idx >= 0]
    s_mu  = float(np.mean(valid)) if len(valid) else 0.0
    s_max = float(np.max(valid))  if len(valid) else 0.0

    return np.concatenate([emb, d_cat, [s_mu, s_max]]).astype(np.float32)


# ---------------------------------------------------------------------------
# Diff parser — extract changed Java methods
# ---------------------------------------------------------------------------

def parse_diff(diff_text: str) -> list[dict]:
    """Extract added/changed Java method bodies from a unified diff."""
    methods = []
    current_file = None
    buf_lines = []
    in_hunk = False

    for line in diff_text.splitlines():
        if line.startswith("+++ b/") and line.endswith(".java"):
            current_file = line[6:]
            buf_lines = []
            in_hunk = False
        elif line.startswith("@@"):
            if buf_lines:
                code = "\n".join(buf_lines)
                mnames = extract_method_names(code)
                for name in mnames:
                    body = extract_method_body(code, name)
                    if body:
                        methods.append({"file": current_file or "unknown",
                                         "method": name, "code": body})
            buf_lines = []
            in_hunk = True
        elif in_hunk and line.startswith("+") and not line.startswith("+++"):
            buf_lines.append(line[1:])

    if buf_lines:
        code = "\n".join(buf_lines)
        for name in extract_method_names(code):
            body = extract_method_body(code, name)
            if body:
                methods.append({"file": current_file or "unknown",
                                 "method": name, "code": body})
    return methods


def extract_method_names(code: str) -> list[str]:
    pattern = re.compile(
        r"(?:public|private|protected|static|synchronized|final|\s)*"
        r"[\w<>\[\],\s]+\s+(\w+)\s*\("
    )
    skip = {"if", "for", "while", "switch", "catch", "try", "else"}
    return [m.group(1) for m in pattern.finditer(code)
            if m.group(1) not in skip]


def extract_method_body(source: str, name: str) -> str:
    lines   = source.splitlines()
    sig_re  = re.compile(
        r"(?:public|private|protected|static|synchronized|final|\s)*"
        r"[\w<>\[\],\s]+\s+" + re.escape(name) + r"\s*\("
    )
    buf, depth, active = [], 0, False
    for line in lines:
        if not active:
            if sig_re.search(line):
                active = True; buf = [line]
                depth  = line.count("{") - line.count("}")
        else:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                break
    return "\n".join(buf) if len(buf) > 1 else ""


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

def screen_methods(
    methods: list[dict],
    index: faiss.IndexFlatIP,
    clf,
    train_labels: np.ndarray,
    device: str = "cpu",
) -> list[dict]:
    results = []
    for m in methods:
        t0   = time.perf_counter()
        emb  = encode(m["code"], device)
        t_enc = (time.perf_counter() - t0) * 1000

        t0   = time.perf_counter()
        feat = build_feature(emb, index, train_labels)
        t_idx = (time.perf_counter() - t0) * 1000

        t0   = time.perf_counter()
        prob = clf.predict_proba(feat.reshape(1, -1))[0]
        t_clf = (time.perf_counter() - t0) * 1000

        pred_lbl = int(np.argmax(prob))
        conf     = float(prob[pred_lbl])

        if conf < TAU:
            verdict = "SAFE"
            label   = "SAFE"
        else:
            label   = LABEL_MAP[pred_lbl]
            verdict = "FAULT" if pred_lbl < 3 else "SAFE"

        results.append({
            "file":        m["file"],
            "method":      m["method"],
            "verdict":     verdict,
            "label":       label,
            "confidence":  round(conf, 4),
            "latency_ms": {
                "encode":  round(t_enc, 1),
                "faiss":   round(t_idx, 1),
                "rf":      round(t_clf, 1),
                "total":   round(t_enc + t_idx + t_clf, 1),
            },
        })
        log.info("%-40s  %-18s  conf=%.3f  %.0fms",
                 f"{m['file']}::{m['method']}",
                 label, conf, t_enc + t_idx + t_clf)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    p = argparse.ArgumentParser(description="FaultEmbed CI/CD screener")
    p.add_argument("--diff",    required=True)
    p.add_argument("--index",   default="embeddings/faiss_index.bin")
    p.add_argument("--model",   default="results/faultembed_rf.pkl")
    p.add_argument("--meta",    default="embeddings/meta.json")
    p.add_argument("--output",  default="ci_report.json")
    p.add_argument("--device",  default="cpu")
    p.add_argument("--fail-on-fault", action="store_true",
                   help="Exit code 1 if any fault detected")
    args = p.parse_args()

    try:
        diff_text = Path(args.diff).read_text(errors="replace")
        methods   = parse_diff(diff_text)
        log.info("Extracted %d methods from diff", len(methods))

        if not methods:
            log.info("No Java methods found in diff — exiting clean")
            return EXIT_CLEAN

        index = faiss.read_index(args.index)
        with open(args.model, "rb") as f:
            clf = pickle.load(f)

        # Reconstruct train_labels from meta
        meta = json.loads(Path(args.meta).read_text())
        train_labels = np.array([
            v["label"] for v in sorted(meta.values(),
                                        key=lambda x: int(x.get("id", 0)))
            if v.get("in_train")
        ], dtype=np.int32)

        results = screen_methods(methods, index, clf, train_labels, args.device)

    except Exception as e:
        log.error("CI screener error: %s", e)
        return EXIT_ERROR

    # Report
    faults = [r for r in results if r["verdict"] == "FAULT"]
    report = {
        "total_methods_screened": len(results),
        "faults_detected":        len(faults),
        "clean_methods":          len(results) - len(faults),
        "results":                results,
        "summary": {
            "LOCK_CONTENTION": sum(1 for r in faults if r["label"] == "LOCK_CONTENTION"),
            "MEMORY_LEAK":     sum(1 for r in faults if r["label"] == "MEMORY_LEAK"),
            "RESOURCE_LEAK":   sum(1 for r in faults if r["label"] == "RESOURCE_LEAK"),
        },
        "avg_latency_ms": round(
            np.mean([r["latency_ms"]["total"] for r in results]) if results else 0.0, 1
        ),
    }

    Path(args.output).write_text(json.dumps(report, indent=2))

    print(f"\n{'='*60}")
    print(f"  FaultEmbed CI Report")
    print(f"  Methods screened:  {report['total_methods_screened']}")
    print(f"  Faults detected:   {report['faults_detected']}")
    print(f"  Avg latency:       {report['avg_latency_ms']} ms/method  (paper: 68ms)")
    if faults:
        print(f"\n  Detected faults:")
        for r in faults:
            print(f"    [{r['label']}]  {r['file']}::{r['method']}  "
                  f"(conf={r['confidence']:.3f})")
    print(f"{'='*60}")
    print(f"  Full report → {args.output}")

    if args.fail_on_fault and faults:
        return EXIT_FAULT
    return EXIT_CLEAN


if __name__ == "__main__":
    sys.exit(main())
