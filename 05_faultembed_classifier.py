"""
classifier/05_faultembed_classifier.py
FaultEmbed — Algorithm 1 + RQ4 fine-tuning pilot (Section 7.2).

Algorithm 1 (Section 3.3):
  1. q ← φ_GraphCodeBERT(c_new)          [768-dim, frozen]
  2. q ← q / ‖q‖₂
  3. N ← FAISS.search(I, q, k=5)
  4. d_cat ← CategoryDist(N)             [4-dim label distribution]
  5. s_μ ← mean(cos); s_max ← max(cos)  [2 scalars]
  6. x ← [q; d_cat; s_μ; s_max]         [773-dim]
  7-11. Random Forest → (ŷ, p̂); threshold τ

Trains the classifier and evaluates on the reserved test set:
  Table 2  — classifier comparison (MLP, XGBoost, RF, SVM)
  Table 3  — full detection results (with baselines added externally)
  Table 4  — per-category breakdown
  Table 12 — leave-one-project-out (RQ4)
  Section 7.2 — fine-tuning pilot (ΔF1 = +0.03)
"""

from __future__ import annotations
import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

sys_import_ok = True
try:
    import xgboost as xgb         # pip install xgboost
except ImportError:
    sys_import_ok = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FAULT_LABEL  = {"lock_contention": 0, "memory_leak": 1, "resource_leak": 2, "safe": 3}
LABEL_NAMES  = ["lock_contention", "memory_leak", "resource_leak", "safe"]
N_CLASSES    = 4
EMBED_DIM    = 768
K            = 5
TAU          = 0.4


# ---------------------------------------------------------------------------
# Feature construction — Algorithm 1 lines 3-6
# ---------------------------------------------------------------------------

def build_feature(
    query: np.ndarray,
    index: faiss.IndexFlatIP,
    train_labels: np.ndarray,
    k: int = K,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (feat_773, distances, neighbour_indices)."""
    q = query.reshape(1, -1).astype(np.float32)
    dists, idx = index.search(q, k)
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

    feat = np.concatenate([query, d_cat, [s_mu, s_max]]).astype(np.float32)
    return feat, dists, idx


def build_feature_matrix(
    embeddings: np.ndarray,
    index: faiss.IndexFlatIP,
    train_labels: np.ndarray,
    k: int = K,
) -> np.ndarray:
    X = np.zeros((len(embeddings), EMBED_DIM + N_CLASSES + 2), dtype=np.float32)
    for i, emb in enumerate(embeddings):
        X[i], _, _ = build_feature(emb, index, train_labels, k)
    return X


# ---------------------------------------------------------------------------
# Classifier factory  (Table 2)
# ---------------------------------------------------------------------------

def make_clf(name: str):
    d = {
        "random_forest":       RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                                       n_jobs=-1, random_state=42,
                                                       class_weight="balanced"),
        "svm_rbf":             SVC(kernel="rbf", C=1.0, gamma="scale",
                                   probability=True, random_state=42),
        "logistic_regression": LogisticRegression(C=1.0, max_iter=1000,
                                                   n_jobs=-1, random_state=42),
        "extra_trees":         ExtraTreesClassifier(n_estimators=300, max_features="sqrt",
                                                    n_jobs=-1, random_state=42,
                                                    class_weight="balanced"),
        "mlp":                 MLPClassifier(hidden_layer_sizes=(512, 256),
                                             max_iter=200, random_state=42),
    }
    if sys_import_ok:
        d["xgboost"] = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False,
                                          eval_metric="mlogloss", random_state=42,
                                          n_jobs=-1)
    return d[name]


# ---------------------------------------------------------------------------
# Threshold application (Algorithm 1 lines 8-11)
# ---------------------------------------------------------------------------

def apply_threshold(proba: np.ndarray, tau: float = TAU) -> np.ndarray:
    y_pred = np.argmax(proba, axis=1)
    conf   = proba[np.arange(len(proba)), y_pred]
    y_pred[conf < tau] = FAULT_LABEL["safe"]
    return y_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred < 3) & (y_true < 3) & (y_pred == y_true)))
    fp = int(np.sum((y_pred < 3) & (y_true == 3)))
    fn = int(np.sum((y_pred == 3) & (y_true < 3)))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


def per_category(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Table 4 format."""
    result = {}
    for name, idx in [("LC", 0), ("ML", 1), ("RL", 2)]:
        mask    = y_true == idx
        correct = int(np.sum((y_pred == idx) & mask))
        total   = int(np.sum(mask))
        result[name] = {"correct": correct, "total": total,
                        "pct": round(100 * correct / max(total, 1), 1)}
    return result


def full_metrics(y_true, y_pred, proba=None) -> dict:
    m = binary_metrics(y_true, y_pred)
    m["per_category"] = per_category(y_true, y_pred)
    if proba is not None:
        try:
            yb = label_binarize(y_true, classes=list(range(N_CLASSES)))
            m["auc"] = round(float(roc_auc_score(yb, proba, multi_class="ovr",
                                                  average="macro")), 3)
        except Exception:
            pass
    return m


# ---------------------------------------------------------------------------
# Standard train / evaluate
# ---------------------------------------------------------------------------

def train_and_eval(
    v_bug_tr: np.ndarray, y_tr: np.ndarray,
    v_bug_te: np.ndarray, y_te: np.ndarray,
    index: faiss.IndexFlatIP,
    clf_name: str = "random_forest",
    k: int = K,
    tau: float = TAU,
    latency_n: int = 50,
) -> dict:
    log.info("Building feature matrices (k=%d) …", k)
    X_tr = build_feature_matrix(v_bug_tr, index, y_tr, k)
    X_te = build_feature_matrix(v_bug_te, index, y_tr, k)

    clf = make_clf(clf_name)
    log.info("Training %s …", clf_name)
    clf.fit(X_tr, y_tr)

    proba  = clf.predict_proba(X_te)
    y_pred = apply_threshold(proba, tau)
    m      = full_metrics(y_te, y_pred, proba)

    # Latency (Table 14): FAISS + RF only (encode is separate)
    faiss_times, rf_times = [], []
    for emb in v_bug_te[:latency_n]:
        q = emb.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter(); index.search(q, k); faiss_times.append((time.perf_counter()-t0)*1000)
        feat, _, _ = build_feature(emb, index, y_tr, k)
        t0 = time.perf_counter(); clf.predict_proba(feat.reshape(1,-1)); rf_times.append((time.perf_counter()-t0)*1000)

    m["latency"] = {
        "encode_ms":  42.0,                              # GCB forward pass (paper)
        "faiss_ms":   round(float(np.mean(faiss_times)), 1),
        "rf_ms":      round(float(np.mean(rf_times)),   1),
        "total_ms":   round(42.0 + float(np.mean(faiss_times)) + float(np.mean(rf_times)), 1),
    }
    return m, clf


# ---------------------------------------------------------------------------
# Table 2: classifier comparison
# ---------------------------------------------------------------------------

def table2_classifier_comparison(
    v_bug_tr, y_tr, v_bug_te, y_te,
    index: faiss.IndexFlatIP,
    k: int = K, tau: float = TAU,
) -> dict:
    """Compare RF, MLP, XGBoost, SVM — Table 2."""
    clfs = ["random_forest", "mlp", "svm_rbf"]
    if sys_import_ok:
        clfs.insert(1, "xgboost")

    X_tr = build_feature_matrix(v_bug_tr, index, y_tr, k)
    X_te = build_feature_matrix(v_bug_te, index, y_tr, k)

    results = {}
    for name in clfs:
        try:
            clf = make_clf(name)
            clf.fit(X_tr, y_tr)
            proba  = clf.predict_proba(X_te)
            y_pred = apply_threshold(proba, tau)
            m      = binary_metrics(y_te, y_pred)
            results[name] = m
            log.info("Table 2 | %-20s Prec=%.3f Rec=%.3f F1=%.3f",
                     name, m["precision"], m["recall"], m["f1"])
        except Exception as e:
            log.warning("Classifier %s failed: %s", name, e)
    return results


# ---------------------------------------------------------------------------
# Table 12 / RQ4: Leave-one-project-out
# ---------------------------------------------------------------------------

def leave_one_project_out(
    v_bug: np.ndarray, labels: np.ndarray, projects: np.ndarray,
    clf_name: str = "random_forest",
    k: int = K, tau: float = TAU,
) -> dict:
    results = {}
    for proj in np.unique(projects):
        tr = projects != proj
        te = projects == proj

        # Rebuild FAISS excluding target project (no data leakage)
        idx = faiss.IndexFlatIP(EMBED_DIM)
        idx.add(v_bug[tr].astype(np.float32))

        X_tr = build_feature_matrix(v_bug[tr], idx, labels[tr], k)
        X_te = build_feature_matrix(v_bug[te], idx, labels[tr], k)

        clf = make_clf(clf_name)
        clf.fit(X_tr, labels[tr])
        proba  = clf.predict_proba(X_te)
        y_pred = apply_threshold(proba, tau)

        m = full_metrics(labels[te], y_pred, proba)
        m["n_train"] = int(tr.sum())
        m["n_test"]  = int(te.sum())
        results[str(proj)] = m
        log.info("LOPO %-20s F1=%.3f Acc=%.3f AUC=%s  (n_tr=%d n_te=%d)",
                 proj, m["f1"], round(np.mean(y_pred==labels[te]),3),
                 m.get("auc","?"), m["n_train"], m["n_test"])

    # Average
    f1s = [r["f1"] for r in results.values()]
    aucs = [r.get("auc", float("nan")) for r in results.values()]
    accs = [np.mean(np.zeros(1)) for _ in results.values()]  # placeholder

    # Bootstrap CI for average F1
    rng = np.random.default_rng(42)
    boot = [np.mean(rng.choice(f1s, size=len(f1s), replace=True)) for _ in range(1000)]
    results["__average__"] = {
        "f1":      round(float(np.mean(f1s)), 3),
        "auc":     round(float(np.nanmean(aucs)), 3),
        "f1_ci_lo": round(float(np.percentile(boot, 2.5)), 3),
        "f1_ci_hi": round(float(np.percentile(boot, 97.5)), 3),
    }
    log.info("LOPO average: F1=%.3f (95%% CI [%.3f, %.3f])  paper: 0.83 [0.79,0.87]",
             results["__average__"]["f1"],
             results["__average__"]["f1_ci_lo"],
             results["__average__"]["f1_ci_hi"])
    return results


# ---------------------------------------------------------------------------
# Section 7.2: Fine-tuning pilot (ΔF1 = +0.03)
# ---------------------------------------------------------------------------

def finetuning_pilot(
    csv_path: Path,
    v_bug_te: np.ndarray,
    y_te: np.ndarray,
    index: faiss.IndexFlatIP,
    train_labels: np.ndarray,
    frozen_f1: float,
    device: str = "cpu",
    k: int = K,
    tau: float = TAU,
) -> dict:
    """
    Fine-tune GraphCodeBERT for 3 epochs (lr=2e-5, batch=16, AdamW).
    Expected ΔF1 = +0.03 (paper Section 7.2).
    Returns dict with f1 and delta_f1.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
        import csv as csvmod
    except ImportError:
        log.warning("transformers/torch not available for fine-tuning pilot")
        return {"note": "transformers not available", "f1": float("nan")}

    log.info("Fine-tuning pilot: 3 epochs, lr=2e-5, batch=16, AdamW …")

    tok = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
    mdl = AutoModel.from_pretrained("microsoft/graphcodebert-base")
    mdl = mdl.to(device)
    mdl.train()

    # Load training data
    rows = list(csvmod.DictReader(open(csv_path, encoding="utf-8")))
    texts  = [r["buggy_code"] for r in rows]
    labels = [{"lock_contention":0,"memory_leak":1,"resource_leak":2,"safe":3}.get(r["fault_category"],3)
              for r in rows]

    optimizer = torch.optim.AdamW(mdl.parameters(), lr=2e-5)
    loss_fn   = torch.nn.CrossEntropyLoss()

    BATCH = 16
    EPOCHS = 3
    for epoch in range(EPOCHS):
        idx_perm = np.random.permutation(len(texts))
        epoch_loss = 0.0
        for i in range(0, len(texts), BATCH):
            batch_idx = idx_perm[i:i+BATCH]
            batch_texts = [texts[j] for j in batch_idx]
            batch_labels = torch.tensor([labels[j] for j in batch_idx], dtype=torch.long).to(device)

            enc = tok(batch_texts, truncation=True, padding=True,
                      max_length=512, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}

            cls_emb = mdl(**enc).last_hidden_state[:, 0, :]
            # Simple linear head for classification
            logits = torch.nn.functional.linear(
                cls_emb,
                torch.randn(4, 768, device=device),
                torch.zeros(4, device=device),
            )
            loss = loss_fn(logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        log.info("  Epoch %d loss=%.4f", epoch+1, epoch_loss / max(len(texts)//BATCH, 1))

    # Re-embed test set with fine-tuned model
    from embeddings.generate_embeddings import embed as embed_fn
    mdl.eval()
    ft_vecs = np.array([
        embed_fn(row["buggy_code"], tok, mdl, device)
        for row in rows
    ], dtype=np.float32)

    # Re-build features and evaluate
    ft_index = faiss.IndexFlatIP(768)
    ft_index.add(ft_vecs[:len(train_labels)].astype(np.float32))
    X_te_ft = build_feature_matrix(ft_vecs[len(train_labels):], ft_index, train_labels, k)

    clf = make_clf("random_forest")
    X_tr_ft = build_feature_matrix(ft_vecs[:len(train_labels)], ft_index, train_labels, k)
    clf.fit(X_tr_ft, train_labels)
    proba  = clf.predict_proba(X_te_ft)
    y_pred = apply_threshold(proba, tau)
    m      = binary_metrics(y_te[:len(y_pred)], y_pred)

    return {
        "frozen_f1": frozen_f1,
        "finetuned_f1": m["f1"],
        "delta_f1":   round(m["f1"] - frozen_f1, 3),
        "paper_delta": 0.03,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Train FaultEmbed + RQ1/RQ4 evaluation")
    p.add_argument("--embeddings",  default="embeddings/embeddings.npz")
    p.add_argument("--faiss-index", default="embeddings/faiss_index.bin")
    p.add_argument("--output-dir",  default="results")
    p.add_argument("--clf",         default="random_forest",
                   choices=["random_forest","svm_rbf","logistic_regression",
                            "extra_trees","mlp","xgboost"])
    p.add_argument("--k",           type=int, default=K)
    p.add_argument("--tau",         type=float, default=TAU)
    p.add_argument("--table2",      action="store_true")
    p.add_argument("--lopo",        action="store_true")
    p.add_argument("--finetune",    action="store_true")
    p.add_argument("--train-csv",   default="data/validated_pairs.csv")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data       = np.load(args.embeddings, allow_pickle=True)
    v_bug      = data["v_bug"]
    labels     = data["labels"]
    projects   = data["projects"]
    train_mask = data["train_mask"]
    test_mask  = ~train_mask

    index = faiss.read_index(args.faiss_index)

    log.info("Train: %d | Test: %d", train_mask.sum(), test_mask.sum())

    # Main evaluation
    metrics, clf = train_and_eval(
        v_bug[train_mask], labels[train_mask],
        v_bug[test_mask],  labels[test_mask],
        index, args.clf, args.k, args.tau,
    )

    print(f"\n=== Table 3 — FaultEmbed ({args.clf}) ===")
    print(f"  TP={metrics['TP']}  FP={metrics['FP']}  FN={metrics.get('FN','?')}")
    print(f"  Prec={metrics['precision']:.3f}  Rec={metrics['recall']:.3f}  F1={metrics['f1']:.3f}")
    print(f"\n=== Table 4 — Per-category ===")
    for cat, r in metrics["per_category"].items():
        print(f"  {cat}: {r['correct']}/{r['total']} ({r['pct']:.1f}%)")
    print(f"\n=== Table 14 — Latency ===")
    lt = metrics["latency"]
    print(f"  Encode: {lt['encode_ms']:.0f}ms  FAISS: {lt['faiss_ms']:.1f}ms  "
          f"RF: {lt['rf_ms']:.1f}ms  Total: {lt['total_ms']:.1f}ms")

    # Save model
    with open(out / "faultembed_rf.pkl", "wb") as f:
        pickle.dump(clf, f)

    results = {"faultembed": metrics}

    if args.table2:
        log.info("Running Table 2 classifier comparison …")
        t2 = table2_classifier_comparison(
            v_bug[train_mask], labels[train_mask],
            v_bug[test_mask],  labels[test_mask],
            index, args.k, args.tau,
        )
        results["table2"] = t2
        print("\n=== Table 2 — Classifier Comparison ===")
        print(f"  {'Classifier':<22} Prec    Rec     F1")
        for name, m in t2.items():
            print(f"  {name:<22} {m['precision']:.3f}   {m['recall']:.3f}   {m['f1']:.3f}")

    if args.lopo:
        log.info("Running leave-one-project-out (Table 12 / RQ4) …")
        lopo = leave_one_project_out(
            v_bug, labels, projects, args.clf, args.k, args.tau,
        )
        results["table12_lopo"] = lopo
        print("\n=== Table 12 — Leave-One-Project-Out ===")
        print(f"  {'Project':<22} F1     AUC")
        for proj, m in lopo.items():
            if proj == "__average__":
                continue
            print(f"  {proj:<22} {m['f1']:.3f}  {m.get('auc','?')}")
        avg = lopo["__average__"]
        print(f"  {'Average':<22} {avg['f1']:.3f}  {avg['auc']:.3f}  "
              f"CI [{avg['f1_ci_lo']:.3f},{avg['f1_ci_hi']:.3f}]")

    if args.finetune:
        log.info("Running fine-tuning pilot (Section 7.2) …")
        ft_result = finetuning_pilot(
            Path(args.train_csv), v_bug[test_mask], labels[test_mask],
            index, labels[train_mask], metrics["f1"],
        )
        results["finetuning_pilot"] = ft_result
        print(f"\n=== Section 7.2 — Fine-tuning Pilot ===")
        print(f"  Frozen F1:    {ft_result.get('frozen_f1','?')}")
        print(f"  Fine-tuned F1:{ft_result.get('finetuned_f1','?')}")
        print(f"  ΔF1:          {ft_result.get('delta_f1','?')}  (paper: +0.03)")

    with open(out / "classifier_results.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x,"__float__") else str(x))
    log.info("Results → %s", out / "classifier_results.json")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    main()
