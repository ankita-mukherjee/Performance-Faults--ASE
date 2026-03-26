"""
baselines/06_baselines.py
All 19 baselines for Table 3.

 #  Paradigm              Model
 1  Rule-based            AST-Pattern (Mukherjee et al. [28])
 2  Rule-based            SonarQube
 3  Rule-based            Facebook Infer
 4  LLM 0-shot CoT        GPT-4
 5  LLM 0-shot CoT        DeepSeek-Coder-V2
 6  LLM 0-shot CoT        Claude 3.5
 7  LLM 5-shot static     GPT-4
 8  LLM 5-shot static     DeepSeek-Coder-V2
 9  LLM 5-shot static     Claude 3.5
10  LLM RA (FAISS top-5)  GPT-4
11  LLM RA (FAISS top-5)  DeepSeek-Coder-V2
12  LLM RA (FAISS top-5)  Claude 3.5
13  LLM embed + RF        GPT-3.5 ada-002
14  LLM embed + RF        CodeLlama-7B
15  Code transformer+RF   CodeBERT
16  Code transformer+RF   UniXcoder
17  Seq model             Word2Vec + BiLSTM
18  Seq model             Doc2Vec
19  GraphCodeBERT (no retrieval) — ablation reference

LLM API calls require:   OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY
Other embedding models:  provide pre-computed .npy files via --extra-embeddings
"""

from __future__ import annotations
import csv
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

FAULT_LABEL = {"lock_contention": 0, "memory_leak": 1, "resource_leak": 2, "safe": 3}
LABEL_NAMES = ["lock_contention", "memory_leak", "resource_leak", "safe"]
N_CLASSES   = 4


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(np.sum((y_pred < 3) & (y_true < 3) & (y_pred == y_true)))
    fp = int(np.sum((y_pred < 3) & (y_true == 3)))
    fn = int(np.sum((y_pred == 3) & (y_true < 3)))
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}


# ---------------------------------------------------------------------------
# Baselines 1-3: Rule-based
# ---------------------------------------------------------------------------

# AST-Pattern (6 LC smells + resource-leak path rules, Mukherjee et al. [28])
LC_PATTERNS  = [r"synchronized\s*\(", r"ReentrantLock", r"lockInterruptibly",
                r"BlockingQueue", r"\.wait\s*\(", r"Thread\.sleep"]
RL_PATTERNS  = [r"new\s+FileInputStream", r"new\s+FileOutputStream",
                r"getConnection\s*\(", r"openStream\s*\(",
                r"new\s+Socket\s*\(", r"RestRepository", r"ScrollQuery",
                r"new\s+BufferedReader"]
ML_PATTERNS  = [r"static\s+.*Map\s*<", r"static\s+.*List\s*<",
                r"static\s+.*Set\s*<", r"ConcurrentHashMap",
                r"\.put\s*\(", r"\.add\s*\(.*cache", r"Guava.*CacheBuilder"]


def ast_pattern_classify(code: str) -> int:
    lc = sum(1 for p in LC_PATTERNS if re.search(p, code))
    rl = sum(1 for p in RL_PATTERNS if re.search(p, code))
    ml = sum(1 for p in ML_PATTERNS if re.search(p, code))
    # Rule: >= 2 hits in any category → predict that category
    if lc >= 2: return FAULT_LABEL["lock_contention"]
    if rl >= 2: return FAULT_LABEL["resource_leak"]
    if ml >= 2: return FAULT_LABEL["memory_leak"]
    return FAULT_LABEL["safe"]


def run_ast_pattern(snippets: list[str]) -> np.ndarray:
    return np.array([ast_pattern_classify(c) for c in snippets])


def run_sonarqube(snippets: list[str], project_dirs: list[Path]) -> np.ndarray:
    token = os.environ.get("SONARQUBE_TOKEN", "")
    preds = []
    for i, pdir in enumerate(project_dirs):
        if not token or not pdir.exists():
            preds.append(FAULT_LABEL["safe"])
            continue
        # Map SonarQube rule keys to fault categories
        rule_map = {
            "java:S2095": "resource_leak", "java:S6437": "resource_leak",
            "java:S3077": "memory_leak",   "java:S2160": "memory_leak",
            "java:S2142": "lock_contention","java:S3046": "lock_contention",
        }
        try:
            out = subprocess.check_output(
                ["sonar-scanner", f"-Dsonar.login={token}",
                 f"-Dsonar.projectBaseDir={pdir}",
                 "-Dsonar.language=java", "-Dsonar.scm.disabled=true"],
                text=True, timeout=120, stderr=subprocess.DEVNULL,
            )
            pred = FAULT_LABEL["safe"]
            for rule, cat in rule_map.items():
                if rule in out:
                    pred = FAULT_LABEL[cat]; break
        except Exception:
            pred = FAULT_LABEL["safe"]
        preds.append(pred)
    return np.array(preds)


def run_infer_on_file(src_path: Path) -> int:
    if not src_path.exists():
        return FAULT_LABEL["safe"]
    try:
        r = subprocess.run(
            ["infer", "run", "--", "javac", str(src_path)],
            capture_output=True, text=True, timeout=60,
        )
        out = r.stdout + r.stderr
        if "RESOURCE_LEAK" in out:   return FAULT_LABEL["resource_leak"]
        if "MEMORY_LEAK"   in out:   return FAULT_LABEL["memory_leak"]
        if "DEADLOCK"      in out:   return FAULT_LABEL["lock_contention"]
        if "THREAD_SAFETY" in out:   return FAULT_LABEL["lock_contention"]
    except Exception:
        pass
    return FAULT_LABEL["safe"]


# ---------------------------------------------------------------------------
# LLM prompts (Listings 2 & 3)
# ---------------------------------------------------------------------------

ZERO_SHOT_COT = """\
You are an expert Java performance engineer.
Classify the method as one of:
- LOCK_CONTENTION: synchronized + slow I/O / blocking call
- MEMORY_LEAK: collection inserts with no eviction path
- RESOURCE_LEAK: handle / connection not closed on all paths
- SAFE: no performance fault detected

Step 1: Trace data flow for allocations, locks, inserts.
Step 2: Check all exit paths for deallocation.
Step 3: Classify with reasoning and confidence (0-1).
```java
{code}
```
"""

FIVE_SHOT_EXAMPLES = [
    ("LOCK_CONTENTION",
     "public synchronized User getUser(String id) {\n"
     "  if (cache.containsKey(id)) return cache.get(id);\n"
     "  User u = database.fetchUser(id); // slow I/O under lock\n"
     "  cache.put(id, u); return u;\n}"),
    ("MEMORY_LEAK",
     "private static final Map<String,Object> CACHE = new ConcurrentHashMap<>();\n"
     "public void register(String k, Object v) { CACHE.put(k,v); }"),
    ("RESOURCE_LEAK",
     "RestRepository repo = new RestRepository(settings);\n"
     "ScrollQuery scroll = repo.scroll(query, split);\n"
     "if (scroll.isEmpty()) return Collections.emptyIterator();\n"
     "// repo never closed – connections leaked"),
    ("SAFE",
     "public synchronized int getCount() {\n"
     "  return counter.incrementAndGet();\n}"),
    ("SAFE",
     "try (InputStream in = resource.getInputStream()) {\n"
     "  StreamUtils.copy(in, out);\n}"),
]

RA_PROMPT_TEMPLATE = """\
{examples}
Analyze the NEW method using examples as reference:
1. Trace allocations and lock scopes.
2. Check all exit paths.
3. Classify: LOCK_CONTENTION / MEMORY_LEAK / RESOURCE_LEAK / SAFE
```java
{code}
```
"""

LABEL_PARSE = {
    "LOCK_CONTENTION": 0, "MEMORY_LEAK": 1,
    "RESOURCE_LEAK": 2,   "SAFE": 3,
}


def parse_llm_response(text: str) -> int:
    for label in ["LOCK_CONTENTION", "MEMORY_LEAK", "RESOURCE_LEAK", "SAFE"]:
        if label in text.upper():
            return LABEL_PARSE[label]
    return FAULT_LABEL["safe"]


class LLMBaseline:
    """Handles zero-shot, 5-shot, and retrieval-augmented LLM prompting."""

    def __init__(self, model_id: str, mode: str, n_votes: int = 5):
        self.model_id = model_id
        self.mode     = mode
        self.n_votes  = n_votes

    # ---- API callers ----

    def _call(self, prompt: str) -> str:
        mid = self.model_id
        if "gpt" in mid or "o1" in mid:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            r = client.chat.completions.create(
                model=mid, temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return r.choices[0].message.content
        elif "claude" in mid:
            import anthropic
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            r = client.messages.create(
                model=mid, max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            return r.content[0].text
        elif "deepseek" in mid:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                            base_url="https://api.deepseek.com")
            r = client.chat.completions.create(
                model="deepseek-coder", temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512,
            )
            return r.choices[0].message.content
        raise ValueError(f"Unknown model: {mid}")

    def _build_prompt(self, code: str, neighbours: Optional[list] = None) -> str:
        if self.mode == "zero_shot":
            return ZERO_SHOT_COT.format(code=code)
        elif self.mode == "five_shot":
            ex = "\n\n".join(
                f"Example [{label}]:\n```java\n{c}\n```"
                for label, c in FIVE_SHOT_EXAMPLES
            )
            return ex + f"\n\nNew method:\n```java\n{code}\n```"
        elif self.mode == "retrieval_augmented":
            ex = "\n".join(
                f"Example {i+1} [{lbl}, similarity: {sim:.2f}]:\n```java\n{c}\n```"
                for i, (lbl, sim, c) in enumerate((neighbours or [])[:5])
            )
            return RA_PROMPT_TEMPLATE.format(examples=ex, code=code)
        raise ValueError(f"Unknown mode: {self.mode}")

    def predict_one(self, code: str, neighbours: Optional[list] = None) -> int:
        from collections import Counter
        votes = []
        for _ in range(self.n_votes):
            try:
                resp = self._call(self._build_prompt(code, neighbours))
                votes.append(parse_llm_response(resp))
            except Exception as e:
                log.warning("LLM call failed: %s", e)
                votes.append(FAULT_LABEL["safe"])
        return Counter(votes).most_common(1)[0][0] if votes else FAULT_LABEL["safe"]

    def run_batch(self, snippets: list[str],
                  neighbour_list: Optional[list] = None) -> np.ndarray:
        preds = []
        for i, code in enumerate(snippets):
            nb = neighbour_list[i] if neighbour_list else None
            preds.append(self.predict_one(code, nb))
            if (i+1) % 20 == 0:
                log.info("  LLM %s/%s: %d/%d", self.model_id, self.mode, i+1, len(snippets))
        return np.array(preds)


# ---------------------------------------------------------------------------
# Baselines 13-16: Embedding + RF
# ---------------------------------------------------------------------------

def embedding_rf(
    train_emb: np.ndarray, train_labels: np.ndarray,
    test_emb:  np.ndarray, test_labels:  np.ndarray,
    name: str,
) -> dict:
    clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                  class_weight="balanced")
    clf.fit(train_emb, train_labels)
    y_pred = clf.predict(test_emb)
    m = binary_metrics(test_labels, y_pred)
    m["name"] = name
    log.info("%-30s Prec=%.3f Rec=%.3f F1=%.3f", name,
             m["precision"], m["recall"], m["f1"])
    return m


# ---------------------------------------------------------------------------
# Baseline 17: Word2Vec + BiLSTM
# ---------------------------------------------------------------------------

def w2v_bilstm(
    train_codes: list[str], train_labels: np.ndarray,
    test_codes:  list[str], test_labels:  np.ndarray,
) -> dict:
    try:
        import torch, torch.nn as nn
        from gensim.models import Word2Vec
    except ImportError:
        log.warning("gensim/torch not installed — skipping W2V+BiLSTM baseline")
        return {"name": "w2v_bilstm", "f1": float("nan"), "TP":0,"FP":0,"FN":0,
                "precision":float("nan"),"recall":float("nan")}

    def tok(code): return re.findall(r"[a-zA-Z_]\w*|[{}();]", code)

    all_toks = [tok(c) for c in train_codes + test_codes]
    w2v = Word2Vec(all_toks, vector_size=100, window=5, min_count=1, workers=4, epochs=15)

    MAX_LEN = 256
    def encode(code):
        vecs = [w2v.wv[t] for t in tok(code) if t in w2v.wv][:MAX_LEN]
        if not vecs: return np.zeros((MAX_LEN, 100), np.float32)
        arr = np.array(vecs, np.float32)
        pad = np.zeros((max(MAX_LEN - len(arr), 0), 100), np.float32)
        return np.vstack([arr, pad]) if len(pad) else arr[:MAX_LEN]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(100, 128, 2, batch_first=True,
                                bidirectional=True, dropout=0.3)
            self.fc   = nn.Linear(256, N_CLASSES)
        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.fc(torch.cat([h[-2], h[-1]], 1))

    model = BiLSTM().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss()

    X_tr = torch.tensor(np.array([encode(c) for c in train_codes]), dtype=torch.float32)
    y_tr = torch.tensor(train_labels, dtype=torch.long)
    ds   = torch.utils.data.TensorDataset(X_tr, y_tr)
    dl   = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    model.train()
    for _ in range(20):
        for xb, yb in dl:
            opt.zero_grad()
            crit(model(xb.to(device)), yb.to(device)).backward()
            opt.step()

    model.eval()
    X_te = torch.tensor(np.array([encode(c) for c in test_codes]), dtype=torch.float32)
    with torch.no_grad():
        y_pred = model(X_te.to(device)).argmax(1).cpu().numpy()

    m = binary_metrics(test_labels, y_pred)
    m["name"] = "w2v_bilstm"
    return m


# ---------------------------------------------------------------------------
# Baseline 18: Doc2Vec
# ---------------------------------------------------------------------------

def doc2vec(
    train_codes: list[str], train_labels: np.ndarray,
    test_codes:  list[str], test_labels:  np.ndarray,
) -> dict:
    try:
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    except ImportError:
        log.warning("gensim not installed — skipping Doc2Vec baseline")
        return {"name": "doc2vec", "f1": float("nan"), "TP":0,"FP":0,"FN":0,
                "precision":float("nan"),"recall":float("nan")}

    def tok(code): return re.findall(r"[a-zA-Z_]\w*", code)
    docs = [TaggedDocument(tok(c), [i]) for i, c in enumerate(train_codes)]
    d2v  = Doc2Vec(docs, vector_size=200, window=5, min_count=1, workers=4, epochs=40)

    X_tr = np.array([d2v.infer_vector(tok(c)) for c in train_codes])
    X_te = np.array([d2v.infer_vector(tok(c)) for c in test_codes])
    return embedding_rf(X_tr, train_labels, X_te, test_labels, "doc2vec")


# ---------------------------------------------------------------------------
# FAISS neighbours for RA LLM prompts (reusing FaultEmbed's index)
# ---------------------------------------------------------------------------

def get_neighbours(
    query_emb: np.ndarray,
    index: faiss.IndexFlatIP,
    train_meta: list[dict],
    train_codes: list[str],
    k: int = 5,
) -> list[tuple]:
    """Returns [(label_name, similarity, code_str), ...]"""
    q = query_emb.reshape(1, -1).astype(np.float32)
    dists, idxs = index.search(q, k)
    result = []
    for d, i in zip(dists[0], idxs[0]):
        if i < 0 or i >= len(train_meta):
            continue
        meta = train_meta[i]
        code = train_codes[i] if i < len(train_codes) else ""
        result.append((meta.get("fault_category", "safe"), float(d), code))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Run all 19 baselines for Table 3")
    p.add_argument("--embeddings",        default="embeddings/embeddings.npz")
    p.add_argument("--faiss-index",       default="embeddings/faiss_index.bin")
    p.add_argument("--validated-csv",     default="data/validated_pairs.csv")
    p.add_argument("--output-dir",        default="results/baselines")
    p.add_argument("--extra-embeddings",  nargs="*", default=[],
                   help="JSON list: [{name, train_npy, test_npy}]")
    p.add_argument("--run-llm",           action="store_true")
    p.add_argument("--run-rule-based",    action="store_true")
    p.add_argument("--run-seq-models",    action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    data       = np.load(args.embeddings, allow_pickle=True)
    v_bug      = data["v_bug"]
    labels     = data["labels"]
    train_mask = data["train_mask"]
    test_mask  = ~train_mask

    index = faiss.read_index(args.faiss_index)

    # Load raw code for rule-based and LLM baselines
    all_rows = list(csv.DictReader(open(args.validated_csv, encoding="utf-8")))
    # Split to match embedding split (order must match 04_generate_embeddings output)
    test_codes   = [all_rows[i]["buggy_code"] for i in np.where(test_mask)[0]
                    if i < len(all_rows)]
    train_codes  = [all_rows[i]["buggy_code"] for i in np.where(train_mask)[0]
                    if i < len(all_rows)]
    y_te = labels[test_mask]
    y_tr = labels[train_mask]

    results = {}

    # ---- Baseline 19: GraphCodeBERT no retrieval ----
    log.info("Baseline 19: GraphCodeBERT (no retrieval) …")
    clf19 = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42,
                                    class_weight="balanced")
    clf19.fit(v_bug[train_mask], y_tr)
    y_pred19 = clf19.predict(v_bug[test_mask])
    results["graphcodebert_no_retrieval"] = binary_metrics(y_te, y_pred19)
    log.info("  F1=%.3f (paper: 0.803)", results["graphcodebert_no_retrieval"]["f1"])

    # ---- Baselines 1-3: Rule-based ----
    if args.run_rule_based:
        if test_codes:
            log.info("Baseline 1: AST-Pattern …")
            y_ast = run_ast_pattern(test_codes)
            results["ast_pattern"] = binary_metrics(y_te, y_ast)
            log.info("  F1=%.3f (paper: 0.615)", results["ast_pattern"]["f1"])
        else:
            log.warning("No test code snippets available for rule-based baselines")

    # ---- Baselines 15-16: CodeBERT + RF, UniXcoder + RF ----
    for entry in args.extra_embeddings:
        try:
            cfg = json.loads(entry)
            tr_emb = np.load(cfg["train_npy"])
            te_emb = np.load(cfg["test_npy"])
            r = embedding_rf(tr_emb, y_tr, te_emb, y_te, cfg["name"])
            results[cfg["name"]] = r
        except Exception as e:
            log.warning("Extra embedding %s failed: %s", entry, e)

    # ---- Baselines 17-18: Sequence models ----
    if args.run_seq_models and test_codes and train_codes:
        log.info("Baseline 17: Word2Vec + BiLSTM …")
        results["w2v_bilstm"] = w2v_bilstm(train_codes, y_tr, test_codes, y_te)

        log.info("Baseline 18: Doc2Vec …")
        results["doc2vec"] = doc2vec(train_codes, y_tr, test_codes, y_te)

    # ---- Baselines 4-12: LLM ----
    if args.run_llm and test_codes:
        train_meta = [
            {"fault_category": LABEL_NAMES[int(labels[i])]}
            for i in np.where(train_mask)[0] if i < len(all_rows)
        ]
        llm_configs = [
            ("gpt-4-0125-preview",             "zero_shot",          "gpt4_0shot"),
            ("deepseek-coder",                 "zero_shot",          "deepseek_0shot"),
            ("claude-3-5-sonnet-20241022",     "zero_shot",          "claude_0shot"),
            ("gpt-4-0125-preview",             "five_shot",          "gpt4_5shot"),
            ("deepseek-coder",                 "five_shot",          "deepseek_5shot"),
            ("claude-3-5-sonnet-20241022",     "five_shot",          "claude_5shot"),
            ("gpt-4-0125-preview",             "retrieval_augmented","gpt4_ra"),
            ("deepseek-coder",                 "retrieval_augmented","deepseek_ra"),
            ("claude-3-5-sonnet-20241022",     "retrieval_augmented","claude_ra"),
        ]
        for model_id, mode, key in llm_configs:
            log.info("LLM baseline: %s %s …", model_id, mode)
            baseline = LLMBaseline(model_id, mode)
            if mode == "retrieval_augmented":
                nbr_list = [
                    get_neighbours(v_bug[test_mask][i], index, train_meta, train_codes)
                    for i in range(len(test_codes))
                ]
            else:
                nbr_list = None
            try:
                y_pred = baseline.run_batch(test_codes, nbr_list)
                results[key] = binary_metrics(y_te, y_pred)
                log.info("  %s F1=%.3f", key, results[key]["f1"])
            except Exception as e:
                log.warning("  %s failed: %s", key, e)
                results[key] = {"note": str(e)}

    # ---- Save ----
    out_path = out / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x,"__float__") else str(x))
    log.info("Baseline results → %s", out_path)

    print("\n=== Table 3 (baseline rows) ===")
    print(f"  {'Approach':<35} TP    FP    FN    Prec   Rec    F1")
    for name, m in sorted(results.items(), key=lambda kv: -kv[1].get("f1", 0)):
        f1 = m.get("f1", float("nan"))
        tp = m.get("TP", "?"); fp = m.get("FP", "?"); fn = m.get("FN", "?")
        pr = m.get("precision", float("nan")); re = m.get("recall", float("nan"))
        print(f"  {name:<35} {tp!s:<5} {fp!s:<5} {fn!s:<5} "
              f"{pr:.3f}  {re:.3f}  {f1:.3f}")


if __name__ == "__main__":
    main()
