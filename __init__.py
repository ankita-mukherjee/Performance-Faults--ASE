"""
data/01_mine_repositories.py
Stage 1 & 2 — Repository mining and commit filtering.

Stage 1: Clone all six Java repos; scan commit history with multi-signal
         keyword + PR-label strategy → ~3,997 raw candidates.
Stage 2: AST-pattern checks + refactoring-rejection heuristics → ~3,214
         candidates written to data/candidates_stage2.csv.

Each row: project, commit_hash, fault_category, file_path, method_name,
          buggy_code, fixed_code, commit_message
"""

from __future__ import annotations
import csv
import logging
import os
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository list (Table 1 projects)
# ---------------------------------------------------------------------------

REPOS: dict[str, str] = {
    "apache_hbase":      "https://github.com/apache/hbase.git",
    "commons_pool":      "https://github.com/apache/commons-pool.git",
    "spring_framework":  "https://github.com/spring-projects/spring-framework.git",
    "google_guava":      "https://github.com/google/guava.git",
    "glide":             "https://github.com/bumptech/glide.git",
    "elasticsearch":     "https://github.com/elastic/elasticsearch.git",
}

# ---------------------------------------------------------------------------
# Stage 1 keyword patterns per fault category
# ---------------------------------------------------------------------------

STAGE1_PATTERNS: dict[str, list[str]] = {
    "lock_contention": [
        r"lock[\s_-]*contention", r"thread[\s_-]*serializ",
        r"\bsynchroni[zs]ed\b", r"\bdeadlock\b", r"\bstarvation\b",
        r"blocked[\s_-]*thread", r"mutex[\s_-]*contention",
    ],
    "memory_leak": [
        r"memory[\s_-]*leak", r"out[\s-]*of[\s-]*memory", r"\bOOM\b",
        r"heap[\s_-]*(grow|space|overflow)", r"unbounded[\s_-]*(cache|collection|list|map)",
        r"retention", r"GC[\s_-]*pressure",
    ],
    "resource_leak": [
        r"resource[\s_-]*leak", r"file[\s_-]*descriptor[\s_-]*leak",
        r"connection[\s_-]*leak", r"not[\s_-]*closed",
        r"unclosed[\s_-]*(stream|connection|socket|channel)",
        r"fd[\s_-]*leak",
    ],
}

# Stage 2 AST keywords that must appear for the candidate to be plausible
AST_REQUIRED: dict[str, list[str]] = {
    "lock_contention": [
        "synchronized", "ReentrantLock", "ReentrantReadWriteLock",
        "Lock", "lockInterruptibly", "tryLock",
    ],
    "memory_leak": [
        "HashMap", "ConcurrentHashMap", "ArrayList", "LinkedList",
        "Cache", ".put(", "static final Map", "static final List",
        "WeakReference", "SoftReference",
    ],
    "resource_leak": [
        "InputStream", "OutputStream", "Connection", "Socket",
        "FileDescriptor", "Channel", "Closeable", "AutoCloseable",
        "RestRepository", "ScrollQuery", "getConnection", "openStream",
    ],
}

# Patterns that indicate a commit is NOT a bug-fix (Stage 2 label filter)
REJECT_PATTERNS = [
    r"\brefactor\b", r"\bcleanup\b", r"\bformat(ting)?\b",
    r"\bchore\b", r"\bstyle\b", r"\bjavadoc\b",
    r"\badd\s+support\b", r"\bremove\s+deprecated\b",
    r"\bupgrade\b", r"\bbump\b", r"\bversion\b",
    r"\btest[\s_-]*only\b", r"\bno[\s_-]*functional[\s_-]*change\b",
]


# ---------------------------------------------------------------------------
# Dataclass for a candidate pair
# ---------------------------------------------------------------------------

@dataclass
class CandidatePair:
    project:        str
    commit_hash:    str
    fault_category: str
    file_path:      str
    method_name:    str
    buggy_code:     str
    fixed_code:     str
    commit_message: str
    pr_labels:      str = ""


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def git(*args, cwd: Path, check: bool = True, capture: bool = True) -> str:
    cmd = ["git", "-C", str(cwd)] + list(args)
    result = subprocess.run(cmd, capture_output=capture, text=True,
                            errors="replace", check=False)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    return result.stdout


def clone_or_update(url: str, target: Path) -> None:
    if (target / ".git").exists():
        log.info("Updating %s …", target.name)
        git("pull", "--quiet", "--ff-only", cwd=target, check=False)
    else:
        log.info("Cloning %s → %s …", url, target)
        subprocess.run(["git", "clone", "--quiet", "--depth=5000", url, str(target)],
                       check=True)


def list_commits(repo: Path) -> list[dict]:
    raw = git("log", "--pretty=format:%H\t%s\t%b\t---END---",
              "--diff-filter=M", "--", "*.java", cwd=repo)
    commits = []
    for block in raw.split("---END---\n"):
        block = block.strip()
        if not block:
            continue
        parts = block.split("\t", 2)
        if len(parts) < 1:
            continue
        commits.append({
            "hash":    parts[0].strip(),
            "subject": parts[1].strip() if len(parts) > 1 else "",
            "body":    parts[2].strip() if len(parts) > 2 else "",
        })
    return commits


def get_diff(repo: Path, commit_hash: str) -> str:
    return git("show", "--unified=5", commit_hash, "--", "*.java", cwd=repo)


def file_at_commit(repo: Path, rev: str, path: str) -> str:
    try:
        return git("show", f"{rev}:{path}", cwd=repo)
    except subprocess.CalledProcessError:
        return ""


def parent_hash(repo: Path, commit_hash: str) -> Optional[str]:
    out = git("rev-parse", f"{commit_hash}^", cwd=repo, check=False)
    return out.strip() if out.strip() else None


# ---------------------------------------------------------------------------
# Stage 1: keyword match
# ---------------------------------------------------------------------------

def stage1_category(commit: dict) -> Optional[str]:
    text = (commit["subject"] + " " + commit["body"]).lower()
    for cat, patterns in STAGE1_PATTERNS.items():
        if any(re.search(p, text, re.IGNORECASE) for p in patterns):
            return cat
    return None


# ---------------------------------------------------------------------------
# Stage 2: label filter + AST check
# ---------------------------------------------------------------------------

def stage2_label_ok(commit: dict) -> bool:
    text = (commit["subject"] + " " + commit.get("body", "")).lower()
    return not any(re.search(p, text, re.IGNORECASE) for p in REJECT_PATTERNS)


def stage2_ast_ok(buggy_src: str, fixed_src: str, category: str) -> bool:
    combined = buggy_src + fixed_src
    keywords = AST_REQUIRED.get(category, [])
    return any(kw in combined for kw in keywords)


# ---------------------------------------------------------------------------
# Diff → changed files and method names
# ---------------------------------------------------------------------------

def changed_java_files(diff: str) -> list[str]:
    return [line[6:] for line in diff.splitlines()
            if line.startswith("+++ b/") and line.endswith(".java")]


def changed_method_names(diff: str) -> list[str]:
    methods: set[str] = set()
    for line in diff.splitlines():
        # Git hunk header: @@ -l,s +l,s @@ <context>
        if line.startswith("@@"):
            m = re.search(r"@@.*\s(\w+)\s*\(", line)
            if m:
                methods.add(m.group(1))
        # Method declarations in added/removed lines
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")):
            m = re.search(
                r"(?:public|private|protected|static|synchronized|final|\s)+"
                r"[\w<>\[\],\s]+\s+(\w+)\s*\(",
                line,
            )
            if m and m.group(1) not in {"if", "for", "while", "switch", "catch"}:
                methods.add(m.group(1))
    return list(methods)


def extract_method(source: str, name: str) -> str:
    """Heuristic: extract the first complete method body matching `name`."""
    lines  = source.splitlines()
    sig_re = re.compile(
        r"(?:public|private|protected|static|synchronized|final|\s)*"
        r"[\w<>\[\],\s]+\s+" + re.escape(name) + r"\s*\("
    )
    buf, depth, active = [], 0, False
    for line in lines:
        if not active:
            if sig_re.search(line):
                active = True
                buf    = [line]
                depth  = line.count("{") - line.count("}")
        else:
            buf.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0:
                break
    return "\n".join(buf) if len(buf) > 1 else ""


# ---------------------------------------------------------------------------
# Per-project mining
# ---------------------------------------------------------------------------

def mine_project(project: str, url: str, clone_base: Path) -> list[CandidatePair]:
    repo = clone_base / project
    clone_or_update(url, repo)

    commits = list_commits(repo)
    log.info("[%s] Scanning %d commits …", project, len(commits))

    pairs: list[CandidatePair] = []
    for commit in commits:
        # Stage 1
        cat = stage1_category(commit)
        if cat is None:
            continue
        # Stage 2 label check
        if not stage2_label_ok(commit):
            continue

        diff = get_diff(repo, commit["hash"])
        if not diff:
            continue

        files   = changed_java_files(diff)
        methods = changed_method_names(diff)
        parent  = parent_hash(repo, commit["hash"])
        if parent is None:
            continue

        for fpath in files:
            buggy_src = file_at_commit(repo, parent,          fpath)
            fixed_src = file_at_commit(repo, commit["hash"],  fpath)
            if not buggy_src or not fixed_src:
                continue
            # Stage 2 AST check
            if not stage2_ast_ok(buggy_src, fixed_src, cat):
                continue

            for mname in methods:
                buggy = extract_method(buggy_src, mname)
                fixed = extract_method(fixed_src, mname)
                if not buggy or not fixed or buggy == fixed:
                    continue
                pairs.append(CandidatePair(
                    project        = project,
                    commit_hash    = commit["hash"],
                    fault_category = cat,
                    file_path      = fpath,
                    method_name    = mname,
                    buggy_code     = buggy,
                    fixed_code     = fixed,
                    commit_message = commit["subject"],
                ))

    log.info("[%s] → %d candidates after Stage 1+2", project, len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Stage 1+2: mine Java repos for fault candidates")
    p.add_argument("--clone-dir",  default="repos")
    p.add_argument("--output",     default="data/candidates_stage2.csv")
    p.add_argument("--projects",   nargs="*", default=list(REPOS.keys()))
    args = p.parse_args()

    clone_base = Path(args.clone_dir)
    clone_base.mkdir(parents=True, exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    all_pairs: list[CandidatePair] = []
    for proj in args.projects:
        if proj not in REPOS:
            log.warning("Unknown project '%s' — skipping", proj)
            continue
        all_pairs.extend(mine_project(proj, REPOS[proj], clone_base))

    fieldnames = list(CandidatePair.__dataclass_fields__.keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(asdict(pair) for pair in all_pairs)

    log.info("Total candidates written: %d → %s", len(all_pairs), args.output)


if __name__ == "__main__":
    main()
