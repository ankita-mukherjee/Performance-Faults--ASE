#!/usr/bin/env python3
"""
FaultEmbed - Step 2: Method-Level Before/After Extractor (Full Taxonomy)
=========================================================================
Extracts method-level before/after Java source snapshots from candidate commits.
Detects structural changes mapped to the full fault taxonomy.

Uses tree-sitter (preferred) or regex fallback for Java parsing.

Usage:
  pip install tree-sitter tree-sitter-java
  python 02_extract_methods.py \
    --repos-dir ./repos \
    --candidates output/candidate_commits.json \
    --config configs/fault_taxonomy.json \
    --output output/method_pairs.json
"""

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser

    HAS_TREESITTER = True
except ImportError:
    HAS_TREESITTER = False
    print("[WARN] Install tree-sitter: pip install tree-sitter tree-sitter-java")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class MethodPair:
    pair_id: str
    repo: str
    commit_hash: str
    file_path: str
    method_name: str
    class_name: str
    before_source: str
    after_source: str
    before_line_start: int = 0
    before_line_end: int = 0
    after_line_start: int = 0
    after_line_end: int = 0
    before_token_count: int = 0
    after_token_count: int = 0
    # Taxonomy-aware tagging
    candidate_categories: list = field(default_factory=list)
    candidate_parents: list = field(default_factory=list)
    is_multi_fault: bool = False
    confidence: str = "low"
    # Structural changes detected in before→after diff
    structural_changes: list = field(default_factory=list)
    # Mapped fault categories based on structural analysis
    detected_fault_categories: list = field(default_factory=list)
    # Multi-fault indicators
    multi_fault_signals: list = field(default_factory=list)
    # ── PR / commit traceability (paper requirement) ──────────────────────────
    # Three columns required by the paper:
    #   pr_number  : GitHub PR number (string) or "" if not linked to a PR
    #   pr_url     : Full URL to the GitHub PR  (or "" if N/A)
    #   commit_url : Full URL to this commit on GitHub
    # pr_details is a single human-readable summary combining the three columns,
    # useful when exporting to CSV / spreadsheet for quick inspection.
    pr_number: str = ""
    pr_url: str = ""
    commit_url: str = ""
    pr_title: str = ""
    pr_details: str = ""  # ← synthesised column: "PR #N · <title> · <pr_url>"


# ── Git Helpers ────────────────────────────────────────────────────────────────


def run_git(repo_path, args):
    try:
        result = subprocess.run(
            ["git", "-C", repo_path] + args,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        return result.stdout if result.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""


def get_file_at_commit(repo_path, commit_hash, file_path):
    return run_git(repo_path, ["show", f"{commit_hash}:{file_path}"])


def get_changed_hunks(repo_path, parent, commit, file_path):
    diff = run_git(repo_path, ["diff", "-U0", parent, commit, "--", file_path])
    hunks = []
    for m in re.finditer(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff):
        hunks.append(
            {
                "old_start": int(m.group(1)),
                "old_count": int(m.group(2) or 1),
                "new_start": int(m.group(3)),
                "new_count": int(m.group(4) or 1),
            }
        )
    return hunks


# ── Java Parsing ───────────────────────────────────────────────────────────────


def create_java_parser():
    if not HAS_TREESITTER:
        return None
    return Parser(Language(tsjava.language()))


# def extract_methods_treesitter(parser, source):
#     if not parser or not source:
#         return []
#     tree = parser.parse(bytes(source, "utf8"))
#     methods = []

#     def find_class(node):
#         cur = node.parent
#         while cur:
#             if cur.type in (
#                 "class_declaration",
#                 "interface_declaration",
#                 "enum_declaration",
#             ):
#                 for ch in cur.children:
#                     if ch.type == "identifier":
#                         return ch.text.decode("utf8")
#             cur = cur.parent
#         return "Unknown"

#     def traverse(node):
#         if node.type == "method_declaration":
#             name = ""
#             params = ""
#             for ch in node.children:
#                 if ch.type == "identifier":
#                     name = ch.text.decode("utf8")
#                 elif ch.type == "formal_parameters":
#                     params = ch.text.decode("utf8")
#             cls = find_class(node)
#             methods.append(
#                 {
#                     "name": name,
#                     "class_name": cls,
#                     "start_line": node.start_point[0] + 1,
#                     "end_line": node.end_point[0] + 1,
#                     "source": source[node.start_byte : node.end_byte],
#                     "signature": f"{cls}.{name}{params}",
#                 }
#             )
#         for ch in node.children:
#             traverse(ch)

#     traverse(tree.root_node)
#     return methods


def extract_methods_treesitter(parser, source):
    if not parser or not source:
        return []
    tree = parser.parse(bytes(source, "utf8"))
    methods = []

    def find_class(node):
        cur = node.parent
        while cur:
            if cur.type in (
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
            ):
                for ch in cur.children:
                    if ch.type == "identifier":
                        return ch.text.decode("utf8")
            cur = cur.parent
        return "Unknown"

    # ✅ Iterative traversal instead of recursive
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type == "method_declaration":
            name = ""
            params = ""
            for ch in node.children:
                if ch.type == "identifier":
                    name = ch.text.decode("utf8")
                elif ch.type == "formal_parameters":
                    params = ch.text.decode("utf8")
            cls = find_class(node)
            methods.append(
                {
                    "name": name,
                    "class_name": cls,
                    "start_line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "source": source[node.start_byte : node.end_byte],
                    "signature": f"{cls}.{name}{params}",
                }
            )
        # Push children onto stack (reversed to preserve left-to-right order)
        stack.extend(reversed(node.children))

    return methods


def extract_methods_regex(source):
    methods = []
    lines = source.split("\n")
    pattern = re.compile(
        r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:synchronized\s+)?(?:final\s+)?"
        r"(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[^{]+)?\s*\{",
        re.MULTILINE,
    )
    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            brace = 0
            end = i
            for j in range(i, len(lines)):
                brace += lines[j].count("{") - lines[j].count("}")
                if brace == 0 and j > i:
                    end = j
                    break
            methods.append(
                {
                    "name": m.group(1),
                    "class_name": "Unknown",
                    "start_line": i + 1,
                    "end_line": end + 1,
                    "source": "\n".join(lines[i : end + 1]),
                    "signature": f"Unknown.{m.group(1)}({m.group(2)})",
                }
            )
    return methods


def match_methods(before_methods, after_methods, hunks):
    """Match before/after methods affected by hunks."""

    def overlaps(method, hunks, start_key, count_key):
        for h in hunks:
            h_end = h[start_key] + h[count_key] - 1
            if method["start_line"] <= h_end and method["end_line"] >= h[start_key]:
                return True
        return False

    changed_before = {
        m["signature"]
        for m in before_methods
        if overlaps(m, hunks, "old_start", "old_count")
    }
    changed_after = {
        m["signature"]
        for m in after_methods
        if overlaps(m, hunks, "new_start", "new_count")
    }

    b_by_sig = {m["signature"]: m for m in before_methods}
    a_by_sig = {m["signature"]: m for m in after_methods}

    matched_sigs = (changed_before | changed_after) & set(b_by_sig) & set(a_by_sig)
    pairs = []
    for sig in matched_sigs:
        if b_by_sig[sig]["source"].strip() != a_by_sig[sig]["source"].strip():
            pairs.append((b_by_sig[sig], a_by_sig[sig]))
    return pairs


# ── Full Taxonomy Structural Change Detector ───────────────────────────────────

STRUCTURAL_CHANGE_RULES = [
    # ── LOOPS ──
    {
        "id": "LOOP_OPTIMIZATION",
        "category": "loops.inefficient_loops",
        "before": r"for\s*\(|while\s*\(",
        "after": r"\.stream\(\)|\.parallelStream\(\)|\.forEach\(",
        "parent": "loops",
    },
    {
        "id": "REDUNDANT_TRAVERSAL_FIX",
        "category": "loops.redundant_traversal",
        "before": r"for.*\{[\s\S]*?\}[\s\S]*?for.*\{",
        "after": r"(single|combined|merged)",
        "parent": "loops",
    },
    {
        "id": "INFINITE_LOOP_FIX",
        "category": "loops.infinite_loop",
        "before": r"while\s*\(\s*true\s*\)",
        "after": r"break\s*;|while\s*\([^)]*[<>=!]",
        "parent": "loops",
    },
    # ── REDUNDANT COMPUTATION ──
    {
        "id": "DATA_STRUCTURE_UPGRADE",
        "category": "redundant_computation.inefficient_data_structure",
        "before": r"new\s+(LinkedList|Vector|Hashtable|Stack)",
        "after": r"new\s+(ArrayList|ArrayDeque|HashMap|ConcurrentHashMap)",
        "parent": "redundant_computation",
    },
    {
        "id": "NEGATIVE_CACHE_ADDED",
        "category": "redundant_computation.searching_nonexistent_objects",
        "before": r"\.get\(.*\)\s*==\s*null|\.containsKey",
        "after": r"computeIfAbsent|getOrDefault|Optional",
        "parent": "redundant_computation",
    },
    {
        "id": "DYNAMIC_CHECK_REMOVED",
        "category": "redundant_computation.unnecessary_dynamic_checking",
        "before": r"instanceof\s+\w+",
        "after": r"generic|typed|@SuppressWarnings",
        "parent": "redundant_computation",
    },
    # ── MEMORY ──
    {
        "id": "MEMORY_BLOAT_FIX",
        "category": "memory.memory_bloat",
        "before": r"new\s+String\(|new\s+Integer|new\s+Long|\+\s*\"",
        "after": r"StringBuilder|StringBuffer|Integer\.valueOf|\.intern\(\)",
        "parent": "memory",
    },
    {
        "id": "MEMORY_LEAK_FIX",
        "category": "memory.memory_leak",
        "before": r"\.(add|put)\(",
        "after": r"WeakReference|SoftReference|\.remove\(|\.clear\(|maximumSize|expireAfter",
        "parent": "memory",
    },
    {
        "id": "EXCESSIVE_MEMORY_FIX",
        "category": "memory.excessive_memory_consumption",
        "before": r"\.readAllBytes\(\)|\.readAllLines\(\)|toArray\(\)|collect\(Collectors\.toList",
        "after": r"BufferedReader|\.lines\(\)|Stream\.|iterator\(\)|paginate|batch|chunk",
        "parent": "memory",
    },
    # ── SYNCHRONIZATION ──
    {
        "id": "LOOP_IN_CRITICAL_SECTION_FIX",
        "category": "synchronization.nested_loops_in_critical_sections",
        "before": r"synchronized.*\{[\s\S]*?(for|while)\s*\(",
        "after": r"ConcurrentHashMap|CopyOnWriteArrayList|\.lock\(\)[\s\S]{1,100}\.unlock\(\)",
        "parent": "synchronization",
    },
    {
        "id": "DEADLOCK_FIX_LOCK_ORDERING",
        "category": "synchronization.improper_locking",
        "before": r"synchronized.*\{[\s\S]*?synchronized",
        "after": r"tryLock|\.getId\(\).*[<>]|identityHashCode|compareTo",
        "parent": "synchronization",
    },
    {
        "id": "SYNC_TO_ATOMIC",
        "category": "synchronization.inefficient_synchronization",
        "before": r"synchronized",
        "after": r"Atomic(Integer|Long|Boolean|Reference)|\.incrementAndGet|\.compareAndSet",
        "parent": "synchronization",
    },
    {
        "id": "SYNC_METHOD_TO_BLOCK",
        "category": "synchronization.inefficient_synchronization",
        "before": r"public\s+synchronized\s+",
        "after": r"synchronized\s*\(\s*\w+\s*\)\s*\{",
        "parent": "synchronization",
    },
    {
        "id": "EXCLUSIVE_TO_READWRITE",
        "category": "synchronization.inefficient_synchronization",
        "before": r"synchronized|ReentrantLock",
        "after": r"ReadWriteLock|StampedLock|readLock\(\)|writeLock\(\)",
        "parent": "synchronization",
    },
    {
        "id": "HASHMAP_TO_CONCURRENT",
        "category": "synchronization.inefficient_synchronization",
        "before": r"synchronized[\s\S]*?HashMap|HashMap[\s\S]*?synchronized",
        "after": r"ConcurrentHashMap",
        "parent": "synchronization",
    },
    {
        "id": "LOCK_SCOPE_REDUCTION",
        "category": "synchronization.inefficient_synchronization",
        "before": r"synchronized",
        "after": r"synchronized",
        "extra_check": "lock_scope_reduced",
        "parent": "synchronization",
    },
    # ── ENERGY ──
    {
        "id": "POLLING_TO_EVENT",
        "category": "energy.energy_leaks",
        "before": r"while.*Thread\.sleep|while.*\{[\s\S]*?poll|busy.*wait|spin.*wait",
        "after": r"ScheduledExecutorService|CompletableFuture|\.await\(|BlockingQueue\.take|Observer|Listener",
        "parent": "energy",
    },
    # ── RESOURCE ──
    {
        "id": "UNBOUNDED_THREADS_FIX",
        "category": "resource.resource_excessive_consumption",
        "before": r"new\s+Thread\(|Executors\.newCachedThreadPool",
        "after": r"Executors\.newFixedThreadPool|ThreadPoolExecutor|maxPoolSize|corePoolSize",
        "parent": "resource",
    },
    {
        "id": "RESOURCE_LEAK_FIX",
        "category": "resource.resource_leak",
        "before": r"new\s+\w*(Stream|Reader|Writer|Connection|Socket|Channel)\s*\(",
        "after": r"try\s*\(|finally\s*\{[\s\S]*?\.close\(|IOUtils\.closeQuietly",
        "parent": "resource",
    },
    {
        "id": "RESOURCE_OVERUSE_FIX",
        "category": "resource.resource_overuse",
        "before": r"new\s+Socket\(|new\s+ServerSocket\(|Runtime\.exec",
        "after": r"NIO|Channel|Selector|ProcessBuilder",
        "parent": "resource",
    },
    {
        "id": "ADDED_CLOSE_CALL",
        "category": "resource.resource_leak",
        "before": r"new\s+\w*(Stream|Connection|Socket)",
        "after": r"\.close\(\)",
        "parent": "resource",
    },
]


def detect_structural_changes(before_src: str, after_src: str) -> list[dict]:
    """Run all structural change rules against a before/after pair."""
    changes = []
    for rule in STRUCTURAL_CHANGE_RULES:
        try:
            before_match = re.search(
                rule["before"], before_src, re.MULTILINE | re.DOTALL
            )
            after_match = re.search(rule["after"], after_src, re.MULTILINE | re.DOTALL)

            if not (before_match and after_match):
                continue

            # Extra checks for special rules
            if rule.get("extra_check") == "lock_scope_reduced":
                # Check if synchronized block got shorter
                before_sync_len = (
                    len(re.findall(r"\S", before_src.split("synchronized")[-1][:200]))
                    if "synchronized" in before_src
                    else 0
                )
                after_sync_len = (
                    len(re.findall(r"\S", after_src.split("synchronized")[-1][:200]))
                    if "synchronized" in after_src
                    else 0
                )
                if before_sync_len <= after_sync_len:
                    continue

            changes.append(
                {
                    "id": rule["id"],
                    "category": rule["category"],
                    "parent": rule["parent"],
                }
            )
        except re.error:
            pass

    return changes


# ── Multi-Fault Signal Detection ──────────────────────────────────────────────

MULTI_FAULT_RULES = [
    # 2-fault combinations
    {
        "id": "ML+LC:synchronized_hashmap_to_concurrent",
        "categories": [
            "memory.memory_leak",
            "synchronization.inefficient_synchronization",
        ],
        "parents": ["memory", "synchronization"],
        "before_all": [r"synchronized", r"HashMap"],
        "after_all": [r"ConcurrentHashMap"],
        "after_any": [r"computeIfAbsent", r"putIfAbsent"],
    },
    {
        "id": "ML+LC:global_lock_unbounded_collection",
        "categories": [
            "memory.memory_leak",
            "synchronization.inefficient_synchronization",
        ],
        "parents": ["memory", "synchronization"],
        "before_all": [r"synchronized", r"\.(add|put)\("],
        "before_none": [r"\.remove\(", r"\.clear\(", r"maximumSize"],
        "after_any": [r"\.remove\(", r"maximumSize", r"expireAfter", r"\.clear\("],
    },
    {
        "id": "ML+RL:unclosed_resources_in_collection",
        "categories": ["memory.memory_leak", "resource.resource_leak"],
        "parents": ["memory", "resource"],
        "before_all": [r"\.(add|put)\(", r"(Stream|Connection|Channel|Socket)"],
        "before_none": [r"\.close\(\)"],
        "after_any": [r"\.close\(\)", r"try\s*\(", r"finally"],
    },
    {
        "id": "LC+RL:io_inside_lock_no_cleanup",
        "categories": [
            "synchronization.inefficient_synchronization",
            "resource.resource_leak",
        ],
        "parents": ["synchronization", "resource"],
        "before_all": [
            r"synchronized|\.lock\(\)",
            r"new\s+\w*(Stream|Connection|Socket)",
        ],
        "before_none": [r"finally", r"try\s*\("],
        "after_any": [r"try\s*\(", r"finally", r"\.close\(\)"],
    },
    {
        "id": "DL+LC:nested_locks_to_concurrent",
        "categories": [
            "synchronization.improper_locking",
            "synchronization.inefficient_synchronization",
        ],
        "parents": ["synchronization"],
        "before_check": lambda b, a: b.count("synchronized") >= 2,
        "after_any": [r"ConcurrentHashMap", r"tryLock"],
    },
    {
        "id": "LOOP+LC:loop_in_sync_optimized",
        "categories": [
            "loops.inefficient_loops",
            "synchronization.nested_loops_in_critical_sections",
        ],
        "parents": ["loops", "synchronization"],
        "before_all": [r"synchronized[\s\S]*?(for|while)\s*\("],
        "after_any": [r"\.stream\(\)", r"ConcurrentHashMap", r"CopyOnWriteArrayList"],
    },
    {
        "id": "BLOAT+ENERGY:object_creation_in_hot_loop",
        "categories": ["memory.memory_bloat", "energy.energy_leaks"],
        "parents": ["memory", "energy"],
        "before_all": [r"(for|while)[\s\S]*?new\s+\w+\("],
        "after_any": [r"pool|cache|reuse|StringBuilder|\.intern\(\)"],
    },
    {
        "id": "MEMCONS+RL:full_read_unclosed_stream",
        "categories": ["memory.excessive_memory_consumption", "resource.resource_leak"],
        "parents": ["memory", "resource"],
        "before_all": [r"\.readAllBytes\(\)|\.readAllLines\(\)"],
        "before_none": [r"try\s*\(", r"\.close\(\)"],
        "after_any": [r"try\s*\(", r"BufferedReader", r"\.lines\(\)"],
    },
    {
        "id": "REXC+LC:unbounded_pool_with_lock",
        "categories": [
            "resource.resource_excessive_consumption",
            "synchronization.inefficient_synchronization",
        ],
        "parents": ["resource", "synchronization"],
        "before_all": [
            r"new\s+Thread\(|newCachedThreadPool",
            r"synchronized|\.lock\(\)",
        ],
        "after_any": [r"newFixedThreadPool|ThreadPoolExecutor", r"ConcurrentHashMap"],
    },
    # 3-fault combinations
    {
        "id": "LOOP+BLOAT+ENERGY:inefficient_loop_with_allocation",
        "categories": [
            "loops.inefficient_loops",
            "memory.memory_bloat",
            "energy.energy_leaks",
        ],
        "parents": ["loops", "memory", "energy"],
        "before_all": [r"(for|while)", r"new\s+(String|Integer|Long|Double)\("],
        "after_any": [r"StringBuilder|valueOf|stream|cached"],
    },
    {
        "id": "DL+LC+RL:nested_locks_holding_resources",
        "categories": [
            "synchronization.improper_locking",
            "synchronization.inefficient_synchronization",
            "resource.resource_leak",
        ],
        "parents": ["synchronization", "resource"],
        "before_all": [
            r"synchronized[\s\S]*?synchronized",
            r"new\s+\w*(Stream|Connection)",
        ],
        "before_none": [r"finally", r"try\s*\("],
        "after_any": [r"tryLock|\.getId\(\)", r"\.close\(\)|try\s*\("],
    },
]


def detect_multi_fault_signals(before_src: str, after_src: str) -> list[dict]:
    """Detect multi-fault patterns in a before/after pair."""
    signals = []
    for rule in MULTI_FAULT_RULES:
        matched = True

        # Check before_all: all must match
        for pattern in rule.get("before_all", []):
            if not re.search(pattern, before_src, re.DOTALL):
                matched = False
                break

        if not matched:
            continue

        # Check before_none: none must match
        for pattern in rule.get("before_none", []):
            if re.search(pattern, before_src, re.DOTALL):
                matched = False
                break

        if not matched:
            continue

        # Check before_check lambda
        if "before_check" in rule:
            if not rule["before_check"](before_src, after_src):
                continue

        # Check after_any: at least one must match
        after_any = rule.get("after_any", [])
        if after_any:
            if not any(re.search(p, after_src, re.DOTALL) for p in after_any):
                continue

        # Check after_all: all must match
        for pattern in rule.get("after_all", []):
            if not re.search(pattern, after_src, re.DOTALL):
                matched = False
                break

        if matched:
            signals.append(
                {
                    "id": rule["id"],
                    "categories": rule["categories"],
                    "parents": rule["parents"],
                }
            )

    return signals


# ── Token Estimation ───────────────────────────────────────────────────────────


def estimate_tokens(source):
    return len(re.findall(r"\w+|[^\w\s]", source))


# ── PR Details Helper ──────────────────────────────────────────────────────────


def _build_pr_details(candidate: dict) -> str:
    """
    Build a single human-readable 'pr_details' string combining the three
    traceability columns required by the paper:
      pr_number | pr_url | commit_url
    Format: "PR #<n> · <pr_title> | PR: <pr_url> | Commit: <commit_url>"
    If no PR was found the field degrades to just the commit URL.
    """
    pr_num = candidate.get("pr_number") or ""
    pr_url = candidate.get("pr_url") or ""
    pr_title = (candidate.get("pr_title") or "")[:80]
    commit_url = (
        candidate.get("commit_url")
        or f"https://github.com/{candidate['repo']}/commit/{candidate['commit_hash']}"
    )

    if pr_num and pr_url:
        title_part = f" · {pr_title}" if pr_title else ""
        return f"PR #{pr_num}{title_part} | PR: {pr_url} | Commit: {commit_url}"
    return f"Commit: {commit_url}"


# ── Main Pipeline ──────────────────────────────────────────────────────────────


def process_candidate(candidate, repos_dir, parser, project_map):
    repo_name = candidate["repo"]
    local_dir = project_map.get(repo_name, {}).get(
        "local_dir", repo_name.split("/")[-1]
    )
    repo_path = os.path.join(repos_dir, local_dir)
    if not os.path.isdir(repo_path):
        return []

    pairs = []
    for file_path in candidate.get("changed_java_files", []):
        before_content = get_file_at_commit(
            repo_path, candidate["parent_hash"], file_path
        )
        after_content = get_file_at_commit(
            repo_path, candidate["commit_hash"], file_path
        )
        if not before_content or not after_content:
            continue

        hunks = get_changed_hunks(
            repo_path, candidate["parent_hash"], candidate["commit_hash"], file_path
        )
        if not hunks:
            continue

        if parser:
            before_methods = extract_methods_treesitter(parser, before_content)
            after_methods = extract_methods_treesitter(parser, after_content)
        else:
            before_methods = extract_methods_regex(before_content)
            after_methods = extract_methods_regex(after_content)

        matched = match_methods(before_methods, after_methods, hunks)

        for before_m, after_m in matched:
            structural = detect_structural_changes(
                before_m["source"], after_m["source"]
            )
            multi_signals = detect_multi_fault_signals(
                before_m["source"], after_m["source"]
            )

            # Collect all detected fault categories
            detected_cats = list(
                set(
                    [s["category"] for s in structural]
                    + [c for ms in multi_signals for c in ms["categories"]]
                )
            )

            pair_id = hashlib.sha256(
                f"{repo_name}:{candidate['commit_hash']}:{file_path}:{before_m['signature']}".encode()
            ).hexdigest()[:16]

            pair = MethodPair(
                pair_id=pair_id,
                repo=repo_name,
                commit_hash=candidate["commit_hash"],
                file_path=file_path,
                method_name=before_m["name"],
                class_name=before_m["class_name"],
                before_source=before_m["source"],
                after_source=after_m["source"],
                before_line_start=before_m["start_line"],
                before_line_end=before_m["end_line"],
                after_line_start=after_m["start_line"],
                after_line_end=after_m["end_line"],
                before_token_count=estimate_tokens(before_m["source"]),
                after_token_count=estimate_tokens(after_m["source"]),
                candidate_categories=candidate.get("all_categories", []),
                candidate_parents=candidate.get("all_parents", []),
                is_multi_fault=candidate.get("is_multi_fault", False)
                or len(multi_signals) > 0,
                confidence=candidate.get("confidence", "low"),
                structural_changes=[s["id"] for s in structural],
                detected_fault_categories=detected_cats,
                multi_fault_signals=[s["id"] for s in multi_signals],
                # ── PR traceability ───────────────────────────────────────────
                pr_number=str(candidate.get("pr_number") or ""),
                pr_url=candidate.get("pr_url") or "",
                commit_url=candidate.get("commit_url")
                or f"https://github.com/{repo_name}/commit/{candidate['commit_hash']}",
                pr_title=candidate.get("pr_title") or "",
                pr_details=_build_pr_details(candidate),
            )
            pairs.append(pair)

    return pairs


def _map_to_paper_category(detected_cats: list) -> str:
    """
    Map the broad internal taxonomy categories to the three paper labels:
      LC  – Lock Contention  (synchronization.*)
      ML  – Memory Leak      (memory.memory_leak, memory.memory_bloat)
      RL  – Resource Leak    (resource.resource_leak, resource.*)
    Returns the highest-priority match, or the raw first category if no
    mapping exists.  Multi-fault pairs get all matching labels joined by '+'.
    """
    mapping = {
        "synchronization": "LC",
        "memory.memory_leak": "ML",
        "memory.memory_bloat": "ML",
        "memory.excessive_memory_consumption": "ML",
        "resource.resource_leak": "RL",
        "resource.resource_excessive_consumption": "RL",
        "resource.resource_overuse": "RL",
    }
    labels = []
    for cat in detected_cats:
        for key, label in mapping.items():
            if cat.startswith(key) and label not in labels:
                labels.append(label)
    if labels:
        return "+".join(labels)
    return detected_cats[0] if detected_cats else "UNKNOWN"


def main():
    ap = argparse.ArgumentParser(
        description="FaultEmbed: Extract method pairs (full taxonomy)"
    )
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--config", required=True, help="Path to fault_taxonomy.json")
    ap.add_argument("--output", default="output/method_pairs.json")
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    project_map = config.get("target_projects", {})

    with open(args.candidates) as f:
        data = json.load(f)
    candidates = data["candidates"]
    logger.info(f"Loaded {len(candidates)} candidates")

    parser = create_java_parser()
    logger.info(f"Parser: {'tree-sitter' if parser else 'regex fallback'}")

    all_pairs = []
    truncated = 0
    for i, cand in enumerate(candidates):
        if i % 100 == 0 and i > 0:
            logger.info(
                f"  Processed {i}/{len(candidates)}, {len(all_pairs)} pairs so far"
            )
        pairs = process_candidate(cand, args.repos_dir, parser, project_map)
        for p in pairs:
            if (
                p.before_token_count > args.max_tokens
                or p.after_token_count > args.max_tokens
            ):
                truncated += 1
            all_pairs.append(p)

    # Summary stats
    cat_counts = defaultdict(int)
    parent_counts = defaultdict(int)
    multi_signal_counts = defaultdict(int)
    for p in all_pairs:
        for cat in p.detected_fault_categories:
            cat_counts[cat] += 1
            parent_counts[cat.split(".")[0]] += 1
        for ms in p.multi_fault_signals:
            multi_signal_counts[ms] += 1

    output_data = {
        "metadata": {
            "total_pairs": len(all_pairs),
            "multi_fault_pairs": sum(1 for p in all_pairs if p.is_multi_fault),
            "truncated_512": truncated,
            "by_parent_category": dict(parent_counts),
            "by_fault_category": dict(cat_counts),
            "multi_fault_signal_counts": dict(multi_signal_counts),
        },
        "pairs": [asdict(p) for p in all_pairs],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # ── CSV export: before_source | after_source | pr_details ─────────────────
    # The paper requires these three columns for the published dataset.
    # Additional context columns (repo, commit, method, fault category) are
    # included so the CSV is self-contained for manual review / JMH harness work.
    csv_path = args.output.replace(".json", "_pairs.csv")
    csv_fields = [
        "pair_id",
        "repo",
        "commit_hash",
        "file_path",
        "class_name",
        "method_name",
        "fault_category",  # top detected fault (LC / ML / RL aligned)
        "candidate_parents",
        "confidence",
        "before_source",  # ← Column 1 required by paper
        "after_source",  # ← Column 2 required by paper
        "pr_details",  # ← Column 3 required by paper (pr_number|pr_url|commit_url)
        # Individual sub-columns for programmatic access
        "pr_number",
        "pr_url",
        "commit_url",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for p in all_pairs:
            pd = asdict(p)
            # Map the broad taxonomy to the paper's three fault classes
            fault_category = _map_to_paper_category(
                pd.get("detected_fault_categories", [])
            )
            pd["fault_category"] = fault_category
            pd["candidate_parents"] = "|".join(pd.get("candidate_parents", []))
            writer.writerow(pd)

    logger.info("=" * 70)
    logger.info(f"EXTRACTION COMPLETE: {len(all_pairs)} method pairs")
    logger.info(f"  Multi-fault: {output_data['metadata']['multi_fault_pairs']}")
    logger.info(f"  Exceeding 512 tokens: {truncated}")
    logger.info(f"  By parent: {dict(parent_counts)}")
    logger.info(f"  Multi-fault signals: {dict(multi_signal_counts)}")
    logger.info(f"Output JSON : {args.output}")
    logger.info(
        f"Output CSV  : {csv_path}  ← before_source | after_source | pr_details"
    )


if __name__ == "__main__":
    main()
