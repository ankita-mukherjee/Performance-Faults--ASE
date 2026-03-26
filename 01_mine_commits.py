#!/usr/bin/env python3
"""
FaultEmbed - Step 1: Repository Mining (Full Taxonomy)
=======================================================
Mines candidate bug-fix commits covering ALL runtime fault categories:
  Loops | Redundant Computation | Memory | Synchronization | Energy | Resource

Uses three-layer detection:
  Layer 1: Git log keyword matching (commit messages)
  Layer 2: Structural pattern matching (diff content)
  Layer 3: GitHub API enrichment (issue/PR labels)

Usage:
  python 01_mine_commits.py --repos-dir ./repos --config configs/fault_taxonomy.json
  python 01_mine_commits.py --repos-dir ./repos --config configs/fault_taxonomy.json --github-token TOKEN
"""

import argparse
import json
import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from github import Github

    HAS_GITHUB = True
except ImportError:
    HAS_GITHUB = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# ── Fault Filter ───────────────────────────────────────────────────────────────
# Maps the 3 paper fault classes to the internal taxonomy parent/category keys.

PAPER_CLASS_TO_INTERNAL = {
    "LC": {
        "parents": {"synchronization"},
        "categories": {
            "synchronization.inefficient_synchronization",
            "synchronization.improper_locking",
            "synchronization.nested_loops_in_critical_sections",
        },
    },
    "ML": {
        "parents": {"memory"},
        "categories": {
            "memory.memory_leak",
            "memory.memory_bloat",
            "memory.excessive_memory_consumption",
        },
    },
    "RL": {
        "parents": {"resource"},
        "categories": {
            "resource.resource_leak",
            "resource.resource_excessive_consumption",
            "resource.resource_overuse",
        },
    },
}

VALID_FILTER_TOKENS = {"LC", "ML", "RL", "all"}


def parse_fault_filter(raw: str) -> tuple[set, bool]:
    """
    Parse --fault-filter value.

    Examples:
        "LC"       → only lock contention commits
        "ML+RL"    → dual-fix commits with BOTH memory leak AND resource leak
        "LC+ML+RL" → triple-fix commits with all three
        "all"      → no filter (default)

    Returns:
        required_classes : set of paper-class labels that must be present
        combination_mode : True = ALL classes must be present (dual/triple-fix)
                           False = ANY one class is enough (single-class filter)
    """
    if raw.strip().lower() == "all":
        return set(), False

    tokens = [t.strip().upper() for t in raw.split("+")]
    unknown = set(tokens) - VALID_FILTER_TOKENS
    if unknown:
        raise ValueError(
            f"Unknown fault class(es): {unknown}. "
            f"Valid: LC, ML, RL, all  — combine with '+', e.g. ML+RL"
        )
    required = set(tokens)
    combination_mode = len(required) >= 2
    return required, combination_mode


def candidate_matches_filter(
    candidate, required_classes: set, combination_mode: bool
) -> bool:
    """
    True if the candidate satisfies the fault filter.

    Single class  (e.g. --fault-filter LC):
        candidate must have at least one LC signal.
    Combination   (e.g. --fault-filter ML+RL):
        candidate must have BOTH ML and RL signals — genuine dual-fix.
    No filter     (required_classes is empty):
        always True.
    """
    if not required_classes:
        return True

    candidate_classes = set()
    for cat in candidate.all_categories:
        for paper_class, mapping in PAPER_CLASS_TO_INTERNAL.items():
            if cat in mapping["categories"] or cat.split(".")[0] in mapping["parents"]:
                candidate_classes.add(paper_class)

    if combination_mode:
        # ALL requested classes must be present
        return required_classes.issubset(candidate_classes)
    else:
        # ANY of the requested classes is enough
        return bool(required_classes & candidate_classes)


# ── Data Structures ────────────────────────────────────────────────────────────


@dataclass
class FaultSignal:
    """A detected fault signal with category and evidence."""

    category: str  # e.g., "memory.memory_leak"
    subcategory: str  # e.g., "memory_leak"
    parent: str  # e.g., "memory"
    source: str  # "keyword_msg", "keyword_diff", "structural", "github"
    evidence: list = field(default_factory=list)  # matched patterns


@dataclass
class CandidateCommit:
    """A candidate bug-fix commit with full taxonomy tagging."""

    repo: str
    commit_hash: str
    parent_hash: str
    message: str
    author: str
    date: str
    changed_java_files: list = field(default_factory=list)
    # Multi-layer signals
    signals: list = field(default_factory=list)  # list of FaultSignal dicts
    github_labels: list = field(default_factory=list)
    issue_ids: list = field(default_factory=list)
    # Derived taxonomy tags
    all_categories: list = field(default_factory=list)  # unique full paths
    all_parents: list = field(default_factory=list)  # unique parent categories
    # Multi-fault detection
    is_multi_fault: bool = False  # 2+ different PARENT categories
    multi_fault_parents: list = field(default_factory=list)  # which parents overlap
    fault_count: int = 0  # how many distinct subcategories
    # Confidence
    confidence: str = "low"
    signal_count: int = 0
    # ── PR / commit traceability (paper requirement) ──────────────────────────
    # pr_number  : GitHub PR number that introduced this fix (None if not found)
    # pr_url     : Full GitHub URL to the PR
    # commit_url : Full GitHub URL to this specific commit
    pr_number: Optional[str] = None
    pr_url: str = ""
    commit_url: str = ""
    pr_title: str = ""
    pr_body_snippet: str = ""  # first 300 chars of PR description


# ── Taxonomy Loader ────────────────────────────────────────────────────────────


class FaultTaxonomy:
    """Loads and provides access to the fault taxonomy config."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)
        self.taxonomy = self.config["taxonomy"]
        self.dependencies = self.config.get("fault_dependencies", {})
        self.projects = self.config.get("target_projects", {})

        # Build flat lookup: subcategory_key → {parent, keywords, patterns}
        self.subcategories = {}
        for parent_key, parent_val in self.taxonomy.items():
            for sub_key, sub_val in parent_val.get("subcategories", {}).items():
                full_key = f"{parent_key}.{sub_key}"
                self.subcategories[full_key] = {
                    "parent": parent_key,
                    "parent_label": parent_val["label"],
                    "label": sub_val["label"],
                    "description": sub_val.get("description", ""),
                    "commit_keywords": sub_val.get("commit_keywords", []),
                    "structural_patterns_before": sub_val.get(
                        "structural_patterns_before", []
                    ),
                    "structural_patterns_after": sub_val.get(
                        "structural_patterns_after", []
                    ),
                    "sub_patterns": sub_val.get("sub_patterns", {}),
                }

        logger.info(
            f"Loaded taxonomy: {len(self.subcategories)} subcategories "
            f"across {len(self.taxonomy)} parent categories"
        )

    def match_keywords(self, text: str) -> list[FaultSignal]:
        """Match text against all subcategory keywords."""
        signals = []
        text_lower = text.lower()
        for full_key, sub in self.subcategories.items():
            matches = []
            for kw in sub["commit_keywords"]:
                if re.search(kw.lower(), text_lower):
                    matches.append(kw)
            if matches:
                signals.append(
                    FaultSignal(
                        category=full_key,
                        subcategory=full_key.split(".")[-1],
                        parent=sub["parent"],
                        source="keyword",
                        evidence=matches,
                    )
                )
        return signals

    def match_structural(self, diff_text: str) -> list[FaultSignal]:
        """Match diff against structural before/after patterns."""
        signals = []
        # Split diff into removed (-) and added (+) lines
        removed_lines = "\n".join(
            line[1:]
            for line in diff_text.split("\n")
            if line.startswith("-") and not line.startswith("---")
        )
        added_lines = "\n".join(
            line[1:]
            for line in diff_text.split("\n")
            if line.startswith("+") and not line.startswith("+++")
        )

        for full_key, sub in self.subcategories.items():
            evidence = []

            # Check "before" patterns in removed lines
            for pattern in sub["structural_patterns_before"]:
                try:
                    if re.search(pattern, removed_lines, re.MULTILINE | re.DOTALL):
                        evidence.append(f"before:{pattern}")
                except re.error:
                    pass

            # Check "after" patterns in added lines
            for pattern in sub["structural_patterns_after"]:
                try:
                    if re.search(pattern, added_lines, re.MULTILINE | re.DOTALL):
                        evidence.append(f"after:{pattern}")
                except re.error:
                    pass

            # Also check sub_patterns (e.g., synchronization anti-patterns)
            for sp_key, sp_val in sub.get("sub_patterns", {}).items():
                for pattern in sp_val.get("patterns_before", []):
                    try:
                        if re.search(pattern, removed_lines, re.MULTILINE):
                            evidence.append(f"subpattern:{sp_key}:before:{pattern}")
                    except re.error:
                        pass
                for pattern in sp_val.get("patterns_after", []):
                    try:
                        if re.search(pattern, added_lines, re.MULTILINE):
                            evidence.append(f"subpattern:{sp_key}:after:{pattern}")
                    except re.error:
                        pass

            # Require at least one before + one after pattern for structural signal
            has_before = any(
                e.startswith("before:") or ":before:" in e for e in evidence
            )
            has_after = any(e.startswith("after:") or ":after:" in e for e in evidence)

            if has_before and has_after:
                signals.append(
                    FaultSignal(
                        category=full_key,
                        subcategory=full_key.split(".")[-1],
                        parent=sub["parent"],
                        source="structural",
                        evidence=evidence,
                    )
                )
            elif len(evidence) >= 2:
                # Multiple one-sided signals still worth flagging (lower confidence)
                signals.append(
                    FaultSignal(
                        category=full_key,
                        subcategory=full_key.split(".")[-1],
                        parent=sub["parent"],
                        source="structural_partial",
                        evidence=evidence,
                    )
                )

        return signals


# ── Git Helpers ────────────────────────────────────────────────────────────────


def run_git(repo_path: str, args: list) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", repo_path] + args, capture_output=True, timeout=120
        )
        if result.returncode != 0:
            return ""
        return result.stdout.decode("utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def get_all_commits(repo_path: str) -> list[dict]:
    SEP = "|||"
    log_format = f"%H{SEP}%P{SEP}%an{SEP}%aI{SEP}%s"
    output = run_git(
        repo_path, ["log", "--all", f"--format={log_format}", "--no-merges"]
    )
    commits = []
    if not output:
        return commits
    for line in output.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(SEP)
        if len(parts) < 5:
            continue
        commits.append(
            {
                "hash": parts[0].strip(),
                "parent": parts[1].strip().split()[0] if parts[1].strip() else "",
                "author": parts[2].strip(),
                "date": parts[3].strip(),
                "message": parts[4].strip(),
            }
        )
    return commits


def get_changed_java_files(
    repo_path: str, commit_hash: str, parent_hash: str
) -> list[str]:
    if not parent_hash:
        return []
    output = run_git(
        repo_path,
        [
            "diff",
            "--name-only",
            "--diff-filter=M",
            parent_hash,
            commit_hash,
            "--",
            "*.java",
        ],
    )
    return [f.strip() for f in output.strip().split("\n") if f.strip()]


def get_commit_diff(repo_path: str, commit_hash: str, parent_hash: str) -> str:
    if not parent_hash:
        return ""
    return run_git(repo_path, ["diff", parent_hash, commit_hash, "--", "*.java"])


def extract_issue_ids(message: str) -> list[str]:
    ids = []
    ids.extend(re.findall(r"#(\d+)", message))
    ids.extend(re.findall(r"([A-Z]+-\d+)", message))
    ids.extend(
        re.findall(r"(?:fixes|resolves|closes)\s+#?(\d+)", message, re.IGNORECASE)
    )
    return list(set(ids))


# ── GitHub API ─────────────────────────────────────────────────────────────────


def mine_github_issues(github_token: str, owner_repo: str) -> tuple[dict, dict, set]:
    """
    Returns:
      issue_map        : {issue_number_str → {labels, title}}
      pr_map           : {commit_sha → {pr_number, pr_url, pr_title, pr_body_snippet}}
      merged_sha_set   : set of all commit SHAs that belong to a MERGED PR.
                         Used to filter out unmerged/experimental commits so that
                         every before/after pair in the dataset has a confirmed fix.
    """
    if not HAS_GITHUB or not github_token:
        return {}, {}, set()
    logger.info(f"  Fetching GitHub issues/PRs for {owner_repo}...")
    g = Github(github_token)
    repo = g.get_repo(owner_repo)

    # ── Issue map ──────────────────────────────────────────────────────────────
    issue_map = {}
    target_labels = {
        "bug",
        "performance",
        "concurrency",
        "leak",
        "deadlock",
        "memory",
        "resource",
        "threading",
        "synchronization",
        "optimization",
        "energy",
        "cpu",
        "loop",
    }
    for label_name in target_labels:
        try:
            for issue in repo.get_issues(
                state="closed", labels=[repo.get_label(label_name)]
            ):
                issue_map[str(issue.number)] = {
                    "labels": [l.name for l in issue.labels],
                    "title": issue.title,
                }
        except Exception:
            continue

    # ── PR map: commit SHA → PR details ───────────────────────────────────────
    # Only walk MERGED PRs. This guarantees:
    #   - The fix was reviewed and accepted (not just proposed)
    #   - The "after" code snapshot is the confirmed, validated fix
    #   - No experimental branches or abandoned fixes pollute the dataset
    pr_map = {}
    merged_sha_set = set()
    try:
        for pr in repo.get_pulls(state="closed", sort="updated", direction="desc"):
            if not pr.merged:
                # Skip unmerged closed PRs (rejected, abandoned, superseded)
                continue
            try:
                for commit in pr.get_commits():
                    pr_map[commit.sha] = {
                        "pr_number": str(pr.number),
                        "pr_url": pr.html_url,
                        "pr_title": pr.title or "",
                        "pr_body_snippet": (pr.body or "")[:300],
                    }
                    merged_sha_set.add(commit.sha)
            except Exception:
                continue
            if len(pr_map) >= 10_000:  # safety cap — raise if needed
                break
    except Exception as exc:
        logger.warning(f"  PR fetch error: {exc}")

    logger.info(
        f"    → {len(issue_map)} issues, "
        f"{len(pr_map)} merged-PR commits ({len(merged_sha_set)} unique SHAs)"
    )
    return issue_map, pr_map, merged_sha_set


# ── Confidence & Multi-Fault Scoring ──────────────────────────────────────────


def compute_candidate(
    repo: str,
    commit: dict,
    java_files: list,
    all_signals: list[FaultSignal],
    github_labels: list,
    issue_ids: list,
    pr_info: dict = None,
) -> CandidateCommit:
    """Build a CandidateCommit with multi-fault analysis and PR traceability."""

    # Deduplicate categories
    cat_set = set()
    parent_set = set()
    for sig in all_signals:
        cat_set.add(sig.category)
        parent_set.add(sig.parent)

    # Multi-fault: 2+ different PARENT categories affected
    is_multi = len(parent_set) >= 2
    multi_parents = sorted(parent_set) if is_multi else []

    # Count signal layers
    sources = set(sig.source for sig in all_signals)
    layer_count = len(sources & {"keyword", "structural", "github"})

    # Confidence scoring
    has_structural = "structural" in sources
    has_keyword = "keyword" in sources
    has_github = "github" in sources
    total_signals = len(all_signals)

    if has_structural and has_keyword and (has_github or total_signals >= 4):
        confidence = "high"
    elif (has_structural and has_keyword) or (has_keyword and has_github):
        confidence = "medium"
    else:
        confidence = "low"

    # ── Build commit URL and PR traceability columns ──────────────────────────
    # commit_url is always constructable from repo + sha
    commit_url = f"https://github.com/{repo}/commit/{commit['hash']}"

    pr_number = None
    pr_url = ""
    pr_title = ""
    pr_body_snippet = ""
    if pr_info:
        pr_number = pr_info.get("pr_number")
        pr_url = pr_info.get("pr_url", "")
        pr_title = pr_info.get("pr_title", "")
        pr_body_snippet = pr_info.get("pr_body_snippet", "")

    return CandidateCommit(
        repo=repo,
        commit_hash=commit["hash"],
        parent_hash=commit["parent"],
        message=commit["message"],
        author=commit["author"],
        date=commit["date"],
        changed_java_files=java_files,
        signals=[asdict(s) for s in all_signals],
        github_labels=github_labels,
        issue_ids=issue_ids,
        all_categories=sorted(cat_set),
        all_parents=sorted(parent_set),
        is_multi_fault=is_multi,
        multi_fault_parents=multi_parents,
        fault_count=len(cat_set),
        confidence=confidence,
        signal_count=total_signals,
        pr_number=pr_number,
        pr_url=pr_url,
        commit_url=commit_url,
        pr_title=pr_title,
        pr_body_snippet=pr_body_snippet,
    )


# ── Main Mining Loop ──────────────────────────────────────────────────────────


def mine_repo(
    repo_path: str,
    repo_name: str,
    taxonomy: FaultTaxonomy,
    github_token: str = None,
    required_classes: set = None,
    combination_mode: bool = False,
    no_merge_filter: bool = False,
) -> list[CandidateCommit]:
    logger.info(f"Mining {repo_name}...")
    if required_classes:
        mode_str = "ALL of" if combination_mode else "ANY of"
        logger.info(f"  Fault filter: {mode_str} {sorted(required_classes)}")
    else:
        logger.info("  Fault filter: all (no filter)")

    commits = get_all_commits(repo_path)
    logger.info(f"  {len(commits)} non-merge commits")

    # Pre-fetch GitHub issue map, PR map, and merged SHA set
    issue_map = {}
    pr_map = {}
    merged_sha_set = set()
    if github_token:
        issue_map, pr_map, merged_sha_set = mine_github_issues(github_token, repo_name)
        if no_merge_filter:
            logger.info(
                "  --no-merge-filter set: PR data fetched for URL enrichment only "
                "(merged-PR restriction is OFF — all commits are candidates)"
            )
        else:
            logger.info(
                f"  Filtering to merged-PR commits only "
                f"({len(merged_sha_set)} merged SHAs from GitHub)"
            )
    else:
        logger.info("  No GitHub token — processing all commits (no merged-PR filter)")

    candidates = []
    skipped_unmerged = 0
    skipped_filter = 0

    for i, commit in enumerate(commits):
        if i % 1000 == 0 and i > 0:
            logger.info(
                f"  Processed {i}/{len(commits)} commits | "
                f"{len(candidates)} kept | "
                f"{skipped_unmerged} skipped (unmerged) | "
                f"{skipped_filter} skipped (fault filter)"
            )

        # ── MERGED-ONLY FILTER ────────────────────────────────────────────────
        # Skipped entirely when --no-merge-filter is set.
        # Before/after pairs are always extracted from git regardless — the
        # parent commit is always the "before" and the commit itself is "after".
        if (
            not no_merge_filter
            and merged_sha_set
            and commit["hash"] not in merged_sha_set
        ):
            skipped_unmerged += 1
            continue

        # Layer 1: Keyword match on commit message
        msg_signals = taxonomy.match_keywords(commit["message"])

        # Quick check: does it touch Java files?
        java_files = get_changed_java_files(repo_path, commit["hash"], commit["parent"])
        if not java_files:
            continue

        # Layer 2: Structural match on diff
        diff = get_commit_diff(repo_path, commit["hash"], commit["parent"])
        diff_keyword_signals = taxonomy.match_keywords(diff[:5000])
        structural_signals = taxonomy.match_structural(diff)

        # Layer 3: GitHub label enrichment
        github_signals = []
        github_labels = []
        issue_ids = extract_issue_ids(commit["message"])
        for iid in issue_ids:
            if iid in issue_map:
                github_labels.extend(issue_map[iid].get("labels", []))

        all_signals = (
            msg_signals + diff_keyword_signals + structural_signals + github_signals
        )
        if not all_signals:
            continue

        # Deduplicate: keep strongest signal per category
        best_per_cat = {}
        source_priority = {
            "structural": 3,
            "keyword": 2,
            "structural_partial": 1,
            "github": 2,
        }
        for sig in all_signals:
            key = sig.category
            priority = source_priority.get(sig.source, 0)
            if key not in best_per_cat or priority > source_priority.get(
                best_per_cat[key].source, 0
            ):
                best_per_cat[key] = sig
        deduped_signals = list(best_per_cat.values())

        pr_info = pr_map.get(commit["hash"])
        candidate = compute_candidate(
            repo_name,
            commit,
            java_files,
            deduped_signals,
            github_labels,
            issue_ids,
            pr_info=pr_info,
        )

        # ── FAULT CLASS FILTER ────────────────────────────────────────────────
        if not candidate_matches_filter(
            candidate, required_classes or set(), combination_mode
        ):
            skipped_filter += 1
            continue

        candidates.append(candidate)

    logger.info(
        f"  → {len(candidates)} candidates kept | "
        f"{skipped_unmerged} skipped (unmerged) | "
        f"{skipped_filter} skipped (fault filter) | "
        f"{sum(1 for c in candidates if c.is_multi_fault)} multi-fault"
    )
    return candidates
    logger.info(f"Mining {repo_name}...")
    if required_classes:
        mode_str = "ALL of" if combination_mode else "ANY of"
        logger.info(f"  Fault filter: {mode_str} {sorted(required_classes)}")
    else:
        logger.info("  Fault filter: all (no filter)")

    commits = get_all_commits(repo_path)
    logger.info(f"  {len(commits)} non-merge commits")

    # Pre-fetch GitHub issue map, PR map, and merged SHA set
    issue_map = {}
    pr_map = {}
    merged_sha_set = set()
    if github_token:
        issue_map, pr_map, merged_sha_set = mine_github_issues(github_token, repo_name)
        logger.info(
            f"  Filtering to merged-PR commits only "
            f"({len(merged_sha_set)} merged SHAs from GitHub)"
        )
    else:
        logger.info("  No GitHub token — processing all commits (no merged-PR filter)")

    candidates = []
    skipped_unmerged = 0
    skipped_filter = 0

    for i, commit in enumerate(commits):
        if i % 1000 == 0 and i > 0:
            logger.info(
                f"  Processed {i}/{len(commits)} commits | "
                f"{len(candidates)} kept | "
                f"{skipped_unmerged} skipped (unmerged) | "
                f"{skipped_filter} skipped (fault filter)"
            )

        # ── MERGED-ONLY FILTER ────────────────────────────────────────────────
        if merged_sha_set and commit["hash"] not in merged_sha_set:
            skipped_unmerged += 1
            continue

        # Layer 1: Keyword match on commit message
        msg_signals = taxonomy.match_keywords(commit["message"])

        # Quick check: does it touch Java files?
        java_files = get_changed_java_files(repo_path, commit["hash"], commit["parent"])
        if not java_files:
            continue

        # Layer 2: Structural match on diff
        diff = get_commit_diff(repo_path, commit["hash"], commit["parent"])
        diff_keyword_signals = taxonomy.match_keywords(diff[:5000])
        structural_signals = taxonomy.match_structural(diff)

        # Layer 3: GitHub label enrichment
        github_signals = []
        github_labels = []
        issue_ids = extract_issue_ids(commit["message"])
        for iid in issue_ids:
            if iid in issue_map:
                github_labels.extend(issue_map[iid].get("labels", []))

        all_signals = (
            msg_signals + diff_keyword_signals + structural_signals + github_signals
        )
        if not all_signals:
            continue

        # Deduplicate: keep strongest signal per category
        best_per_cat = {}
        source_priority = {
            "structural": 3,
            "keyword": 2,
            "structural_partial": 1,
            "github": 2,
        }
        for sig in all_signals:
            key = sig.category
            priority = source_priority.get(sig.source, 0)
            if key not in best_per_cat or priority > source_priority.get(
                best_per_cat[key].source, 0
            ):
                best_per_cat[key] = sig
        deduped_signals = list(best_per_cat.values())

        pr_info = pr_map.get(commit["hash"])
        candidate = compute_candidate(
            repo_name,
            commit,
            java_files,
            deduped_signals,
            github_labels,
            issue_ids,
            pr_info=pr_info,
        )

        # ── FAULT CLASS FILTER ────────────────────────────────────────────────
        if not candidate_matches_filter(
            candidate, required_classes or set(), combination_mode
        ):
            skipped_filter += 1
            continue

        candidates.append(candidate)

    logger.info(
        f"  → {len(candidates)} candidates kept | "
        f"{skipped_unmerged} skipped (unmerged) | "
        f"{skipped_filter} skipped (fault filter) | "
        f"{sum(1 for c in candidates if c.is_multi_fault)} multi-fault"
    )
    return candidates


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="FaultEmbed: Mine commits (full taxonomy)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Fault filter examples:
  --fault-filter LC          lock contention only
  --fault-filter ML          memory leak only
  --fault-filter RL          resource leak only
  --fault-filter ML+RL       dual-fix: commits fixing BOTH memory leak AND resource leak
  --fault-filter LC+ML       dual-fix: commits fixing BOTH lock contention AND memory leak
  --fault-filter LC+RL       dual-fix: commits fixing BOTH lock contention AND resource leak
  --fault-filter LC+ML+RL    triple-fix: all three categories together
  --fault-filter all         no filter — keep everything (default)
        """,
    )
    parser.add_argument("--repos-dir", required=True)
    parser.add_argument("--config", required=True, help="Path to fault_taxonomy.json")
    parser.add_argument("--github-token", default=None)
    parser.add_argument("--output", default="output/candidate_commits.json")
    parser.add_argument(
        "--projects", default=None, help="Comma-separated owner/repo to mine"
    )
    parser.add_argument(
        "--min-confidence", default="low", choices=["low", "medium", "high"]
    )
    parser.add_argument(
        "--fault-filter",
        default="all",
        help="Fault class filter: LC, ML, RL, or combinations like ML+RL (default: all)",
    )
    parser.add_argument(
        "--no-merge-filter",
        action="store_true",
        default=False,
        help=(
            "Disable the merged-PR filter. When set, ALL commits matching the "
            "fault signals are kept regardless of whether they belong to a merged PR. "
            "Produces more candidates. Useful for initial exploration before JMH validation."
        ),
    )
    args = parser.parse_args()

    # Parse and validate the fault filter
    try:
        required_classes, combination_mode = parse_fault_filter(args.fault_filter)
    except ValueError as e:
        parser.error(str(e))

    taxonomy = FaultTaxonomy(args.config)

    # Select projects
    if args.projects:
        selected = {
            k: v for k, v in taxonomy.projects.items() if k in args.projects.split(",")
        }
    else:
        selected = taxonomy.projects

    all_candidates = []
    summary = {}

    for owner_repo, proj_info in selected.items():
        local_dir = proj_info["local_dir"]
        repo_path = os.path.join(args.repos_dir, local_dir)
        if not os.path.isdir(repo_path):
            logger.warning(f"  Not found: {repo_path}. Clone first.")
            continue

        candidates = mine_repo(
            repo_path,
            owner_repo,
            taxonomy,
            args.github_token,
            required_classes=required_classes,
            combination_mode=combination_mode,
            no_merge_filter=args.no_merge_filter,
        )

        # Filter by confidence
        conf_order = {"low": 1, "medium": 2, "high": 3}
        min_conf = conf_order[args.min_confidence]
        candidates = [
            c for c in candidates if conf_order.get(c.confidence, 0) >= min_conf
        ]

        all_candidates.extend(candidates)

        # Build per-repo summary
        cat_counts = defaultdict(int)
        parent_counts = defaultdict(int)
        for c in candidates:
            for cat in c.all_categories:
                cat_counts[cat] += 1
            for p in c.all_parents:
                parent_counts[p] += 1

        multi_fault_combos = defaultdict(int)
        for c in candidates:
            if c.is_multi_fault:
                combo = "+".join(c.multi_fault_parents)
                multi_fault_combos[combo] += 1

        summary[owner_repo] = {
            "total": len(candidates),
            "multi_fault": sum(1 for c in candidates if c.is_multi_fault),
            "by_parent": dict(parent_counts),
            "by_category": dict(cat_counts),
            "multi_fault_combos": dict(multi_fault_combos),
            "by_confidence": {
                conf: sum(1 for c in candidates if c.confidence == conf)
                for conf in ["low", "medium", "high"]
            },
        }

    # ── Fault class breakdown ──────────────────────────────────────────────────
    # Map every candidate to its paper-class label(s): LC, ML, RL.
    # A candidate can belong to multiple classes (dual/triple fix).
    # We count:
    #   single_counts  : candidates with EXACTLY one class  → LC only, ML only, RL only
    #   combo_counts   : candidates with 2+ classes         → LC+ML, ML+RL, LC+RL, LC+ML+RL
    #   class_totals   : every candidate that touches a class (single + all combos it appears in)

    PARENT_TO_CLASS = {
        "synchronization": "LC",
        "memory": "ML",
        "resource": "RL",
    }

    def candidate_paper_classes(c) -> frozenset:
        """Return frozenset of paper classes this candidate covers."""
        classes = set()
        for parent in c.all_parents:
            if parent in PARENT_TO_CLASS:
                classes.add(PARENT_TO_CLASS[parent])
        return frozenset(classes)

    single_counts = defaultdict(int)  # {"LC": N, "ML": N, "RL": N}
    combo_counts = defaultdict(int)  # {"LC+ML": N, "ML+RL": N, ...}
    class_totals = defaultdict(int)  # total appearances per class across all candidates

    for c in all_candidates:
        classes = candidate_paper_classes(c)
        if not classes:
            continue
        label = "+".join(sorted(classes))
        if len(classes) == 1:
            single_counts[label] += 1
        else:
            combo_counts[label] += 1
        for cls in classes:
            class_totals[cls] += 1

    # Global multi-fault summary (internal taxonomy combos, kept for JSON detail)
    global_combos = defaultdict(int)
    for c in all_candidates:
        if c.is_multi_fault:
            combo = "+".join(c.multi_fault_parents)
            global_combos[combo] += 1

    fault_breakdown = {
        # Individual counts (single-class only)
        "LC_only": single_counts.get("LC", 0),
        "ML_only": single_counts.get("ML", 0),
        "RL_only": single_counts.get("RL", 0),
        # Dual-fix counts
        "LC+ML": combo_counts.get("LC+ML", 0),
        "LC+RL": combo_counts.get("LC+RL", 0),
        "ML+RL": combo_counts.get("ML+RL", 0),
        # Triple-fix
        "LC+ML+RL": combo_counts.get("LC+ML+RL", 0),
        # Class totals (single + every combo the class appears in)
        "LC_total": class_totals.get("LC", 0),
        "ML_total": class_totals.get("ML", 0),
        "RL_total": class_totals.get("RL", 0),
    }

    output_data = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "repos_mined": list(selected.keys()),
            "taxonomy_subcategories": len(taxonomy.subcategories),
            "total_candidates": len(all_candidates),
            "total_multi_fault": sum(1 for c in all_candidates if c.is_multi_fault),
            "fault_filter": args.fault_filter,
            "combination_mode": combination_mode,
            "merge_filter_active": not args.no_merge_filter,
            "fault_breakdown": fault_breakdown,
            "multi_fault_combos": dict(global_combos),
        },
        "summary": summary,
        "candidates": [asdict(c) for c in all_candidates],
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    # ── Final console report ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info(f"MINING COMPLETE  —  {len(all_candidates)} total candidates")
    logger.info(f"Filter applied   :  {args.fault_filter}")
    logger.info(
        f"Merge filter     :  {'ON (merged PRs only)' if not args.no_merge_filter else 'OFF (all commits)'}"
    )
    logger.info("")
    logger.info("  INDIVIDUAL COUNTS (single-class only)")
    logger.info(f"    LC  (Lock Contention)  : {fault_breakdown['LC_only']:>5}")
    logger.info(f"    ML  (Memory Leak)      : {fault_breakdown['ML_only']:>5}")
    logger.info(f"    RL  (Resource Leak)    : {fault_breakdown['RL_only']:>5}")
    logger.info("")
    logger.info("  DUAL-FIX COUNTS (commit fixes both classes)")
    logger.info(f"    LC + ML                : {fault_breakdown['LC+ML']:>5}")
    logger.info(f"    LC + RL                : {fault_breakdown['LC+RL']:>5}")
    logger.info(f"    ML + RL                : {fault_breakdown['ML+RL']:>5}")
    logger.info("")
    logger.info("  TRIPLE-FIX COUNTS")
    logger.info(f"    LC + ML + RL           : {fault_breakdown['LC+ML+RL']:>5}")
    logger.info("")
    logger.info("  CLASS TOTALS  (single + all combos each class appears in)")
    logger.info(f"    LC total               : {fault_breakdown['LC_total']:>5}")
    logger.info(f"    ML total               : {fault_breakdown['ML_total']:>5}")
    logger.info(f"    RL total               : {fault_breakdown['RL_total']:>5}")
    logger.info("")
    logger.info(f"Output: {args.output}")


if __name__ == "__main__":
    main()
