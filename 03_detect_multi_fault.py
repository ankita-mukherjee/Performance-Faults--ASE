#!/usr/bin/env python3
"""
FaultEmbed - Step 3: Multi-Fault Dependency Detection & Analysis
=================================================================
Analyzes method pairs for dual-fix AND triple-fix instances.
Discovers fault dependency chains across the full taxonomy.

Three detection layers:
  1. Pattern rules (structural heuristics on code changes)
  2. Data-flow analysis (allocation/deallocation/lock lifecycle tracking)
  3. Cross-category co-occurrence (statistical dependency analysis)

Outputs:
  - Detailed dual/triple-fix instances with evidence
  - Fault dependency matrix (which categories co-occur)
  - Dependency chains (transitive relationships)
  - Validation checklist for JMH/dynamic verification

Usage:
  python 03_detect_multi_fault.py \
    --method-pairs output/method_pairs.json \
    --config configs/fault_taxonomy.json \
    --output output/multi_fault_analysis.json
"""

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from itertools import combinations

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Data Flow Analyzer ─────────────────────────────────────────────────────────

@dataclass
class DataFlowProfile:
    """Comprehensive data-flow profile of a Java method."""
    # Resource lifecycle
    resource_allocations: list = field(default_factory=list)
    resource_deallocations: list = field(default_factory=list)
    has_try_with_resources: bool = False
    has_finally: bool = False
    unclosed_resources: int = 0

    # Collection lifecycle
    collection_adds: int = 0
    collection_removes: int = 0
    collection_types: list = field(default_factory=list)
    has_eviction: bool = False
    has_bounding: bool = False

    # Lock lifecycle
    lock_acquires: int = 0
    lock_depth: int = 0
    has_lock_ordering: bool = False
    has_trylock: bool = False
    has_readwrite_lock: bool = False
    has_atomic: bool = False
    sync_scope_chars: int = 0   # rough size of synchronized region

    # I/O and compute in lock
    io_in_lock: bool = False
    loop_in_lock: bool = False
    heavy_compute_in_lock: bool = False

    # Loops
    loop_count: int = 0
    nested_loop_count: int = 0
    loop_has_allocation: bool = False
    loop_has_io: bool = False

    # Memory patterns
    string_concat_in_loop: bool = False
    autoboxing: bool = False
    full_collection_load: bool = False

    # Energy/polling
    polling_pattern: bool = False
    busy_wait: bool = False
    thread_sleep_in_loop: bool = False

    # Threading
    raw_thread_creation: bool = False
    unbounded_pool: bool = False


def analyze_data_flow(source: str) -> DataFlowProfile:
    """Extract comprehensive data-flow profile from Java source."""
    p = DataFlowProfile()

    # Resource allocations
    alloc_re = re.compile(
        r"new\s+(File\w*Stream|Buffered\w*|Input\w*Stream|Output\w*Stream|"
        r"Connection|Socket|Channel|Scanner|PreparedStatement|ResultSet|"
        r"Http\w*|URL\w*|Database\w*)\s*\("
    )
    p.resource_allocations = alloc_re.findall(source)
    dealloc_re = re.compile(r"\.(close|release|dispose|shutdown|disconnect)\s*\(")
    p.resource_deallocations = dealloc_re.findall(source)
    p.has_try_with_resources = bool(re.search(r"try\s*\(", source))
    p.has_finally = "finally" in source
    p.unclosed_resources = max(0, len(p.resource_allocations) - len(p.resource_deallocations))
    if p.has_try_with_resources or p.has_finally:
        p.unclosed_resources = 0  # assume covered

    # Collections
    p.collection_adds = len(re.findall(r"\.(add|put|offer|push|append)\s*\(", source))
    p.collection_removes = len(re.findall(r"\.(remove|poll|pop|clear|evict|invalidate)\s*\(", source))
    p.collection_types = re.findall(r"new\s+(ArrayList|HashMap|HashSet|LinkedList|TreeMap|ConcurrentHashMap|LinkedHashMap)\s*[<(]", source)
    p.has_eviction = bool(re.search(r"maximumSize|expireAfter|removalListener|onEviction", source))
    p.has_bounding = bool(re.search(r"maximumSize|capacity|maxSize|limit", source))

    # Locks
    sync_count = source.count("synchronized")
    lock_calls = len(re.findall(r"\.lock\(\)", source))
    p.lock_acquires = sync_count + lock_calls
    p.has_lock_ordering = bool(re.search(r"\.getId\(\)|identityHashCode|compareTo.*lock", source, re.I))
    p.has_trylock = "tryLock" in source
    p.has_readwrite_lock = bool(re.search(r"ReadWriteLock|StampedLock|readLock|writeLock", source))
    p.has_atomic = bool(re.search(r"Atomic(Integer|Long|Boolean|Reference)", source))

    # Lock depth
    depth = 0
    max_depth = 0
    for line in source.split("\n"):
        if "synchronized" in line or ".lock()" in line:
            depth += 1
            max_depth = max(max_depth, depth)
        if "}" in line and depth > 0:
            depth -= 1
    p.lock_depth = max_depth

    # Sync scope size
    sync_blocks = re.findall(r"synchronized[^{]*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", source, re.DOTALL)
    p.sync_scope_chars = sum(len(b) for b in sync_blocks)

    # I/O in lock
    in_sync = False
    for line in source.split("\n"):
        if "synchronized" in line or ".lock()" in line:
            in_sync = True
        if in_sync:
            if alloc_re.search(line):
                p.io_in_lock = True
            if re.search(r"\.(fetch|query|read|write|send|receive|execute|connect)\s*\(", line):
                p.io_in_lock = True
            if re.search(r"(for|while)\s*\(", line):
                p.loop_in_lock = True
            if re.search(r"Thread\.sleep|\.join\(\)", line):
                p.heavy_compute_in_lock = True
        if "}" in line:
            in_sync = False

    # Loops
    p.loop_count = len(re.findall(r"(for|while)\s*\(", source))
    # Nested loops: simple heuristic
    in_loop = 0
    for line in source.split("\n"):
        if re.search(r"(for|while)\s*\(", line):
            in_loop += 1
            if in_loop >= 2:
                p.nested_loop_count += 1
        if "}" in line and in_loop > 0:
            in_loop -= 1

    # Allocation in loop
    loop_blocks = re.findall(r"(for|while)\s*\([^)]*\)\s*\{([^}]*(?:\{[^}]*\}[^}]*)*)\}", source, re.DOTALL)
    for _, body in loop_blocks:
        if re.search(r"new\s+\w+\(", body):
            p.loop_has_allocation = True
        if alloc_re.search(body):
            p.loop_has_io = True
        if re.search(r"\+\s*\"|\"\s*\+", body):
            p.string_concat_in_loop = True

    # Memory patterns
    p.autoboxing = bool(re.search(r"new\s+(Integer|Long|Double|Float|Short|Byte)\s*\(", source))
    p.full_collection_load = bool(re.search(r"\.readAllBytes|\.readAllLines|toArray\(\)|collect\(Collectors\.toList", source))

    # Energy/polling
    p.polling_pattern = bool(re.search(r"while.*\{[\s\S]*?poll", source, re.DOTALL))
    p.busy_wait = bool(re.search(r"while.*\{[\s\S]*?Thread\.yield|busy.*wait|spin", source, re.DOTALL))
    p.thread_sleep_in_loop = bool(re.search(r"(while|for)[\s\S]*?Thread\.sleep", source, re.DOTALL))

    # Threading
    p.raw_thread_creation = bool(re.search(r"new\s+Thread\(", source))
    p.unbounded_pool = bool(re.search(r"newCachedThreadPool", source))

    return p


# ── Data-Flow Based Multi-Fault Detection ──────────────────────────────────────

def detect_dataflow_multi_faults(before: DataFlowProfile, after: DataFlowProfile) -> list[dict]:
    """Detect multi-fault fixes using data-flow profile comparison."""
    findings = []

    # ML + LC: Lock contention reduced AND collection lifecycle fixed
    if (before.lock_acquires > after.lock_acquires and
        before.collection_adds > 0 and before.collection_removes == 0 and
        (after.collection_removes > 0 or after.has_eviction or after.has_bounding)):
        findings.append({
            "id": "DF:ML+LC",
            "parents": ["memory", "synchronization"],
            "categories": ["memory.memory_leak", "synchronization.inefficient_synchronization"],
            "evidence": f"Locks: {before.lock_acquires}→{after.lock_acquires}, "
                        f"Adds: {before.collection_adds}, Removes: 0→{after.collection_removes}, "
                        f"Eviction: {after.has_eviction}",
        })

    # ML + RL: Accumulated unclosed resources → proper lifecycle
    if (before.unclosed_resources > 0 and before.collection_adds > 0 and
        (after.unclosed_resources == 0 or after.has_try_with_resources)):
        findings.append({
            "id": "DF:ML+RL",
            "parents": ["memory", "resource"],
            "categories": ["memory.memory_leak", "resource.resource_leak"],
            "evidence": f"Unclosed: {before.unclosed_resources}→{after.unclosed_resources}, "
                        f"Collection adds: {before.collection_adds}",
        })

    # LC + RL: I/O inside lock without cleanup → fixed
    if (before.io_in_lock and not before.has_finally and not before.has_try_with_resources and
        (not after.io_in_lock or after.has_try_with_resources or after.has_finally)):
        findings.append({
            "id": "DF:LC+RL",
            "parents": ["synchronization", "resource"],
            "categories": ["synchronization.inefficient_synchronization", "resource.resource_leak"],
            "evidence": "I/O in lock without cleanup → fixed",
        })

    # DL + LC: Nested locks flattened
    if (before.lock_depth >= 2 and
        (after.lock_depth < before.lock_depth or after.has_lock_ordering)):
        findings.append({
            "id": "DF:DL+LC",
            "parents": ["synchronization"],
            "categories": ["synchronization.improper_locking", "synchronization.inefficient_synchronization"],
            "evidence": f"Lock depth: {before.lock_depth}→{after.lock_depth}, "
                        f"Ordering: {after.has_lock_ordering}",
        })

    # LOOP + LC: Loop removed from lock scope
    if before.loop_in_lock and not after.loop_in_lock:
        findings.append({
            "id": "DF:LOOP+LC",
            "parents": ["loops", "synchronization"],
            "categories": ["loops.inefficient_loops", "synchronization.nested_loops_in_critical_sections"],
            "evidence": "Loop in lock → removed from lock scope",
        })

    # BLOAT + ENERGY: Allocation in loop fixed
    if (before.loop_has_allocation and before.string_concat_in_loop and
        not after.string_concat_in_loop):
        findings.append({
            "id": "DF:BLOAT+ENERGY",
            "parents": ["memory", "energy"],
            "categories": ["memory.memory_bloat", "energy.energy_leaks"],
            "evidence": "String concat in loop → StringBuilder or equivalent",
        })

    # MEMCONS + RL: Full collection load from unclosed stream
    if (before.full_collection_load and before.unclosed_resources > 0 and
        (not after.full_collection_load or after.has_try_with_resources)):
        findings.append({
            "id": "DF:MEMCONS+RL",
            "parents": ["memory", "resource"],
            "categories": ["memory.excessive_memory_consumption", "resource.resource_leak"],
            "evidence": "Full load from unclosed stream → streaming with proper close",
        })

    # Triple: LOOP + BLOAT + ENERGY
    if (before.loop_has_allocation and before.loop_count > 0 and
        (before.autoboxing or before.string_concat_in_loop) and
        not (after.loop_has_allocation and (after.autoboxing or after.string_concat_in_loop))):
        findings.append({
            "id": "DF:LOOP+BLOAT+ENERGY",
            "parents": ["loops", "memory", "energy"],
            "categories": ["loops.inefficient_loops", "memory.memory_bloat", "energy.energy_leaks"],
            "evidence": "Object creation/autoboxing in loop → optimized",
        })

    # Triple: DL + LC + RL
    if (before.lock_depth >= 2 and before.unclosed_resources > 0 and
        (after.lock_depth < before.lock_depth or after.has_lock_ordering) and
        (after.unclosed_resources == 0 or after.has_try_with_resources)):
        findings.append({
            "id": "DF:DL+LC+RL",
            "parents": ["synchronization", "resource"],
            "categories": ["synchronization.improper_locking", "synchronization.inefficient_synchronization",
                           "resource.resource_leak"],
            "evidence": f"Nested locks ({before.lock_depth}) with unclosed resources → fixed",
        })

    return findings


# ── Multi-Fault Result ─────────────────────────────────────────────────────────

@dataclass
class MultiFaultResult:
    pair_id: str
    repo: str
    commit_hash: str
    method_name: str
    class_name: str
    # From step 02
    structural_changes: list = field(default_factory=list)
    step02_multi_signals: list = field(default_factory=list)
    # From data-flow analysis
    dataflow_findings: list = field(default_factory=list)
    # Combined
    is_multi_fault: bool = False
    fault_count: int = 1
    all_categories: list = field(default_factory=list)
    all_parents: list = field(default_factory=list)
    dependency_label: str = ""          # e.g. "ML+LC" or "ML+RL" (paper's 3-class labels)
    detection_layers: int = 0           # how many layers detected it
    confidence: str = "low"
    # Validation guidance
    validation_needed: list = field(default_factory=list)
    # ── PR / commit traceability (carried through from step 02) ───────────────
    pr_number: str = ""
    pr_url: str = ""
    commit_url: str = ""
    pr_details: str = ""     # synthesised column: "PR #N · <title> · <pr_url>"


def analyze_pair(pair: dict) -> MultiFaultResult:
    """Full multi-fault analysis on a method pair."""
    result = MultiFaultResult(
        pair_id=pair["pair_id"],
        repo=pair["repo"],
        commit_hash=pair["commit_hash"],
        method_name=pair["method_name"],
        class_name=pair.get("class_name", ""),
        structural_changes=pair.get("structural_changes", []),
        step02_multi_signals=pair.get("multi_fault_signals", []),
        # ── carry PR traceability forward ──────────────────────────────────
        pr_number=pair.get("pr_number", ""),
        pr_url=pair.get("pr_url", ""),
        commit_url=pair.get("commit_url", ""),
        pr_details=pair.get("pr_details", ""),
    )

    before_src = pair.get("before_source", "")
    after_src = pair.get("after_source", "")

    # Data-flow analysis
    before_profile = analyze_data_flow(before_src)
    after_profile = analyze_data_flow(after_src)
    result.dataflow_findings = detect_dataflow_multi_faults(before_profile, after_profile)

    # Collect all categories from all sources
    all_cats = set()
    all_parents = set()

    # From step 02 detected categories
    for cat in pair.get("detected_fault_categories", []):
        all_cats.add(cat)
        all_parents.add(cat.split(".")[0])

    # From step 02 multi-fault signals
    # (These are IDs like "ML+LC:synchronized_hashmap_to_concurrent", parse parents from them)
    for sig_id in pair.get("multi_fault_signals", []):
        parts = sig_id.split(":")[0].split("+")
        abbrev_map = {
            "ML": ("memory", "memory.memory_leak"),
            "LC": ("synchronization", "synchronization.inefficient_synchronization"),
            "RL": ("resource", "resource.resource_leak"),
            "DL": ("synchronization", "synchronization.improper_locking"),
            "LOOP": ("loops", "loops.inefficient_loops"),
            "BLOAT": ("memory", "memory.memory_bloat"),
            "ENERGY": ("energy", "energy.energy_leaks"),
            "MEMCONS": ("memory", "memory.excessive_memory_consumption"),
            "REXC": ("resource", "resource.resource_excessive_consumption"),
        }
        for abbr in parts:
            if abbr in abbrev_map:
                all_parents.add(abbrev_map[abbr][0])
                all_cats.add(abbrev_map[abbr][1])

    # From data-flow findings
    for finding in result.dataflow_findings:
        for cat in finding["categories"]:
            all_cats.add(cat)
        for parent in finding["parents"]:
            all_parents.add(parent)

    result.all_categories = sorted(all_cats)
    result.all_parents = sorted(all_parents)
    result.fault_count = len(all_cats)
    result.is_multi_fault = len(all_parents) >= 2 or len(all_cats) >= 2

    if result.is_multi_fault:
        # Build dependency label using the paper's three fault-class abbreviations:
        #   LC = Lock Contention  (synchronization parent)
        #   ML = Memory Leak      (memory parent)
        #   RL = Resource Leak    (resource parent)
        # Other taxonomy parents (loops, energy, redundant_computation) are kept
        # as-is with upper-cased abbreviations so they don't pollute the paper
        # categories but are still visible for exploratory analysis.
        paper_abbrev = {
            "synchronization": "LC",
            "memory": "ML",
            "resource": "RL",
            # broader categories — kept but clearly non-paper
            "loops": "LOOP",
            "redundant_computation": "COMP",
            "energy": "ENERGY",
        }
        result.dependency_label = "+".join(
            paper_abbrev.get(p, p.upper()) for p in sorted(all_parents)
        )

        # Count detection layers
        layers = 0
        if result.step02_multi_signals:
            layers += 1
        if result.dataflow_findings:
            layers += 1
        if len(pair.get("detected_fault_categories", [])) >= 2:
            layers += 1
        result.detection_layers = layers

        result.confidence = "high" if layers >= 3 else ("medium" if layers >= 2 else "low")

        # Validation guidance
        if any("memory" in c for c in all_cats):
            result.validation_needed.append("JMX heap monitoring: before-fix monotonic growth, after-fix stable")
        if any("synchronization" in c or "lock" in c.lower() for c in all_cats):
            result.validation_needed.append("JMH with 4-64 threads, JLM lock hold time, Wilcoxon p<0.05")
        if any("resource" in c for c in all_cats):
            result.validation_needed.append("SpotBugs + file descriptor monitoring under stress")
        if any("deadlock" in c or "improper_locking" in c for c in all_cats):
            result.validation_needed.append("ThreadMXBean deadlock detection, 1000 random interleavings")
        if any("loop" in c for c in all_cats):
            result.validation_needed.append("JMH benchmark: measure iteration count and throughput")
        if any("energy" in c for c in all_cats):
            result.validation_needed.append("CPU profiling: measure idle cycles / polling frequency")

    return result


# ── Dependency Matrix Builder ──────────────────────────────────────────────────

def build_dependency_matrix(results: list[MultiFaultResult]) -> dict:
    """Build a co-occurrence matrix showing which fault categories appear together."""
    multi = [r for r in results if r.is_multi_fault]

    # Parent-level co-occurrence
    parent_cooccurrence = defaultdict(int)
    for r in multi:
        for p1, p2 in combinations(sorted(r.all_parents), 2):
            parent_cooccurrence[f"{p1}+{p2}"] += 1

    # Category-level co-occurrence
    cat_cooccurrence = defaultdict(int)
    for r in multi:
        for c1, c2 in combinations(sorted(r.all_categories), 2):
            cat_cooccurrence[f"{c1} ↔ {c2}"] += 1

    # Dependency chains (transitive)
    chains = []
    for r in multi:
        if len(r.all_parents) >= 3:
            chains.append({
                "parents": r.all_parents,
                "categories": r.all_categories,
                "example_pair": r.pair_id,
                "label": r.dependency_label,
            })

    return {
        "parent_cooccurrence": dict(sorted(parent_cooccurrence.items(), key=lambda x: -x[1])),
        "category_cooccurrence": dict(sorted(cat_cooccurrence.items(), key=lambda x: -x[1])[:30]),
        "triple_fault_chains": chains,
    }


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(results: list[MultiFaultResult], config: dict, output_path: str):
    multi = [r for r in results if r.is_multi_fault]

    # Group by dependency label
    by_label = defaultdict(list)
    for r in multi:
        by_label[r.dependency_label].append(r)

    # By repo
    by_repo = defaultdict(lambda: defaultdict(int))
    for r in multi:
        by_repo[r.repo][r.dependency_label] += 1

    # Confidence distribution
    conf_counts = defaultdict(int)
    for r in multi:
        conf_counts[r.confidence] += 1

    # Dependency matrix
    dep_matrix = build_dependency_matrix(results)

    # Compare with expected counts from config
    known_deps = config.get("fault_dependencies", {}).get("known_dependencies", [])
    expected_comparison = []
    for dep in known_deps:
        alias = dep["alias"]
        found = sum(1 for r in multi if alias.replace("+", "") in r.dependency_label.replace("+", "")
                    or set(dep["categories"]) & set(r.all_categories))
        expected_comparison.append({
            "alias": alias,
            "expected_range": dep.get("expected_count_range", [0, 999]),
            "found": found,
            "in_range": dep.get("expected_count_range", [0, 999])[0] <= found <= dep.get("expected_count_range", [0, 999])[1],
        })

    report = {
        "summary": {
            "total_analyzed": len(results),
            "multi_fault_instances": len(multi),
            "dual_fault": sum(1 for r in multi if len(r.all_parents) == 2),
            "triple_fault": sum(1 for r in multi if len(r.all_parents) >= 3),
            "multi_fault_rate": f"{len(multi)/max(len(results),1)*100:.1f}%",
        },
        "by_dependency_label": {k: len(v) for k, v in sorted(by_label.items(), key=lambda x: -len(x[1]))},
        "by_confidence": dict(conf_counts),
        "by_repo": {repo: dict(labels) for repo, labels in by_repo.items()},
        "dependency_matrix": dep_matrix,
        "expected_comparison": expected_comparison,
        "multi_fault_details": [asdict(r) for r in multi],
        "all_results_summary": [
            {
                "pair_id": r.pair_id,
                "is_multi_fault": r.is_multi_fault,
                "fault_count": r.fault_count,
                "dependency_label": r.dependency_label,
                "confidence": r.confidence,
                "all_parents": r.all_parents,
            }
            for r in results
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── CSV export: before_source | after_source | pr_details for multi-fault pairs ──
    # Provides the three paper-required columns for all confirmed dual/triple-fix instances.
    csv_path = output_path.replace(".json", "_multifault.csv")
    csv_fields = [
        "pair_id", "repo", "commit_hash", "method_name", "class_name",
        "dependency_label",      # e.g. "ML+LC", "ML+RL", "LC+RL"
        "fault_count",
        "all_parents",
        "confidence",
        "detection_layers",
        "before_source",         # ← Column 1
        "after_source",          # ← Column 2
        "pr_details",            # ← Column 3 (pr_number | pr_url | commit_url)
        "pr_number",
        "pr_url",
        "commit_url",
        "validation_needed",
    ]
    # Retrieve full before/after source from the original pairs for multi-fault instances
    pairs_by_id = {p["pair_id"]: p for p in _pairs_cache}
    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in multi:
            row = asdict(r)
            # Hydrate before/after source from the pairs cache
            src = pairs_by_id.get(r.pair_id, {})
            row["before_source"] = src.get("before_source", "")
            row["after_source"] = src.get("after_source", "")
            row["all_parents"] = "|".join(r.all_parents)
            row["validation_needed"] = " | ".join(r.validation_needed)
            writer.writerow(row)

    # Console summary
    logger.info("=" * 70)
    logger.info("MULTI-FAULT DEPENDENCY ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Total analyzed:  {len(results)}")
    logger.info(f"Multi-fault:     {len(multi)} ({len(multi)/max(len(results),1)*100:.1f}%)")
    logger.info(f"  Dual-fault:    {sum(1 for r in multi if len(r.all_parents) == 2)}")
    logger.info(f"  Triple-fault:  {sum(1 for r in multi if len(r.all_parents) >= 3)}")
    logger.info(f"\nBy dependency pattern:")
    for label, count in sorted(by_label.items(), key=lambda x: -len(x[1])):
        logger.info(f"  {label}: {len(count)}")
    logger.info(f"\nParent co-occurrence matrix:")
    for pair, count in list(dep_matrix["parent_cooccurrence"].items())[:10]:
        logger.info(f"  {pair}: {count}")
    if dep_matrix["triple_fault_chains"]:
        logger.info(f"\nTriple-fault chains: {len(dep_matrix['triple_fault_chains'])}")
        for chain in dep_matrix["triple_fault_chains"][:5]:
            logger.info(f"  {chain['label']}: {chain['categories']}")
    logger.info(f"\nExpected vs found:")
    for ec in expected_comparison:
        status = "✓" if ec["in_range"] else "✗"
        logger.info(f"  {status} {ec['alias']}: found {ec['found']} "
                     f"(expected {ec['expected_range'][0]}-{ec['expected_range'][1]})")
    logger.info(f"\nOutput: {output_path}")
    logger.info(f"CSV   : {csv_path}  ← before_source | after_source | pr_details")


# Module-level cache so generate_report can access raw pair source for CSV export
_pairs_cache: list = []


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="FaultEmbed: Multi-fault dependency detection")
    ap.add_argument("--method-pairs", required=True)
    ap.add_argument("--config", required=True, help="Path to fault_taxonomy.json")
    ap.add_argument("--output", default="output/multi_fault_analysis.json")
    args = ap.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    with open(args.method_pairs) as f:
        data = json.load(f)
    pairs = data["pairs"]
    logger.info(f"Loaded {len(pairs)} method pairs")

    # Populate the cache so generate_report can hydrate before/after source in CSV
    global _pairs_cache
    _pairs_cache = pairs

    results = []
    for i, pair in enumerate(pairs):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Analyzed {i}/{len(pairs)}...")
        results.append(analyze_pair(pair))

    generate_report(results, config, args.output)


if __name__ == "__main__":
    main()
