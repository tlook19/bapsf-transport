#!/usr/bin/env python3
"""Validate restructure rename manifests against KB_MANIFEST_SCHEMA (2026-08-26).

Six checks, in the order the schema doc numbers them:

  (1) SCHEMA      -- JSON shape: required top-level keys, enum values,
                     per-row required keys, locator shape.
  (2) LOCATORS    -- every ``old`` locator resolves at ``base_revision`` and
                     every ``new`` locator at ``new_revision``, via
                     ``git cat-file`` plus an AST walk for symbol anchors.
  (3) CHAIN       -- each delta's ``base_revision`` is its predecessor's
                     ``new_revision``, and a cumulative manifest equals the
                     composition of the deltas it is checked against.
  (4) COVERAGE    -- every tracked file that git reports as moved or deleted
                     between base and new has a covering row.  Rename
                     DETECTION is used for coverage only and is never read as
                     continuity evidence: the check consumes
                     ``--diff-filter`` statuses over ``--no-renames`` output,
                     so a move is seen as one delete plus one add and both
                     ends must be covered.
  (5) GROUPS      -- split/merge group integrity.
  (6) DELETES     -- delete rows carry a reason and a resolved replacement
                     disposition.

Standalone: standard library plus the ``git`` executable.  No third-party
imports, no package imports, runnable from anywhere.

    python validate_manifest.py <manifest.json> [<manifest.json> ...]
    python validate_manifest.py --self-test

DRAFT manifests (``new_revision`` == ``"TBD-at-commit"``, filename carrying
``.DRAFT.``) are accepted: the new end of checks (2) and the whole of checks
(3) and (4) are SKIPPED and reported as skipped, because the revision the
``new`` locators resolve against does not exist until the commit is written.
A draft never reports a clean full pass.

Two additive schema extensions this validator understands, both proposed with
the flatten manifest and both reviewer-gated (RENAME_MAP.md Q1):

  * ``prefix_rule`` -- an optional row object
    ``{"old_prefix": str, "new_prefix": str, "covers": [path, ...]}`` that
    makes the schema's "ONE module row + a stated prefix rule" form
    machine-readable, so check (4) can be mechanical rather than prose.
  * ``anchor_kind: "directory"`` -- accepted ONLY on a row that carries a
    ``prefix_rule``; the schema's anchor kinds cannot name a directory that
    is not an importable package, and ``cablp/scripts/`` is one.
"""

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

DRAFT_SENTINEL = "TBD-at-commit"

MANIFEST_KINDS = frozenset({"delta", "cumulative"})

CHANGE_KINDS = frozenset({
    "move", "rename", "move+rename", "split", "merge", "delete", "add",
    "signature_change", "surface_change",
})

ANCHOR_KINDS = frozenset({
    "module", "class", "function", "method", "config_key", "data_file",
    "doc_section",
})

#: Extension anchor kind; legal only on a row carrying a ``prefix_rule``.
DIRECTORY_ANCHOR = "directory"

CONTINUITIES = frozenset({
    "same_entity", "successor", "replacement", "retired_no_successor",
})

#: Anchor kinds whose resolution is an AST symbol lookup rather than mere
#: file existence.
SYMBOL_ANCHORS = frozenset({"class", "function", "method"})

TOP_LEVEL_KEYS = (
    "manifest_kind", "repository", "base_revision", "new_revision",
    "bit_exact", "golden_gate", "mappings",
)

ROW_KEYS = (
    "change_kind", "old", "new", "proposed_continuity", "group_id",
    "deletion_reason", "replacement", "notes",
)

#: change_kinds for which ``old`` must be null.
OLD_IS_NULL = frozenset({"add"})
#: change_kinds for which ``new`` must be null.
NEW_IS_NULL = frozenset({"delete"})


class Findings:
    """Accumulates per-check failures and skips; never raises mid-walk."""

    def __init__(self):
        self.failures = []
        self.skips = []
        self.counts = {}

    def fail(self, check, message):
        self.failures.append((check, message))

    def skip(self, check, message):
        self.skips.append((check, message))

    def note(self, check, count):
        self.counts[check] = count

    @property
    def ok(self):
        return not self.failures


# --------------------------------------------------------------------------
# git plumbing
# --------------------------------------------------------------------------

def git(repo, *arguments, check=True):
    """Run git in ``repo`` and return stdout; ``check=False`` tolerates exit!=0."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=str(repo), capture_output=True, text=True,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def blob_at(repo, revision, path):
    """Return the blob text at ``revision:path``, or None when absent."""
    completed = subprocess.run(
        ["git", "cat-file", "-p", f"{revision}:{path}"],
        cwd=str(repo), capture_output=True, text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout


def tree_paths(repo, revision):
    """Return the set of tracked paths at ``revision``."""
    out = git(repo, "ls-tree", "-r", "--name-only", revision)
    return set(out.split("\n")) - {""}


def revision_exists(repo, revision):
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
        cwd=str(repo), capture_output=True, text=True,
    )
    return completed.returncode == 0


# --------------------------------------------------------------------------
# check 1 -- schema conformance
# --------------------------------------------------------------------------

def _is_sha(value):
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(c in "0123456789abcdef" for c in value)
    )


def _check_locator(findings, where, locator, allow_directory):
    if not isinstance(locator, dict):
        findings.fail("SCHEMA", f"{where}: locator must be an object")
        return
    for key in ("path", "anchor_kind", "symbol"):
        if key not in locator:
            findings.fail("SCHEMA", f"{where}: locator missing {key!r}")
    kind = locator.get("anchor_kind")
    if kind == DIRECTORY_ANCHOR:
        if not allow_directory:
            findings.fail(
                "SCHEMA",
                f"{where}: anchor_kind 'directory' is only legal on a row "
                "carrying a prefix_rule",
            )
    elif kind not in ANCHOR_KINDS:
        findings.fail("SCHEMA", f"{where}: bad anchor_kind {kind!r}")
    path = locator.get("path")
    if not isinstance(path, str) or not path or path.startswith("/"):
        findings.fail("SCHEMA", f"{where}: path must be a relative string")
    if "line_hint" in locator and not isinstance(locator["line_hint"], int):
        findings.fail("SCHEMA", f"{where}: line_hint must be an integer hint")


def _check_prefix_rule(findings, where, rule):
    if not isinstance(rule, dict):
        findings.fail("SCHEMA", f"{where}: prefix_rule must be an object")
        return
    for key in ("old_prefix", "new_prefix", "covers"):
        if key not in rule:
            findings.fail("SCHEMA", f"{where}: prefix_rule missing {key!r}")
            return
    if not isinstance(rule["covers"], list) or not rule["covers"]:
        findings.fail("SCHEMA", f"{where}: prefix_rule.covers must be non-empty")
        return
    for covered in rule["covers"]:
        if not isinstance(covered, str) or not covered.startswith(
            rule["old_prefix"]
        ):
            findings.fail(
                "SCHEMA",
                f"{where}: prefix_rule.covers entry {covered!r} is not under "
                f"old_prefix {rule['old_prefix']!r}",
            )
            return


def check_schema(findings, manifest, source):
    for key in TOP_LEVEL_KEYS:
        if key not in manifest:
            findings.fail("SCHEMA", f"{source}: missing top-level key {key!r}")
    if manifest.get("manifest_kind") not in MANIFEST_KINDS:
        findings.fail(
            "SCHEMA",
            f"{source}: manifest_kind {manifest.get('manifest_kind')!r} "
            f"not in {sorted(MANIFEST_KINDS)}",
        )
    if not _is_sha(manifest.get("base_revision")):
        findings.fail(
            "SCHEMA", f"{source}: base_revision must be a 40-char sha"
        )
    new_revision = manifest.get("new_revision")
    if not (_is_sha(new_revision) or new_revision == DRAFT_SENTINEL):
        findings.fail(
            "SCHEMA",
            f"{source}: new_revision must be a 40-char sha or "
            f"{DRAFT_SENTINEL!r}",
        )
    if manifest.get("bit_exact") is not True:
        findings.fail(
            "SCHEMA",
            f"{source}: bit_exact must be true for an R2 manifest "
            "(R3's semantic manifest is a different document)",
        )
    gate = manifest.get("golden_gate")
    if not isinstance(gate, dict) or "script" not in gate:
        findings.fail("SCHEMA", f"{source}: golden_gate must name a script")
    rows = manifest.get("mappings")
    if not isinstance(rows, list):
        findings.fail("SCHEMA", f"{source}: mappings must be a list")
        return
    for index, r in enumerate(rows):
        where = f"{source}[{index}]"
        if not isinstance(r, dict):
            findings.fail("SCHEMA", f"{where}: row must be an object")
            continue
        for key in ROW_KEYS:
            if key not in r:
                findings.fail("SCHEMA", f"{where}: missing row key {key!r}")
        kind = r.get("change_kind")
        if kind not in CHANGE_KINDS:
            findings.fail("SCHEMA", f"{where}: bad change_kind {kind!r}")
        if r.get("proposed_continuity") not in CONTINUITIES:
            findings.fail(
                "SCHEMA",
                f"{where}: bad proposed_continuity "
                f"{r.get('proposed_continuity')!r}",
            )
        has_rule = "prefix_rule" in r and r["prefix_rule"] is not None
        if has_rule:
            _check_prefix_rule(findings, where, r["prefix_rule"])
        if kind in OLD_IS_NULL:
            if r.get("old") is not None:
                findings.fail("SCHEMA", f"{where}: 'add' row must have old=null")
        else:
            _check_locator(findings, where + ".old", r.get("old"), has_rule)
        if kind in NEW_IS_NULL:
            if r.get("new") is not None:
                findings.fail(
                    "SCHEMA", f"{where}: 'delete' row must have new=null"
                )
        else:
            _check_locator(findings, where + ".new", r.get("new"), has_rule)
    findings.note("SCHEMA", len(rows))


# --------------------------------------------------------------------------
# check 2 -- locator resolution
# --------------------------------------------------------------------------

def _qualified_names(source_text):
    """Return every dotted class/function/method name defined in a module."""
    names = set()

    def walk(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                qualified = f"{prefix}{child.name}"
                names.add(qualified)
                walk(child, qualified + ".")

    walk(ast.parse(source_text), "")
    return names


def _resolve(findings, repo, revision, locator, where, tracked):
    kind = locator["anchor_kind"]
    path = locator["path"]
    if kind == DIRECTORY_ANCHOR:
        prefix = path.rstrip("/") + "/"
        if not any(p.startswith(prefix) for p in tracked):
            findings.fail(
                "LOCATORS",
                f"{where}: directory {path!r} holds no tracked file at "
                f"{revision[:12]}",
            )
        return
    text = blob_at(repo, revision, path)
    if text is None:
        findings.fail(
            "LOCATORS", f"{where}: {path!r} does not exist at {revision[:12]}"
        )
        return
    symbol = locator["symbol"]
    if kind in SYMBOL_ANCHORS:
        if not path.endswith(".py"):
            findings.fail(
                "LOCATORS",
                f"{where}: symbol anchor on non-Python path {path!r}",
            )
            return
        try:
            defined = _qualified_names(text)
        except SyntaxError as error:
            findings.fail(
                "LOCATORS", f"{where}: {path!r} does not parse ({error})"
            )
            return
        # The manifest carries the IMPORT-QUALIFIED symbol; the AST knows only
        # the module-local part, so match on the longest dotted suffix.
        local = symbol.split(".")
        if not any(
            ".".join(local[i:]) in defined for i in range(len(local))
        ):
            findings.fail(
                "LOCATORS",
                f"{where}: symbol {symbol!r} not defined in {path!r} at "
                f"{revision[:12]}",
            )
    elif kind == "config_key":
        if symbol.rsplit(".", 1)[-1] not in text:
            findings.fail(
                "LOCATORS",
                f"{where}: config key {symbol!r} does not appear in {path!r}",
            )
    elif kind == "doc_section":
        if symbol not in text:
            findings.fail(
                "LOCATORS",
                f"{where}: doc section {symbol!r} does not appear in {path!r}",
            )
    # `module` and `data_file` resolve on existence alone.


def check_locators(findings, repo, manifest, source, draft):
    base = manifest["base_revision"]
    new = manifest["new_revision"]
    if not revision_exists(repo, base):
        findings.fail(
            "LOCATORS", f"{source}: base_revision {base[:12]} not in {repo}"
        )
        return
    base_tracked = tree_paths(repo, base)
    new_tracked = None
    if draft:
        findings.skip(
            "LOCATORS",
            f"{source}: new-end resolution skipped (new_revision is "
            f"{DRAFT_SENTINEL})",
        )
    elif not revision_exists(repo, new):
        findings.fail(
            "LOCATORS", f"{source}: new_revision {new[:12]} not in {repo}"
        )
        return
    else:
        new_tracked = tree_paths(repo, new)
    resolved = 0
    for index, r in enumerate(manifest["mappings"]):
        where = f"{source}[{index}]"
        if r.get("old") is not None:
            _resolve(findings, repo, base, r["old"], where + ".old",
                     base_tracked)
            resolved += 1
        if r.get("new") is not None and new_tracked is not None:
            _resolve(findings, repo, new, r["new"], where + ".new",
                     new_tracked)
            resolved += 1
        replacement = r.get("replacement")
        if replacement is not None and new_tracked is not None:
            _resolve(findings, repo, new, replacement,
                     where + ".replacement", new_tracked)
            resolved += 1
    findings.note("LOCATORS", resolved)


# --------------------------------------------------------------------------
# check 3 -- delta-chain composition
# --------------------------------------------------------------------------

def _row_path_pairs(manifest):
    """Return {old_path: frozenset(new_paths)} at file granularity.

    Prefix rules are expanded to their covered files.  A split contributes
    several new paths under one old path; a delete contributes ``None``.  A
    directory-anchored row with no prefix rule contributes nothing -- it
    names no file.
    """
    pairs = {}
    for r in manifest["mappings"]:
        rule = r.get("prefix_rule")
        if rule:
            for covered in rule["covers"]:
                tail = covered[len(rule["old_prefix"]):]
                pairs.setdefault(covered, set()).add(rule["new_prefix"] + tail)
            continue
        old = r.get("old")
        new = r.get("new")
        if old is None or r["change_kind"] == "add":
            continue
        if old["anchor_kind"] == DIRECTORY_ANCHOR:
            continue
        pairs.setdefault(old["path"], set()).add(
            None if new is None else new["path"]
        )
    return {old: frozenset(news) for old, news in pairs.items()}


def _covered_paths(manifest):
    """Return (old_paths, new_paths) that the manifest's rows account for.

    Both ends are collected independently: a row whose ``new`` is an ``add``
    covers only the new end, a ``delete`` only the old end, and a split
    covers one old path and every one of its legs' new paths.
    """
    old_paths, new_paths = set(), set()
    for r in manifest["mappings"]:
        rule = r.get("prefix_rule")
        if rule:
            for covered in rule["covers"]:
                tail = covered[len(rule["old_prefix"]):]
                old_paths.add(covered)
                new_paths.add(rule["new_prefix"] + tail)
            continue
        old = r.get("old")
        new = r.get("new")
        if old is not None and old["anchor_kind"] != DIRECTORY_ANCHOR:
            old_paths.add(old["path"])
        if new is not None and new["anchor_kind"] != DIRECTORY_ANCHOR:
            new_paths.add(new["path"])
    return old_paths, new_paths


def check_chain(findings, deltas, cumulative):
    """Deltas must abut; a cumulative must equal their composition."""
    ordered = list(deltas)
    for earlier, later in zip(ordered, ordered[1:]):
        if later[1]["base_revision"] != earlier[1]["new_revision"]:
            findings.fail(
                "CHAIN",
                f"{later[0]}: base_revision {later[1]['base_revision'][:12]} "
                f"does not continue {earlier[0]} whose new_revision is "
                f"{earlier[1]['new_revision'][:12]}",
            )
    if cumulative is None:
        findings.skip("CHAIN", "no cumulative manifest supplied to compose against")
        findings.note("CHAIN", max(len(ordered) - 1, 0))
        return
    composed = {}
    for _, manifest in ordered:
        step = _row_path_pairs(manifest)
        if not composed:
            composed = dict(step)
            continue
        consumed = set()
        for old, tips in list(composed.items()):
            advanced = set()
            for tip in tips:
                if tip in step:
                    consumed.add(tip)
                    advanced |= set(step[tip])
                else:
                    advanced.add(tip)
            composed[old] = frozenset(advanced)
        for old, news in step.items():
            if old not in consumed and old not in composed:
                composed[old] = news
    name, manifest = cumulative
    declared = _row_path_pairs(manifest)
    if declared != composed:
        missing = sorted(set(composed) - set(declared))
        extra = sorted(set(declared) - set(composed))
        wrong = sorted(
            p for p in set(declared) & set(composed)
            if declared[p] != composed[p]
        )
        findings.fail(
            "CHAIN",
            f"{name}: cumulative does not equal the delta composition "
            f"(missing={missing[:5]} extra={extra[:5]} mismatched={wrong[:5]})",
        )
    findings.note("CHAIN", len(composed))


# --------------------------------------------------------------------------
# check 4 -- git-diff coverage cross-check
# --------------------------------------------------------------------------

def check_coverage(findings, repo, manifest, source):
    """Every path git says left or arrived must be covered by some row.

    ``--no-renames`` is deliberate.  Rename detection is a heuristic and the
    adopted contract forbids reading it as continuity evidence, so the check
    asks git only the question git can answer without guessing: which paths
    exist at one end and not the other.
    """
    base = manifest["base_revision"]
    new = manifest["new_revision"]
    raw = git(
        repo, "diff", "--name-status", "--no-renames", base, new,
    )
    departed, arrived = set(), set()
    for line in raw.split("\n"):
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        path = path.strip()
        if status.startswith("D"):
            departed.add(path)
        elif status.startswith("A"):
            arrived.add(path)
    covered_old, covered_new = _covered_paths(manifest)
    uncovered_departed = sorted(departed - covered_old)
    uncovered_arrived = sorted(arrived - covered_new)
    if uncovered_departed:
        findings.fail(
            "COVERAGE",
            f"{source}: {len(uncovered_departed)} path(s) removed between "
            f"base and new with no covering row: {uncovered_departed[:8]}",
        )
    if uncovered_arrived:
        findings.fail(
            "COVERAGE",
            f"{source}: {len(uncovered_arrived)} path(s) added between base "
            f"and new with no covering row: {uncovered_arrived[:8]}",
        )
    findings.note("COVERAGE", len(departed) + len(arrived))


# --------------------------------------------------------------------------
# check 5 -- split/merge group integrity
# --------------------------------------------------------------------------

def check_groups(findings, manifest, source):
    groups = {}
    for index, r in enumerate(manifest["mappings"]):
        group_id = r.get("group_id")
        if group_id is None:
            if r.get("change_kind") in ("split", "merge"):
                findings.fail(
                    "GROUPS",
                    f"{source}[{index}]: {r['change_kind']} row carries no "
                    "group_id",
                )
            continue
        groups.setdefault(group_id, []).append((index, r))
    for group_id, members in sorted(groups.items()):
        kinds = {r["change_kind"] for _, r in members}
        if kinds not in ({"split"}, {"merge"}):
            findings.fail(
                "GROUPS",
                f"{source}: group {group_id!r} mixes change_kinds "
                f"{sorted(kinds)}; a group is all-split or all-merge",
            )
            continue
        kind = kinds.pop()
        if not group_id.startswith(kind + ":"):
            findings.fail(
                "GROUPS",
                f"{source}: group {group_id!r} must be prefixed {kind + ':'!r}",
            )
        if len(members) < 2:
            findings.fail(
                "GROUPS",
                f"{source}: group {group_id!r} has {len(members)} leg(s); a "
                f"{kind} needs at least 2",
            )
            continue
        shared = "old" if kind == "split" else "new"
        distinct = {json.dumps(r[shared], sort_keys=True) for _, r in members}
        if len(distinct) != 1:
            findings.fail(
                "GROUPS",
                f"{source}: {kind} group {group_id!r} legs must share an "
                f"identical {shared!r} locator ({len(distinct)} distinct)",
            )
    findings.note("GROUPS", len(groups))


# --------------------------------------------------------------------------
# check 6 -- delete-row completeness
# --------------------------------------------------------------------------

def check_deletes(findings, manifest, source):
    count = 0
    for index, r in enumerate(manifest["mappings"]):
        if r.get("change_kind") != "delete":
            continue
        count += 1
        where = f"{source}[{index}]"
        reason = r.get("deletion_reason")
        if not isinstance(reason, str) or not reason.strip():
            findings.fail(
                "DELETES", f"{where}: delete row needs a deletion_reason"
            )
        replacement = r.get("replacement")
        continuity = r.get("proposed_continuity")
        if replacement is None:
            if continuity != "retired_no_successor":
                findings.fail(
                    "DELETES",
                    f"{where}: replacement is null so proposed_continuity "
                    f"must be 'retired_no_successor', not {continuity!r}",
                )
        else:
            _check_locator(findings, where + ".replacement", replacement, False)
            if continuity not in ("successor", "replacement"):
                findings.fail(
                    "DELETES",
                    f"{where}: a replacement locator requires "
                    f"proposed_continuity 'successor' or 'replacement', not "
                    f"{continuity!r}",
                )
    findings.note("DELETES", count)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def is_draft(path, manifest):
    return (
        manifest.get("new_revision") == DRAFT_SENTINEL
        or ".DRAFT." in Path(path).name
    )


def validate(repo, manifest_paths, skip_git=False):
    findings = Findings()
    loaded = []
    for path in manifest_paths:
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            findings.fail("SCHEMA", f"{path}: unreadable ({error})")
            continue
        loaded.append((path, manifest))
        check_schema(findings, manifest, Path(path).name)

    usable = [
        (p, m) for p, m in loaded
        if isinstance(m.get("mappings"), list)
        and isinstance(m.get("base_revision"), str)
    ]
    for path, manifest in usable:
        name = Path(path).name
        draft = is_draft(path, manifest)
        check_groups(findings, manifest, name)
        check_deletes(findings, manifest, name)
        if skip_git:
            findings.skip("LOCATORS", f"{name}: --skip-git")
            findings.skip("COVERAGE", f"{name}: --skip-git")
            continue
        check_locators(findings, repo, manifest, name, draft)
        if draft:
            findings.skip(
                "COVERAGE",
                f"{name}: skipped (new_revision is {DRAFT_SENTINEL}; the "
                "diff has no second end yet)",
            )
        else:
            check_coverage(findings, repo, manifest, name)

    deltas = [
        (Path(p).name, m) for p, m in usable
        if m.get("manifest_kind") == "delta"
    ]
    cumulatives = [
        (Path(p).name, m) for p, m in usable
        if m.get("manifest_kind") == "cumulative"
    ]
    if any(m.get("new_revision") == DRAFT_SENTINEL for _, m in deltas):
        findings.skip(
            "CHAIN", "skipped: a supplied delta is a DRAFT with no new_revision"
        )
    elif len(deltas) < 2 and not cumulatives:
        findings.skip("CHAIN", "skipped: fewer than two manifests to compose")
    else:
        check_chain(
            findings, deltas, cumulatives[0] if cumulatives else None
        )
    return findings


def report(findings, manifest_paths):
    checks = ("SCHEMA", "LOCATORS", "CHAIN", "COVERAGE", "GROUPS", "DELETES")
    failed = {c for c, _ in findings.failures}
    skipped = {c for c, _ in findings.skips}
    print(f"validate_manifest: {len(manifest_paths)} manifest(s)")
    for path in manifest_paths:
        print(f"  input: {path}")
    print()
    for check in checks:
        if check in failed:
            status = "FAIL"
        elif check in skipped and check not in findings.counts:
            status = "SKIP"
        elif check in skipped:
            status = "PARTIAL"
        else:
            status = "PASS"
        count = findings.counts.get(check)
        detail = "" if count is None else f"  ({count} checked)"
        print(f"  [{status:7s}] {check}{detail}")
    if findings.skips:
        print("\nskipped:")
        for check, message in findings.skips:
            print(f"  {check}: {message}")
    if findings.failures:
        print("\nfailures:")
        for check, message in findings.failures:
            print(f"  {check}: {message}")
    print()
    if findings.failures:
        print("RESULT: FAIL")
        return 1
    if findings.skips:
        print("RESULT: PASS (with skips -- not a full validation)")
        return 0
    print("RESULT: PASS")
    return 0


# --------------------------------------------------------------------------
# self-test: one synthetic repository per failure class
# --------------------------------------------------------------------------

def _init_repo(root):
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "selftest@example.invalid")
    git(root, "config", "user.name", "selftest")


def _commit(root, files, message):
    for relative, text in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", message)
    return git(root, "rev-parse", "HEAD").strip()


def _base_manifest(base, new, rows):
    return {
        "manifest_kind": "delta",
        "repository": "selftest",
        "base_revision": base,
        "new_revision": new,
        "bit_exact": True,
        "golden_gate": {
            "script": "scripts/baseline_sim1d.py",
            "result": "exact=True",
            "saves": 2620,
        },
        "mappings": rows,
    }


def _row(kind, old, new, continuity, **extra):
    row = {
        "change_kind": kind,
        "old": old,
        "new": new,
        "proposed_continuity": continuity,
        "group_id": None,
        "deletion_reason": None,
        "replacement": None,
        "notes": "",
    }
    row.update(extra)
    return row


def _loc(path, kind, symbol, **extra):
    return {"path": path, "anchor_kind": kind, "symbol": symbol, **extra}


def _write(directory, name, payload):
    path = directory / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)


def _build_fixture_repo(repo):
    """Three commits: base -> mid (move/split/delete) -> tip (one more move).

    Two steps are needed so the CHAIN check has something real to compose.
    """
    _init_repo(repo)
    base = _commit(repo, {
        "old/mod.py": "def alpha(x):\n    return x\n",
        "old/data.csv": "1,2\n",
        "old/gone.py": "def dead():\n    pass\n",
        "old/whole.py": "def whole():\n    pass\n",
    }, "base")
    git(repo, "rm", "-q", "old/mod.py", "old/data.csv", "old/gone.py",
        "old/whole.py")
    mid = _commit(repo, {
        "new/mod.py": "def alpha(x):\n    return x\n",
        "new/data.csv": "1,2\n",
        "new/part_a.py": "def whole_a():\n    pass\n",
        "new/part_b.py": "def whole_b():\n    pass\n",
    }, "mid")
    git(repo, "rm", "-q", "new/mod.py")
    tip = _commit(repo, {
        "final/mod.py": "def alpha(x):\n    return x\n",
    }, "tip")
    return base, mid, tip


def self_test():
    """Exercise a positive control plus one deliberate failure of each check.

    Every failure fixture is built so that exactly ONE check fires: a case
    that tripped two checks would not tell us the check under test works.
    """
    import copy

    results = []

    def record(label, expect, findings):
        failed = {c for c, _ in findings.failures}
        if expect is None:
            ok, got = not failed, "clean" if not failed else f"failed {sorted(failed)}"
        else:
            ok = failed == {expect}
            got = f"failed {sorted(failed)}" if failed else "clean"
        results.append((label, expect, ok, got))

    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        repo = workspace / "repo"
        repo.mkdir()
        base, mid, tip = _build_fixture_repo(repo)
        manifests = workspace / "manifests"
        manifests.mkdir()

        # --- the base->mid delta, correct in every respect ----------------
        good_rows = [
            _row(
                "move+rename",
                _loc("old/mod.py", "function", "pkg.old.mod.alpha"),
                _loc("new/mod.py", "function", "pkg.new.mod.alpha"),
                "same_entity",
            ),
            _row(
                "move",
                _loc("old/data.csv", "data_file", "old/data.csv"),
                _loc("new/data.csv", "data_file", "new/data.csv"),
                "same_entity",
            ),
            _row(
                "delete",
                _loc("old/gone.py", "module", "pkg.old.gone"),
                None,
                "retired_no_successor",
                deletion_reason="frozen, ungenerable, no consumer",
            ),
            _row(
                "split",
                _loc("old/whole.py", "module", "pkg.old.whole"),
                _loc("new/part_a.py", "module", "pkg.new.part_a"),
                "successor",
                group_id="split:whole",
            ),
            _row(
                "split",
                _loc("old/whole.py", "module", "pkg.old.whole"),
                _loc("new/part_b.py", "module", "pkg.new.part_b"),
                "successor",
                group_id="split:whole",
            ),
        ]

        def one(label, rows, expect, mutate=None):
            manifest = _base_manifest(base, mid, copy.deepcopy(rows))
            if mutate:
                mutate(manifest)
            path = _write(manifests, f"{label}.json", manifest)
            record(label, expect, validate(repo, [path]))

        one("positive-control", good_rows, None)

        bad = copy.deepcopy(good_rows)
        bad[0]["change_kind"] = "teleport"
        one("fail-SCHEMA", bad, "SCHEMA")

        bad = copy.deepcopy(good_rows)
        bad[0]["new"]["symbol"] = "pkg.new.mod.does_not_exist"
        one("fail-LOCATORS", bad, "LOCATORS")

        bad = copy.deepcopy(good_rows)
        del bad[1]                     # drop the data.csv row entirely
        one("fail-COVERAGE", bad, "COVERAGE")

        bad = copy.deepcopy(good_rows)
        bad[4]["old"] = _loc("old/data.csv", "data_file", "old/data.csv")
        one("fail-GROUPS", bad, "GROUPS")   # split legs no longer share `old`

        bad = copy.deepcopy(good_rows)
        bad[2]["deletion_reason"] = None
        one("fail-DELETES", bad, "DELETES")

        # --- CHAIN: abutment, composition, and a clean two-step control ---
        delta1 = _base_manifest(base, mid, copy.deepcopy(good_rows))
        delta2 = _base_manifest(mid, tip, [
            _row(
                "move",
                _loc("new/mod.py", "function", "pkg.new.mod.alpha"),
                _loc("final/mod.py", "function", "pkg.final.mod.alpha"),
                "same_entity",
            ),
        ])
        cumulative_rows = [
            _row(
                "move+rename",
                _loc("old/mod.py", "function", "pkg.old.mod.alpha"),
                _loc("final/mod.py", "function", "pkg.final.mod.alpha"),
                "same_entity",
            ),
            _row(
                "move",
                _loc("old/data.csv", "data_file", "old/data.csv"),
                _loc("new/data.csv", "data_file", "new/data.csv"),
                "same_entity",
            ),
            copy.deepcopy(good_rows[2]),
            copy.deepcopy(good_rows[3]),
            copy.deepcopy(good_rows[4]),
        ]
        cumulative = _base_manifest(base, tip, cumulative_rows)
        cumulative["manifest_kind"] = "cumulative"

        paths = [
            _write(manifests, "chain-ok-a.json", delta1),
            _write(manifests, "chain-ok-b.json", delta2),
            _write(manifests, "chain-ok-cumulative.json", cumulative),
        ]
        record("chain-compose-ok", None, validate(repo, paths))

        # (a) abutment: a second delta that does not continue the first.
        #     Its range is empty (mid..mid) so ONLY the chain check can fire.
        stray = _base_manifest(mid, mid, [])
        stray["base_revision"] = base           # should have been `mid`
        stray["new_revision"] = base
        record("fail-CHAIN-abutment", "CHAIN", validate(repo, [
            _write(manifests, "chain-bad-a.json", delta1),
            _write(manifests, "chain-bad-b.json", stray),
        ]))

        # (b) composition: a cumulative whose ends are individually covered
        #     and individually resolvable, but wired to the wrong old paths.
        crossed = copy.deepcopy(cumulative)
        crossed["mappings"][0]["old"] = _loc(
            "old/data.csv", "data_file", "old/data.csv"
        )
        crossed["mappings"][1]["old"] = _loc(
            "old/mod.py", "function", "pkg.old.mod.alpha"
        )
        crossed["mappings"][1]["new"]["anchor_kind"] = "data_file"
        record("fail-CHAIN-composition", "CHAIN", validate(repo, [
            _write(manifests, "chain-x-a.json", delta1),
            _write(manifests, "chain-x-b.json", delta2),
            _write(manifests, "chain-x-cumulative.json", crossed),
        ]))

        # --- DRAFT mode, the shape delta_flatten.DRAFT.json is in ---------
        draft = _base_manifest(base, DRAFT_SENTINEL, copy.deepcopy(good_rows))
        draft["golden_gate"] = {
            "script": "scripts/baseline_sim1d.py",
            "result": DRAFT_SENTINEL,
            "saves": None,
        }
        findings = validate(
            repo, [_write(manifests, "draft.DRAFT.json", draft)]
        )
        failed = {c for c, _ in findings.failures}
        skipped = {c for c, _ in findings.skips}
        results.append((
            "draft-mode",
            "clean, with COVERAGE+CHAIN skipped",
            not failed and {"COVERAGE", "CHAIN"} <= skipped,
            f"failed={sorted(failed)} skipped={sorted(skipped)}",
        ))

    print("validate_manifest --self-test")
    print(f"  {len(results)} cases over a synthetic three-commit repository:")
    print("  2 positive controls, 7 failure fixtures, 1 draft-mode case.\n")
    width = max(len(label) for label, *_ in results)
    for label, expect, ok, got in results:
        expected = "clean pass" if expect is None else (
            expect if " " in str(expect) else f"FAIL in {expect}"
        )
        print(
            f"  [{'ok ' if ok else 'BAD'}] {label:<{width}}  "
            f"expected {expected:<34} got {got}"
        )
    passed = sum(1 for _, _, ok, _ in results if ok)
    print(f"\n  {passed}/{len(results)} cases behaved as specified")
    if passed != len(results):
        print("\nSELF-TEST: FAIL")
        return 1
    print("\nSELF-TEST: PASS")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate restructure rename manifests.",
    )
    parser.add_argument("manifests", nargs="*", help="manifest JSON files")
    parser.add_argument(
        "--repo",
        default=None,
        help="repository to resolve locators against (default: the git "
             "toplevel containing the first manifest)",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="schema/group/delete checks only; do not touch git",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="exercise every check against synthetic fixtures and exit",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.manifests:
        parser.error("give at least one manifest, or --self-test")

    repo = args.repo
    if repo is None:
        anchor = Path(args.manifests[0]).resolve().parent
        repo = git(anchor, "rev-parse", "--show-toplevel").strip()
    findings = validate(Path(repo), args.manifests, skip_git=args.skip_git)
    return report(findings, args.manifests)


if __name__ == "__main__":
    sys.exit(main())
