#!/usr/bin/env python3
"""Validate restructure rename manifests against the adopted 2026-08-26
manifest-schema constraints.

Those constraints are enforced here and nowhere else -- this module is their
executable statement.  ``RENAME_MAP.md`` section 8 (Q1) carries the same set
in prose for a reader who wants the rationale rather than the mechanism.

Six checks, in the order the schema numbers them:

  (1) SCHEMA      -- JSON shape: required top-level keys, enum values,
                     per-row required keys, locator shape.
  (2) LOCATORS    -- every ``old`` locator resolves at ``base_revision`` and
                     every ``new`` locator at ``new_revision``.  Directory
                     anchors and ``prefix_rule`` ``covers`` resolve by set
                     membership in the ``git ls-tree`` listing of the
                     revision; file, module and symbol anchors read the blob
                     with ``git cat-file``, and symbol anchors additionally
                     walk its AST.
  (3) CHAIN       -- each delta's ``base_revision`` is its predecessor's
                     ``new_revision``; a cumulative manifest SPANS the same
                     window as the chain (its ``base_revision`` is the first
                     delta's and its ``new_revision`` is the last delta's) and
                     equals the composition of the deltas it is checked
                     against.  At most one cumulative may be supplied.
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

PREFIX ROWS (schema-adopted 26dz, Sol assent-with-amendment).  A row may carry
``prefix_rule: {"old_prefix", "new_prefix", "covers"}`` with directory anchors
at both ends.  **A prefix row is a compact mapping MACRO, not an assertion
that the directory is itself a KB entity** -- which is why directory locators
carry ``path`` alone and why ``proposed_continuity`` on such a row VECTORIZES
over the covered file pairs rather than proposing anything about the
directory.  The full constraint set, all enforced below:

  * legal only on ``move`` / ``move+rename``;
  * ``anchor_kind: "directory"`` on BOTH ends, and ``directory`` is legal ONLY
    with a ``prefix_rule``;
  * directory locators carry ``path``, and OMIT symbol/signature/line_hint;
  * ``old.path`` / ``new.path`` equal their prefixes minus the trailing slash;
  * prefixes are normalized, nonempty, repo-relative POSIX paths ending in
    ``/`` -- no absolute paths, no ``.``/``..`` segments, no backslashes;
  * ``covers`` are base-revision old paths: nonempty, sorted, unique, tracked
    at ``base_revision``, each strictly under ``old_prefix``;
  * each derived destination is exactly
    ``new_prefix + old_path.removeprefix(old_prefix)`` and must exist at
    ``new_revision``;
  * no covered path may take a conflicting mapping from another prefix or file
    row -- a finer symbol row on the SAME path pair is a legal override;
  * coverage (check 4) is satisfied by an explicit file/module row OR by
    membership in exactly one prefix rule.

``--emit-expanded`` prints the canonical per-file expansion.  That output is
DERIVED and NON-AUTHORITATIVE: the compact row is the manifest of record.
"""

import argparse
import ast
import json
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

#: Anchor kind for a prefix row's two ends; legal ONLY with a ``prefix_rule``.
DIRECTORY_ANCHOR = "directory"

#: The only change_kinds a prefix_rule may appear on (Sol, 26dz).
PREFIX_RULE_KINDS = frozenset({"move", "move+rename"})

#: Locator fields a directory anchor must NOT carry.
DIRECTORY_FORBIDDEN_FIELDS = ("symbol", "signature", "line_hint")

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

class GitError(RuntimeError):
    """A git invocation failed, or git could not be run at all.

    The validation entry points catch this and record a ``FAIL`` finding, so
    a git-level problem is reported through ``report()``'s normal exit path
    rather than aborting by traceback -- which is what ``Findings`` means by
    "never raises mid-walk".  It subclasses ``RuntimeError`` so the
    self-test's fixture-building calls, where a git failure IS a genuine
    harness abort, keep their previous behaviour.
    """


def _run_git(repo, *arguments):
    """Run git in ``repo`` and return the ``CompletedProcess``.

    Raises ``GitError`` only when the process could not be started at all
    (git absent, ``repo`` not a directory).  Every git access in this module
    funnels through here, so an unrunnable git is a ``GitError`` everywhere
    rather than a bare ``OSError`` from whichever call site reached it first.
    """
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=str(repo), capture_output=True, text=True,
        )
    except OSError as error:
        raise GitError(
            f"git {' '.join(arguments)} could not be run in {repo}: {error}"
        ) from error


def git(repo, *arguments, check=True):
    """Run git in ``repo`` and return stdout; ``check=False`` tolerates exit!=0.

    Raises ``GitError`` when the invocation fails or the executable is absent.
    """
    completed = _run_git(repo, *arguments)
    if check and completed.returncode != 0:
        raise GitError(
            f"git {' '.join(arguments)} failed in {repo}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def blob_at(repo, revision, path):
    """Return the blob text at ``revision:path``, or None when absent."""
    completed = _run_git(repo, "cat-file", "-p", f"{revision}:{path}")
    if completed.returncode != 0:
        return None
    return completed.stdout


def tree_paths(repo, revision):
    """Return the set of tracked paths at ``revision``."""
    out = git(repo, "ls-tree", "-r", "--name-only", revision)
    return set(out.split("\n")) - {""}


def revision_exists(repo, revision):
    completed = _run_git(repo, "rev-parse", "--verify", f"{revision}^{{commit}}")
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
    kind = locator.get("anchor_kind")
    if kind == DIRECTORY_ANCHOR:
        if not allow_directory:
            findings.fail(
                "SCHEMA",
                f"{where}: anchor_kind 'directory' is legal ONLY on a row "
                "carrying a prefix_rule",
            )
        if "path" not in locator:
            findings.fail("SCHEMA", f"{where}: directory locator needs 'path'")
        # A directory is not a KB entity: it has no qualified name to carry.
        for forbidden in DIRECTORY_FORBIDDEN_FIELDS:
            if forbidden in locator:
                findings.fail(
                    "SCHEMA",
                    f"{where}: directory locator must OMIT {forbidden!r} "
                    "(a prefix row is a mapping macro, not an entity claim)",
                )
    else:
        if kind not in ANCHOR_KINDS:
            findings.fail("SCHEMA", f"{where}: bad anchor_kind {kind!r}")
        for key in ("path", "anchor_kind", "symbol"):
            if key not in locator:
                findings.fail("SCHEMA", f"{where}: locator missing {key!r}")
    path = locator.get("path")
    if not isinstance(path, str) or not path or path.startswith("/"):
        findings.fail("SCHEMA", f"{where}: path must be a relative string")
    if "line_hint" in locator and not isinstance(locator["line_hint"], int):
        findings.fail("SCHEMA", f"{where}: line_hint must be an integer hint")


def _prefix_problem(prefix):
    """Return why ``prefix`` is not a normalized repo-relative POSIX prefix."""
    if not isinstance(prefix, str) or not prefix:
        return "must be a non-empty string"
    if "\\" in prefix:
        return "must use POSIX separators (no backslashes)"
    if not prefix.endswith("/"):
        return "must end in '/'"
    if prefix.startswith("/"):
        return "must be repository-relative (no leading '/')"
    segments = prefix.rstrip("/").split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        return "must be normalized (no empty, '.' or '..' segments)"
    return None


def _check_prefix_rule(findings, where, rule, change_kind, old, new):
    """Enforce Sol's prefix-row constraints that need no git.

    Returns the list of well-formed covered paths, so a malformed entry does
    not also cascade into a spurious coverage failure for its siblings.
    """
    if not isinstance(rule, dict):
        findings.fail("SCHEMA", f"{where}: prefix_rule must be an object")
        return []
    if change_kind not in PREFIX_RULE_KINDS:
        findings.fail(
            "SCHEMA",
            f"{where}: prefix_rule is legal only on "
            f"{sorted(PREFIX_RULE_KINDS)}, not change_kind {change_kind!r}",
        )
    for key in ("old_prefix", "new_prefix", "covers"):
        if key not in rule:
            findings.fail("SCHEMA", f"{where}: prefix_rule missing {key!r}")
            return []

    ok = True
    for side, prefix in (("old_prefix", rule["old_prefix"]),
                         ("new_prefix", rule["new_prefix"])):
        problem = _prefix_problem(prefix)
        if problem:
            findings.fail(
                "SCHEMA", f"{where}: prefix_rule.{side} {prefix!r} {problem}"
            )
            ok = False
    if not ok:
        return []

    # Both ends must be directory anchors whose path is the prefix, de-slashed.
    for side, locator, prefix in (("old", old, rule["old_prefix"]),
                                  ("new", new, rule["new_prefix"])):
        if not isinstance(locator, dict):
            continue
        if locator.get("anchor_kind") != DIRECTORY_ANCHOR:
            findings.fail(
                "SCHEMA",
                f"{where}.{side}: a prefix row needs anchor_kind 'directory' "
                f"on BOTH ends, got {locator.get('anchor_kind')!r}",
            )
        expected = prefix.rstrip("/")
        if locator.get("path") != expected:
            findings.fail(
                "SCHEMA",
                f"{where}.{side}: path {locator.get('path')!r} must equal the "
                f"prefix minus its trailing slash ({expected!r})",
            )

    covers = rule["covers"]
    if not isinstance(covers, list) or not covers:
        findings.fail(
            "SCHEMA", f"{where}: prefix_rule.covers must be a non-empty list"
        )
        return []
    if any(not isinstance(c, str) for c in covers):
        findings.fail("SCHEMA", f"{where}: prefix_rule.covers must be strings")
        return []
    if len(set(covers)) != len(covers):
        duplicates = sorted({c for c in covers if covers.count(c) > 1})
        findings.fail(
            "SCHEMA",
            f"{where}: prefix_rule.covers has duplicate entries "
            f"{duplicates[:5]}",
        )
    if covers != sorted(covers):
        findings.fail(
            "SCHEMA",
            f"{where}: prefix_rule.covers must be sorted "
            f"(first out-of-order entry: "
            f"{next(c for c, s in zip(covers, sorted(covers)) if c != s)!r})",
        )

    old_prefix = rule["old_prefix"]
    well_formed = []
    for covered in covers:
        # "strictly under": the prefix itself, or a bare prefix with nothing
        # after it, is not a covered FILE.
        if not covered.startswith(old_prefix) or not covered[len(old_prefix):]:
            findings.fail(
                "SCHEMA",
                f"{where}: prefix_rule.covers entry {covered!r} is not "
                f"strictly under old_prefix {old_prefix!r}",
            )
            continue
        well_formed.append(covered)
    return well_formed


def _derived(rule, covered):
    """The one legal destination for a covered path (Sol: exact, no fuzz)."""
    return rule["new_prefix"] + covered.removeprefix(rule["old_prefix"])


def _well_formed_covers(rule):
    """Covered paths a malformed rule can still be expanded through.

    Used by the expansion helpers so that one bad entry produces exactly one
    SCHEMA failure instead of also cascading into COVERAGE noise for its
    siblings.
    """
    if not isinstance(rule, dict):
        return []
    if _prefix_problem(rule.get("old_prefix")) or _prefix_problem(
        rule.get("new_prefix")
    ):
        return []
    covers = rule.get("covers")
    if not isinstance(covers, list):
        return []
    old_prefix = rule["old_prefix"]
    return [
        c for c in covers
        if isinstance(c, str) and c.startswith(old_prefix) and c[len(old_prefix):]
    ]


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
            _check_prefix_rule(
                findings, where, r["prefix_rule"], kind, r.get("old"),
                r.get("new"),
            )
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
    _check_prefix_conflicts(findings, manifest, source)
    findings.note("SCHEMA", len(rows))


def _check_prefix_conflicts(findings, manifest, source):
    """No covered path may take a conflicting mapping from another row.

    Two distinct violations, both of which the schema names:

      * a covered path belonging to MORE THAN ONE prefix rule -- coverage
        must be "membership in exactly one";
      * a covered path that some other row maps to a DIFFERENT destination.
        A finer symbol row on the SAME path pair is a legal override and is
        deliberately not flagged.
    """
    rows = manifest.get("mappings")
    if not isinstance(rows, list):
        return

    # Which prefix rules claim each covered path, and where each sends it.
    membership = {}
    for index, r in enumerate(rows):
        if not isinstance(r, dict):
            continue
        rule = r.get("prefix_rule")
        if not isinstance(rule, dict):
            continue
        if _prefix_problem(rule.get("old_prefix")) or _prefix_problem(
            rule.get("new_prefix")
        ):
            continue
        covers = rule.get("covers")
        if not isinstance(covers, list):
            continue
        for covered in covers:
            if not isinstance(covered, str):
                continue
            if not covered.startswith(rule["old_prefix"]):
                continue
            membership.setdefault(covered, []).append(
                (index, _derived(rule, covered))
            )

    for covered, claims in sorted(membership.items()):
        if len(claims) > 1:
            findings.fail(
                "SCHEMA",
                f"{source}: {covered!r} is covered by {len(claims)} prefix "
                f"rules (rows {[i for i, _ in claims]}); coverage must be "
                "membership in EXACTLY ONE",
            )

    for index, r in enumerate(rows):
        if not isinstance(r, dict) or r.get("prefix_rule"):
            continue
        old = r.get("old")
        if not isinstance(old, dict):
            continue
        covered = old.get("path")
        if covered not in membership:
            continue
        new = r.get("new")
        destination = new.get("path") if isinstance(new, dict) else None
        for owner, derived in membership[covered]:
            if destination != derived:
                findings.fail(
                    "SCHEMA",
                    f"{source}[{index}]: maps {covered!r} to "
                    f"{destination!r}, conflicting with the prefix rule at "
                    f"row {owner} which derives {derived!r}. A finer row on "
                    "the SAME path pair is a legal override; a different "
                    "destination is not.",
                )


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
        rule = r.get("prefix_rule")
        if isinstance(rule, dict) and not (
            _prefix_problem(rule.get("old_prefix"))
            or _prefix_problem(rule.get("new_prefix"))
        ):
            untracked, missing = [], []
            for covered in rule.get("covers") or []:
                if not isinstance(covered, str) or not covered.startswith(
                    rule["old_prefix"]
                ):
                    continue
                resolved += 1
                if covered not in base_tracked:
                    untracked.append(covered)
                    continue
                if new_tracked is not None:
                    destination = _derived(rule, covered)
                    if destination not in new_tracked:
                        missing.append((covered, destination))
            if untracked:
                findings.fail(
                    "LOCATORS",
                    f"{where}: {len(untracked)} covers entry/entries not "
                    f"tracked at base_revision {base[:12]}: {untracked[:5]}",
                )
            if missing:
                findings.fail(
                    "LOCATORS",
                    f"{where}: {len(missing)} derived destination(s) absent "
                    f"at new_revision {new[:12]}: "
                    + ", ".join(f"{o!r} -> {d!r}" for o, d in missing[:5]),
                )
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
            for covered in _well_formed_covers(rule):
                pairs.setdefault(covered, set()).add(_derived(rule, covered))
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
            for covered in _well_formed_covers(rule):
                old_paths.add(covered)
                new_paths.add(_derived(rule, covered))
            continue
        old = r.get("old")
        new = r.get("new")
        if old is not None and old["anchor_kind"] != DIRECTORY_ANCHOR:
            old_paths.add(old["path"])
        if new is not None and new["anchor_kind"] != DIRECTORY_ANCHOR:
            new_paths.add(new["path"])
    return old_paths, new_paths


def check_chain(findings, deltas, cumulatives):
    """Deltas must abut; a cumulative must span their window and compose.

    ``cumulatives`` is the whole supplied list, not one manifest, because
    "more than one cumulative" is itself a finding: only the first can be
    composed against, so silently dropping the rest would let an unchecked
    manifest ride along inside a green run.

    The WINDOW check is separate from the composition check and both are
    required.  Composition compares path pairs only, so a cumulative that
    declares the wrong end of the window still composes correctly whenever the
    deltas it omits move no file -- a surface or docstring delta, exactly the
    kind that closes an R2 phase.  Such a manifest would then name a window it
    does not span while every check reported PASS.
    """
    ordered = list(deltas)
    for earlier, later in zip(ordered, ordered[1:]):
        if later[1]["base_revision"] != earlier[1]["new_revision"]:
            findings.fail(
                "CHAIN",
                f"{later[0]}: base_revision {later[1]['base_revision'][:12]} "
                f"does not continue {earlier[0]} whose new_revision is "
                f"{earlier[1]['new_revision'][:12]}",
            )
    if not cumulatives:
        findings.skip("CHAIN", "no cumulative manifest supplied to compose against")
        findings.note("CHAIN", max(len(ordered) - 1, 0))
        return
    if len(cumulatives) > 1:
        findings.fail(
            "CHAIN",
            f"{len(cumulatives)} cumulative manifests supplied "
            f"({', '.join(n for n, _ in cumulatives)}); exactly one may be "
            "composed against a chain, and the others would go unchecked",
        )
    cumulative = cumulatives[0]
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
    if ordered:
        for end, chain_end, chain_name in (
            ("base_revision", ordered[0][1]["base_revision"], ordered[0][0]),
            ("new_revision", ordered[-1][1]["new_revision"], ordered[-1][0]),
        ):
            if manifest[end] != chain_end:
                findings.fail(
                    "CHAIN",
                    f"{name}: {end} {manifest[end][:12]} is not the chain's "
                    f"{end} {chain_end[:12]} ({chain_name}); a cumulative must "
                    "span exactly the window its deltas cover",
                )
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
    """Every path git says left or arrived must be covered.

    Coverage means an explicit file/module row OR membership in exactly one
    prefix rule (schema amendment 26dz); the exactly-one half is enforced in
    check (1), which sees the whole row set at once.

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
            "base and new with neither an explicit row nor membership in a "
            f"prefix rule: {uncovered_departed[:8]}",
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

def emit_expanded(manifest_paths):
    """Print the canonical per-file expansion of every prefix row.

    DERIVED AND NON-AUTHORITATIVE (Sol, 26dz).  The compact directory row is
    the manifest of record; this is a reading aid and a review convenience,
    and nothing downstream may treat its output as the manifest.  A prefix
    row's ``proposed_continuity`` is shown against each derived pair, which
    is what vectorization means -- it is proposed per pair, never for the
    directory.
    """
    print("# canonical per-file expansion of the prefix rows")
    print("# DERIVED, NON-AUTHORITATIVE -- the compact row is the manifest of")
    print("# record. Do not cite this output as a manifest.")
    for path in manifest_paths:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"\n## {Path(path).name}")
        print(f"#  base_revision {manifest.get('base_revision')}")
        print(f"#  new_revision  {manifest.get('new_revision')}")
        total = 0
        for index, r in enumerate(manifest.get("mappings") or []):
            rule = r.get("prefix_rule")
            if not isinstance(rule, dict):
                continue
            covers = _well_formed_covers(rule)
            total += len(covers)
            print(
                f"\n#  row {index}: {rule.get('old_prefix')} -> "
                f"{rule.get('new_prefix')}  "
                f"({len(covers)} pairs, change_kind {r.get('change_kind')!r}, "
                f"continuity {r.get('proposed_continuity')!r} per pair)"
            )
            for covered in covers:
                print(f"   {covered}\t{_derived(rule, covered)}")
        print(f"\n#  {total} derived pair(s) in this manifest")
    return 0


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
        try:
            check_locators(findings, repo, manifest, name, draft)
        except GitError as error:
            findings.fail("LOCATORS", f"{name}: {error}")
        if draft:
            findings.skip(
                "COVERAGE",
                f"{name}: skipped (new_revision is {DRAFT_SENTINEL}; the "
                "diff has no second end yet)",
            )
        else:
            try:
                check_coverage(findings, repo, manifest, name)
            except GitError as error:
                findings.fail("COVERAGE", f"{name}: {error}")

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
        check_chain(findings, deltas, cumulatives)
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
    """Five commits: root -> base -> mid -> tip -> tail.

    ``base -> mid -> tip`` carries the structure (moves, a split, a delete, a
    prefix subtree); two such steps are what the CHAIN check needs to have
    something real to compose.

    ``root -> base`` and ``tip -> tail`` are MODIFY-ONLY: one line of one file
    changes and no path arrives or departs.  They exist for the window
    fixtures, which need a delta that legitimately moves nothing -- that is
    precisely the delta a composition-only check cannot see past.
    """
    _init_repo(repo)
    root = _commit(repo, {
        "README": "root\n",
        "old/mod.py": "def alpha(x):\n    return x\n",
        "old/data.csv": "1,2\n",
        "old/gone.py": "def dead():\n    pass\n",
        "old/whole.py": "def whole():\n    pass\n",
        # A subtree the prefix-row fixtures operate on. `orphan.py` exists at
        # base and has NO counterpart at mid, which is what makes the
        # missing-derived-destination fixture possible.
        "old/pkg/a.py": "def a():\n    pass\n",
        "old/pkg/b.py": "def b():\n    pass\n",
        "old/pkg/orphan.py": "def orphan():\n    pass\n",
    }, "root")
    base = _commit(repo, {"README": "base\n"}, "base")
    git(repo, "rm", "-q", "old/mod.py", "old/data.csv", "old/gone.py",
        "old/whole.py", "old/pkg/a.py", "old/pkg/b.py", "old/pkg/orphan.py")
    mid = _commit(repo, {
        "new/mod.py": "def alpha(x):\n    return x\n",
        "new/data.csv": "1,2\n",
        "new/part_a.py": "def whole_a():\n    pass\n",
        "new/part_b.py": "def whole_b():\n    pass\n",
        "new/pkg/a.py": "def a():\n    pass\n",
        "new/pkg/b.py": "def b():\n    pass\n",
    }, "mid")
    git(repo, "rm", "-q", "new/mod.py")
    tip = _commit(repo, {
        "final/mod.py": "def alpha(x):\n    return x\n",
    }, "tip")
    tail = _commit(repo, {"README": "tail\n"}, "tail")
    return root, base, mid, tip, tail


def self_test():
    """Exercise a positive control plus one deliberate failure of each check.

    Every failure fixture is built so that exactly ONE check fires: a case
    that tripped two checks would not tell us the check under test works.
    """
    import copy

    results = []

    def record(label, expect, findings):
        """``expect`` is None for a clean pass, else the exact set of checks."""
        failed = {c for c, _ in findings.failures}
        wanted = set() if expect is None else (
            {expect} if isinstance(expect, str) else set(expect)
        )
        ok = failed == wanted
        got = f"failed {sorted(failed)}" if failed else "clean"
        results.append((label, expect, ok, got))

    with tempfile.TemporaryDirectory() as workspace:
        workspace = Path(workspace)
        repo = workspace / "repo"
        repo.mkdir()
        root, base, mid, tip, tail = _build_fixture_repo(repo)
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
            # The prefix MACRO: directory anchors at both ends, path only.
            _row(
                "move",
                {"path": "old/pkg", "anchor_kind": "directory"},
                {"path": "new/pkg", "anchor_kind": "directory"},
                "same_entity",
                prefix_rule={
                    "old_prefix": "old/pkg/",
                    "new_prefix": "new/pkg/",
                    "covers": ["old/pkg/a.py", "old/pkg/b.py"],
                },
            ),
            _row(
                "delete",
                _loc("old/pkg/orphan.py", "module", "pkg.old.pkg.orphan"),
                None,
                "retired_no_successor",
                deletion_reason="no consumer; retired with the subtree move",
            ),
        ]
        PREFIX_ROW, ORPHAN_ROW = 5, 6

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

        # --- prefix-row constraints (Sol, 26dz) ---------------------------
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["prefix_rule"]["covers"] = [
            "old/pkg/b.py", "old/pkg/a.py",
        ]
        one("fail-covers-unsorted", bad, "SCHEMA")

        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["prefix_rule"]["covers"] = [
            "old/pkg/a.py", "old/pkg/a.py", "old/pkg/b.py",
        ]
        one("fail-covers-duplicate", bad, "SCHEMA")

        # `old/gone.py` is tracked at base and mapped by its own delete row,
        # so listing it here trips ONLY the strictly-under-old_prefix rule.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["prefix_rule"]["covers"] = [
            "old/gone.py", "old/pkg/a.py", "old/pkg/b.py",
        ]
        one("fail-cover-outside-prefix", bad, "SCHEMA")

        # `old/pkg/orphan.py` has no counterpart at mid. Covering it (and
        # dropping its delete row, so there is no conflicting mapping) leaves
        # exactly one failure: the derived destination does not exist.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["prefix_rule"]["covers"] = [
            "old/pkg/a.py", "old/pkg/b.py", "old/pkg/orphan.py",
        ]
        del bad[ORPHAN_ROW]
        one("fail-derived-destination-missing", bad, "LOCATORS")

        # A second row sending a covered path somewhere else. Both ends
        # resolve and coverage stays satisfied, so only the conflict fires.
        bad = copy.deepcopy(good_rows)
        bad.append(_row(
            "move",
            _loc("old/pkg/a.py", "module", "pkg.old.pkg.a"),
            _loc("new/mod.py", "module", "pkg.new.mod"),
            "same_entity",
        ))
        one("fail-conflicting-mapping", bad, "SCHEMA")

        # A directory anchor with no prefix_rule. The real prefix row is left
        # in place so coverage is still satisfied and the fixture isolates.
        bad = copy.deepcopy(good_rows)
        bad.append(_row(
            "move",
            {"path": "old/pkg", "anchor_kind": "directory"},
            {"path": "new/pkg", "anchor_kind": "directory"},
            "same_entity",
        ))
        one("fail-directory-without-rule", bad, "SCHEMA")

        # A directory locator that carries a symbol -- a prefix row is a
        # mapping macro, so it must not name an entity.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["old"]["symbol"] = "pkg.old.pkg"
        one("fail-directory-carries-symbol", bad, "SCHEMA")

        # prefix_rule on a change_kind that may not carry one.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["change_kind"] = "surface_change"
        one("fail-prefix-on-wrong-kind", bad, "SCHEMA")

        # Prefix normalization: a backslash separator. This is the one
        # fixture that legitimately trips TWO checks, and the expectation
        # says so rather than the validator being loosened to make it
        # pretty: a rule whose prefix is unusable expands to nothing, so its
        # covered files are genuinely uncovered and COVERAGE is right to
        # fire alongside SCHEMA.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["prefix_rule"]["new_prefix"] = "new\\pkg\\"
        one("fail-prefix-not-normalized", bad, {"SCHEMA", "COVERAGE"})

        # old.path must equal old_prefix minus the trailing slash.
        bad = copy.deepcopy(good_rows)
        bad[PREFIX_ROW]["old"]["path"] = "old/pkg/"
        one("fail-prefix-path-mismatch", bad, "SCHEMA")

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
            copy.deepcopy(good_rows[PREFIX_ROW]),
            copy.deepcopy(good_rows[ORPHAN_ROW]),
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

        # (c) WINDOW, tip end.  `delta3` spans a modify-only range, so it moves
        #     no file and the composition is byte-for-byte what it was without
        #     it.  The cumulative below therefore COMPOSES correctly, resolves
        #     at both its declared ends and reconciles against its own diff --
        #     and still names a window one delta short of the chain's.  Only
        #     the window check can see that, which is why it is separate.
        delta3 = _base_manifest(tip, tail, [])
        record("fail-CHAIN-window-tip", "CHAIN", validate(repo, [
            _write(manifests, "chain-wt-a.json", delta1),
            _write(manifests, "chain-wt-b.json", delta2),
            _write(manifests, "chain-wt-c.json", delta3),
            _write(manifests, "chain-wt-cumulative.json", cumulative),
        ]))

        # (d) WINDOW, base end.  The mirror image: `delta0` spans the
        #     modify-only root->base range, and the cumulative starts one
        #     delta late.
        delta0 = _base_manifest(root, base, [])
        record("fail-CHAIN-window-base", "CHAIN", validate(repo, [
            _write(manifests, "chain-wb-0.json", delta0),
            _write(manifests, "chain-wb-a.json", delta1),
            _write(manifests, "chain-wb-b.json", delta2),
            _write(manifests, "chain-wb-cumulative.json", cumulative),
        ]))

        # (e) Only ONE cumulative may be composed against a chain.  Both
        #     copies here are the valid manifest, so every other check is
        #     clean and the finding is purely that the second would go
        #     unchecked.
        record("fail-CHAIN-two-cumulatives", "CHAIN", validate(repo, [
            _write(manifests, "chain-2c-a.json", delta1),
            _write(manifests, "chain-2c-b.json", delta2),
            _write(manifests, "chain-2c-cumulative-1.json", cumulative),
            _write(manifests, "chain-2c-cumulative-2.json", cumulative),
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
            "clean; CHAIN+COVERAGE skipped, LOCATORS new-end skipped",
            not failed and {"COVERAGE", "CHAIN", "LOCATORS"} <= skipped,
            f"failed={sorted(failed)} skipped={sorted(skipped)}",
        ))

        # --- a git-level failure reports as a FAIL, never as a traceback ---
        # A repo path that does not exist makes the first git call in
        # check_locators unstartable.  Draft mode keeps COVERAGE skipped so
        # exactly one check fires.
        record("fail-git-unrunnable", "LOCATORS", validate(
            workspace / "no-such-repo",
            [_write(manifests, "gitfail.DRAFT.json", draft)],
        ))

    controls = sum(1 for _, expect, _, _ in results if expect is None)
    drafts = sum(
        1 for _, expect, _, _ in results if isinstance(expect, str) and " " in expect
    )
    print("validate_manifest --self-test")
    print(f"  {len(results)} cases over a synthetic five-commit repository:")
    print(f"  {controls} positive controls, "
          f"{len(results) - controls - drafts} failure fixtures, "
          f"{drafts} draft-mode case.")
    print("  Each failure fixture is built so exactly ONE check fires, except")
    print("  fail-prefix-not-normalized, where a second failure is a true")
    print("  consequence and the expectation names both.\n")
    width = max(len(label) for label, *_ in results)
    for label, expect, ok, got in results:
        if expect is None:
            expected = "clean pass"
        elif isinstance(expect, str) and " " in expect:
            expected = expect
        elif isinstance(expect, str):
            expected = f"FAIL in {expect}"
        else:
            expected = f"FAIL in {sorted(expect)}"
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
        "--emit-expanded",
        action="store_true",
        help="print the canonical per-file expansion of every prefix row and "
             "exit; DERIVED and NON-AUTHORITATIVE (the compact row is the "
             "manifest of record)",
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
    if args.emit_expanded:
        return emit_expanded(args.manifests)

    repo = args.repo
    if repo is None:
        anchor = Path(args.manifests[0]).resolve().parent
        try:
            repo = git(anchor, "rev-parse", "--show-toplevel").strip()
        except GitError as error:
            # Nothing ran, so say so: without these skips report() would
            # print PASS for five checks that were never executed.
            findings = Findings()
            findings.fail(
                "LOCATORS", f"repository discovery failed under {anchor}: {error}"
            )
            for check in ("SCHEMA", "CHAIN", "COVERAGE", "GROUPS", "DELETES"):
                findings.skip(check, "not run: no repository to validate against")
            return report(findings, args.manifests)
    findings = validate(Path(repo), args.manifests, skip_git=args.skip_git)
    return report(findings, args.manifests)


if __name__ == "__main__":
    sys.exit(main())
