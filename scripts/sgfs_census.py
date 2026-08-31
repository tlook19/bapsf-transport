"""Census of the two-argument ``.get`` fallbacks on the solver's config namespaces.

``LAPDSim1D`` reads its configuration from two dicts, ``self._input_dict`` and
``self._flags``.  Both are produced by ``resolve_config``, which fills every key
its namespace's template owns and raises on any key the template does not own.
A read of the form ``self._input_dict.get(key, default)`` whose ``key`` is
template-owned therefore cannot reach ``default``: the key is always present.
The second argument is dead code that reads as a live default, and a reader who
believes it will believe the wrong number whenever it disagrees with the
template.

This instrument is the gate for dropping those second arguments:

* ``--census`` (the default) lists every two-argument site, splits it into
  template-owned and non-template-owned, and for each template-owned site
  compares the dropped default against the template value with
  ``model_families.values_equal``.
* ``--assert-clean`` additionally requires that no template-owned two-argument
  site remains, which is the post-change state.

Reachability and agreement are two different questions, and this instrument
reports them separately.  Reachability is a property of ``resolve_config``
alone: it fills every template-owned key, so the second argument of a
template-owned ``.get`` is unreachable whatever its value.  Agreement is a
property of the site: a fallback that disagrees with the template value is
still unreachable, but it states a number the solver does not use, so a reader
who trusts it is misled.

Sites are therefore split into ``EQUALS`` (unreachable and consistent — safe
to reduce to one argument) and ``DIFFERS`` (unreachable but inconsistent).
``DIFFERS`` is a **stop-and-report** condition rather than an edit target: it
exits non-zero so that the disagreement is adjudicated by a human instead of
being erased by the same pass that found it.  This instrument never claims a
``DIFFERS`` site is reachable; it claims only that its stated default is not
the value in force.

Sites whose key is not template-owned (a literal that no template declares, or
a non-literal key expression the census cannot resolve) are listed for the
record and left alone: the reachability argument above does not apply to them.

``--against <git-rev>`` prints the REMOVAL RECORD: every site that carried a
two-argument fallback at that revision and carries one argument now, with the
dropped expression beside the template value in force, split into the ones whose
fallback AGREED with the template and the STALE ones whose fallback stated a
value the solver never used. That is the record of what the sweep removed, and
it is regenerated from the tree rather than transcribed.

Usage::

    python scripts/sgfs_census.py                     # census, exit 0 if consistent
    python scripts/sgfs_census.py --assert-clean      # additionally require zero remain
    python scripts/sgfs_census.py --against <rev>     # what a sweep removed, vs <rev>
    python scripts/sgfs_census.py --path <file.py>    # census a different file
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cablp.solvers._sim1d.core.config import (  # noqa: E402
    input_dict_template_1d,
    input_flags_template_1d,
)
from cablp.solvers._sim1d.core.model_families import values_equal  # noqa: E402

DEFAULT_PATH = (
    _REPO_ROOT / "cablp" / "solvers" / "_sim1d" / "solver.py"
)

#: The attribute name each config namespace is read through, and the template
#: that owns its keys.
NAMESPACES = {
    "_input_dict": ("input_dict", input_dict_template_1d),
    "_flags": ("input_flags", input_flags_template_1d),
}

_UNRESOLVED = object()


def _namespace_of(node):
    """Return the namespace attribute name for a ``self.<ns>.get`` call node."""
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "get":
        return None
    owner = func.value
    if not isinstance(owner, ast.Attribute) or owner.attr not in NAMESPACES:
        return None
    if not isinstance(owner.value, ast.Name) or owner.value.id != "self":
        return None
    return owner.attr


def _literal(node):
    """``ast.literal_eval`` the node, or ``_UNRESOLVED`` if it is not a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return _UNRESOLVED


def collect(path, source=None):
    """Every ``self._input_dict.get`` / ``self._flags.get`` call site in ``path``."""
    if source is None:
        source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        ns = _namespace_of(node)
        if ns is None:
            continue
        key = _literal(node.args[0]) if node.args else _UNRESOLVED
        n_args = len(node.args)
        default = _literal(node.args[1]) if n_args >= 2 else None
        default_src = ast.unparse(node.args[1]) if n_args >= 2 else None
        sites.append(
            {
                "line": node.lineno,
                "namespace": ns,
                "key": key,
                "n_args": n_args,
                "default": default,
                "default_src": default_src,
            }
        )
    sites.sort(key=lambda s: s["line"])
    return sites


def classify(sites, path):
    """Split the two-argument sites into template-owned and not, and compare."""
    equal, differs, non_template = [], [], []
    for site in sites:
        if site["n_args"] < 2:
            continue
        space, template = NAMESPACES[site["namespace"]]
        key = site["key"]
        if key is _UNRESOLVED or not isinstance(key, str) or key not in template:
            non_template.append(site)
            continue
        expected = template[key]
        site["template_value"] = expected
        site["space"] = space
        if site["default"] is _UNRESOLVED:
            # A non-literal default cannot be compared statically; report it
            # with the disagreements so a human looks at it rather than having
            # it dropped on an unproven equality.
            site["reason"] = f"default is not a literal: {site['default_src']}"
            differs.append(site)
        elif values_equal(site["default"], expected):
            equal.append(site)
        else:
            site["reason"] = (
                f"default {site['default']!r} != template {expected!r}"
            )
            differs.append(site)
    return equal, differs, non_template


def removal_record(path, rev):
    """The two-argument sites at ``rev`` that carry one argument now.

    Keyed by (namespace, key, ordinal) so repeated reads of the same key are
    matched in source order rather than collapsed.
    """
    old_src = subprocess.run(
        ["git", "show", f"{rev}:{path.relative_to(_REPO_ROOT).as_posix()}"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    old = collect(path, old_src)
    new = collect(path)

    def index(sites, n_args):
        out, seen = {}, {}
        for s in sites:
            if not isinstance(s["key"], str) or s["n_args"] != n_args:
                continue
            k = (s["namespace"], s["key"])
            seen[k] = seen.get(k, 0) + 1
            out[(s["namespace"], s["key"], seen[k])] = s
        return out

    old_two = index(old, 2)
    new_one = index(new, 1)

    agreed, stale = [], []
    for key, site in sorted(old_two.items(), key=lambda kv: kv[1]["line"]):
        if key not in new_one:
            continue
        space, template = NAMESPACES[site["namespace"]]
        if site["key"] not in template:
            continue
        expected = template[site["key"]]
        site["space"] = space
        site["template_value"] = expected
        site["new_line"] = new_one[key]["line"]
        if site["default"] is not _UNRESOLVED and values_equal(
            site["default"], expected
        ):
            agreed.append(site)
        else:
            stale.append(site)
    return agreed, stale


def _fmt(site, path):
    space = site.get("space", site["namespace"])
    return (
        f"{path.name}:{site['line']}  {space}:{site['key']!r}"
        f"  default={site['default_src']}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="file to census (default: the sim1d solver)",
    )
    parser.add_argument(
        "--assert-clean",
        action="store_true",
        help="require that no template-owned two-argument site remains",
    )
    parser.add_argument(
        "--against",
        metavar="REV",
        help="print the removal record against this git revision",
    )
    args = parser.parse_args(argv)

    path = args.path.resolve()

    if args.against:
        agreed, stale = removal_record(path, args.against)
        print(f"=== removal record: {path.name} vs {args.against} ===")
        print(f"  {len(agreed) + len(stale)} fallback(s) dropped "
              f"({len(agreed)} agreed with the template, {len(stale)} STALE)")
        print()
        print(f"--- fallback AGREED with the template ({len(agreed)}) ---")
        for s in agreed:
            print(f"  {path.name}:{s['line']} -> :{s['new_line']}  "
                  f"{s['space']}:{s['key']!r}  dropped {s['default_src']}"
                  f"  == template {s['template_value']!r}")
        print()
        print(f"--- STALE fallback dropped ({len(stale)}) ---")
        print("    (each stated a default the solver never used: the key is")
        print("     always present, so the second argument was unreachable)")
        for s in stale:
            print(f"  {path.name}:{s['line']} -> :{s['new_line']}  "
                  f"{s['space']}:{s['key']!r}  dropped {s['default_src']}"
                  f"  != template {s['template_value']!r}")
        return 0
    sites = collect(path)
    equal, differs, non_template = classify(sites, path)
    two_arg = [s for s in sites if s["n_args"] >= 2]
    one_arg = [s for s in sites if s["n_args"] == 1]

    print(f"=== sgfs census: {path} ===")
    print(f"  {len(sites)} config-namespace .get sites")
    print(f"    {len(one_arg)} one-argument (no fallback)")
    print(f"    {len(two_arg)} two-argument (with fallback)")
    print()

    print(f"--- template-owned, default EQUALS template ({len(equal)}) ---")
    print("    (unreachable fallbacks: resolve_config always fills these keys)")
    for site in equal:
        print(f"  {_fmt(site, path)}  == template {site['template_value']!r}")
    print()

    print(f"--- template-owned, default DIFFERS from template ({len(differs)}) ---")
    if not differs:
        print("  (none)")
    for site in differs:
        print(f"  {_fmt(site, path)}  !! {site['reason']}")
    print()

    print(f"--- NOT template-owned, left alone ({len(non_template)}) ---")
    if not non_template:
        print("  (none)")
    for site in non_template:
        print(f"  {_fmt(site, path)}")
    print()

    failed = False
    if differs:
        print(
            f"STOP: {len(differs)} template-owned site(s) carry a fallback that "
            "disagrees with the template value. Each is unreachable, but each "
            "states a default the solver does not use. Adjudicate before "
            "dropping -- this pass leaves them untouched."
        )
        failed = True

    if args.assert_clean:
        remaining = len(equal) + len(differs)
        if remaining:
            print(
                f"FAIL: {remaining} template-owned two-argument site(s) remain; "
                "--assert-clean requires zero."
            )
            failed = True
        else:
            print("PASS: no template-owned two-argument .get site remains.")

    if failed:
        return 1
    print("sgfs census: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
