"""Aggregate and compare folded-stack profiles.

Folded stacks (``frame;frame;... count``) are the archival profiling format:
both ``profile_sim1d.py --mode sample`` and ``py-spy record --format raw`` emit
them, and every other view (self/total tables, flamegraphs, speedscope) can be
regenerated from them.  This script is the reader.

Why normalisation is needed for a comparison
--------------------------------------------
The two producers label frames differently:

* the in-process sampler uses ``co_qualname`` and the function's FIRST line
  (``LAPDSim1D._attempt_step (cablp/.../solver.py:1832)``);
* py-spy uses the bare function name and the CURRENTLY EXECUTING line
  (``_attempt_step (solver.py:1904)``).

So exact string matching finds almost nothing in common.  ``--normalize``
reduces every frame to ``basename:funcname``, dropping line numbers and the
qualname prefix, which makes the two directly comparable.

Usage::

    # rank one profile
    python scripts/analyze_folded_profile.py run_folded.txt

    # cross-check the in-process sampler against py-spy on the same run
    python scripts/analyze_folded_profile.py --normalize \
        --compare es1_prod_sample_nx240_folded.txt es1_prod_pyspy_nx240_folded.txt
"""

import argparse
from collections import Counter
from pathlib import Path


def read_folded(path, root=None):
    """Parse a folded-stack file into ``{stack_tuple: count}``.

    ``root`` keeps only stacks whose outermost frame starts with that string.
    py-spy aggregates every thread into one file with no thread prefix, so this
    is how a multi-threaded capture is split back apart: the solver thread roots
    at ``<module>`` while helper threads root at ``_bootstrap (threading.py..)``.
    Mixing them understates the solver's own percentages.
    """
    counts = Counter()
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            stack, _, count = line.rpartition(" ")
            if not stack:
                continue
            try:
                n = int(count)
            except ValueError:
                continue
            frames = tuple(stack.split(";"))
            if root is not None and not frames[0].startswith(root):
                continue
            counts[frames] += n
    return counts


def normalize_frame(frame):
    """Reduce a frame label to ``basename:funcname``, line numbers dropped.

    Handles both producers' shapes:
      ``LAPDSim1D._attempt_step (cablp/x/solver.py:1832)`` -> ``solver.py:_attempt_step``
      ``_attempt_step (solver.py:1904)``                   -> ``solver.py:_attempt_step``
    """
    name, _, location = frame.partition(" (")
    name = name.strip()
    # qualname -> bare function name, so Class.method matches method
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    location = location.rstrip(")")
    if ":" in location:
        location = location.rsplit(":", 1)[0]
    base = Path(location).name if location else "?"
    return f"{base}:{name}" if base != "?" else name


def normalize_counts(counts):
    out = Counter()
    for stack, n in counts.items():
        out[tuple(normalize_frame(f) for f in stack)] += n
    return out


def self_total(counts):
    """Leaf (self) and any-frame (total) tallies; see profile_sim1d."""
    self_counts = Counter()
    total_counts = Counter()
    for stack, n in counts.items():
        if not stack:
            continue
        self_counts[stack[-1]] += n
        for frame in set(stack):
            total_counts[frame] += n
    return self_counts, total_counts


def print_table(title, tally, total_samples, top):
    print(f"\n{title}")
    print(f"{'%':>7}  {'samples':>9}  frame")
    print("-" * 96)
    for frame, count in tally.most_common(top):
        print(f"{100.0 * count / total_samples:7.2f}  {count:9d}  {frame}")


def compare(path_a, path_b, top, normalize, root=None):
    counts_a = read_folded(path_a, root)
    counts_b = read_folded(path_b, root)
    if normalize:
        counts_a = normalize_counts(counts_a)
        counts_b = normalize_counts(counts_b)
    self_a, total_a = self_total(counts_a)
    self_b, total_b = self_total(counts_b)
    n_a = sum(counts_a.values())
    n_b = sum(counts_b.values())

    print("=" * 96)
    print("FOLDED-PROFILE COMPARISON")
    print("=" * 96)
    print(f"A = {Path(path_a).name}   ({n_a} samples)")
    print(f"B = {Path(path_b).name}   ({n_b} samples)")
    print(f"normalized frames: {normalize}")

    for label, tally_a, tally_b in (
        ("SELF TIME", self_a, self_b),
        ("TOTAL TIME", total_a, total_b),
    ):
        frames = [f for f, _ in tally_a.most_common(top)]
        for frame, _ in tally_b.most_common(top):
            if frame not in frames:
                frames.append(frame)
        rows = []
        for frame in frames:
            pct_a = 100.0 * tally_a.get(frame, 0) / n_a if n_a else 0.0
            pct_b = 100.0 * tally_b.get(frame, 0) / n_b if n_b else 0.0
            rows.append((max(pct_a, pct_b), pct_a, pct_b, frame))
        rows.sort(reverse=True)
        print(f"\n{label}: A% vs B% (sorted by the larger of the two)")
        print(f"{'A%':>7}  {'B%':>7}  {'delta':>7}  frame")
        print("-" * 96)
        for _, pct_a, pct_b, frame in rows[:top]:
            print(f"{pct_a:7.2f}  {pct_b:7.2f}  {pct_b - pct_a:+7.2f}  {frame}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folded", nargs="*", help="folded-stack file(s) to rank")
    parser.add_argument(
        "--compare", nargs=2, metavar=("A", "B"), default=None,
        help="compare two folded profiles frame by frame",
    )
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument(
        "--root", default=None,
        help="keep only stacks whose outermost frame starts with this "
        "(e.g. '<module>' for the solver thread in a py-spy capture)",
    )
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args(argv)

    if args.compare:
        compare(args.compare[0], args.compare[1], args.top, args.normalize, args.root)
        return 0

    if not args.folded:
        parser.error("give at least one folded file, or --compare A B")

    for path in args.folded:
        counts = read_folded(path, args.root)
        if args.normalize:
            counts = normalize_counts(counts)
        total = sum(counts.values())
        self_counts, total_counts = self_total(counts)
        print("=" * 96)
        print(f"{Path(path).name}   ({total} samples, {len(counts)} distinct stacks)")
        print("=" * 96)
        print_table(f"TOP {args.top} BY SELF TIME", self_counts, total, args.top)
        print_table(f"TOP {args.top} BY TOTAL TIME", total_counts, total, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
