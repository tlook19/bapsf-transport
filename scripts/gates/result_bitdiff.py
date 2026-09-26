"""Full-result bit-diff gate for LAPDSim1D ``sim1d-hdf5-v1`` result files.

The golden gate compares three arrays of a run (``time``, ``y``, ``phase``)
with ``np.array_equal``. A saved result carries far more -- the per-term RHS
rows, the cathode diagnostics, the ledgers, the circuit rows, the DVM ledgers,
the configuration lineage -- and a change can move any of those while the
state trajectory stays bit-identical. This gate compares EVERY saved byte.

Modes::

    # compare two result files; exit 0 only if identical
    python scripts/gates/result_bitdiff.py compare A.h5 B.h5

    # run the fixed matrix under two code trees and compare each pair
    python scripts/gates/result_bitdiff.py matrix \\
        --base <rev-or-tree> --head <rev-or-tree> --outdir <dir outside the repo>

    # prove the allow-list complete and the comparator able to fail
    python scripts/gates/result_bitdiff.py --self-test --outdir <dir>

    # print the allow-list
    python scripts/gates/result_bitdiff.py allow-list

WHAT ``compare`` CHECKS. Both files are walked in full: every group, dataset,
soft/external link and attribute, root attributes included. It reports

(a) paths present in one file only (objects and attributes alike);
(b) object-kind, HDF5 datatype (``H5Tequal``) and shape mismatches;
(c) datasets whose RAW BYTES differ -- fixed-width data is compared as its
    underlying buffer, element by element, so ``-0.0`` against ``+0.0`` and
    two NaN payloads are differences; variable-length strings are compared as
    their encoded bytes -- with the first differing flat index and both values;
(d) attributes whose values differ, by the same rules.

Paths are written ``/group/dataset`` for objects and ``/group/dataset@name``
for attributes (``/@name`` for a root attribute).

THE ALLOW-LIST. A field the code stamps from the clock or the environment
differs between two otherwise identical runs. Such a field is excluded ONLY
through :data:`ALLOW_LIST`, one exact path per entry with its reason. An
allow-listed path is still checked for PRESENCE, kind, datatype and shape;
only its value is exempt. Nothing else is ignored.

THE MATRIX. :data:`MATRIX` names committed configurations, each run on the
golden gate's own route -- ``baseline_sim1d.build_baseline_config`` layering
(``default_config()`` + the configuration minus its mesh-sized package +
``BASELINE_PARAM_OVERRIDES`` / ``BASELINE_FLAG_OVERRIDES``), the digest gate's
``max_steps_action = "stop"`` and the golden's run controls with ``max_steps``
set to the entry's step count -- and saved with ``save_result_hdf5``. Each leg
runs in its own process with ``PYTHONPATH`` and the working directory set to
its tree, and imports both ``cablp`` and the ``scripts/`` modules from that
tree, which the leg asserts before running. A tree is either a directory (a
checkout or worktree) or a git revision, which is exported with
``git archive`` under ``<outdir>/trees/`` and given the untracked OPEN-ADAS
masters from the tree this script lives in. Legs run the pure kernel path
with ``CABLP_COMPILED_KERNELS`` removed from their environment; ``--compiled``
sets it on both legs (and builds the extension in an exported tree).

THE SELF-TEST. (1) One matrix entry runs twice on the same tree and the pair
must compare identical. (2) Negative controls on copies of that result: one
``+0.0``/``-0.0`` sign flip in a float dataset, one low-order byte flip in a
different float dataset, one changed attribute and one extra dataset; each
must be reported as exactly that one difference. (3) Allow-list evidence:
every entry must be shown to differ between two identically produced files,
with the allow-list disabled, and those two files must differ nowhere else.
"""

import argparse
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np

THIS_FILE = Path(__file__).resolve()
THIS_TREE = THIS_FILE.parents[2]
SCRIPT_SUBDIRS = ("atomic", "gates", "kinetic", "run", "score", "stance",
                  "verify")


# ----------------------------------------------------------------------
# The allow-list: every run-to-run nondeterministic path, with its reason.
# ----------------------------------------------------------------------
ALLOW_LIST = {
    "/@run_id": (
        "qualified-capture execution identity: an RFC 9562 UUID the capture "
        "workflow assigns per execution, never derived from the run"
    ),
    "/@started_at": (
        "qualified-capture UTC wall-clock stamp taken before the solve"
    ),
    "/@completed_at": (
        "qualified-capture UTC wall-clock stamp taken after the solve"
    ),
    "/ignition_abort@wall_clock_s": (
        "process wall clock (perf_counter) elapsed when an ignition budget "
        "guard opened the cathode switch"
    ),
}


# ----------------------------------------------------------------------
# compare
# ----------------------------------------------------------------------
@dataclasses.dataclass
class Report:
    """Outcome of one comparison: the findings and what was compared."""

    findings: list
    allow_listed: list
    objects: int = 0
    attributes: int = 0
    bytes_compared: int = 0

    @property
    def identical(self):
        return not self.findings


def _walk(h5):
    """Return ``(objects, attributes)`` for an open file.

    ``objects`` maps a path to ``(kind, handle_or_target)`` for every group,
    dataset, soft link and external link, the root group included.
    ``attributes`` maps ``path@name`` to ``(owner, name)``.
    """
    objects = {"/": ("group", h5)}
    attributes = {}

    def attrs_of(path, owner):
        for name in owner.attrs.keys():
            prefix = "/" if path == "/" else path
            attributes[f"{prefix}@{name}"] = (owner, name)

    def recurse(path, group):
        attrs_of(path, group)
        for name in group.keys():
            child = f"{path.rstrip('/')}/{name}"
            link = group.get(name, getlink=True)
            if isinstance(link, h5py.SoftLink):
                objects[child] = ("softlink", link.path)
                continue
            if isinstance(link, h5py.ExternalLink):
                objects[child] = ("extlink", (link.filename, link.path))
                continue
            item = group[name]
            if isinstance(item, h5py.Group):
                objects[child] = ("group", item)
                recurse(child, item)
            else:
                objects[child] = ("dataset", item)
                attrs_of(child, item)

    recurse("/", h5)
    return objects, attributes


def _is_string_type(dtype):
    return h5py.check_string_dtype(dtype) is not None


def _as_bytes_list(value):
    """Flatten a string value (scalar or array) to a list of exact bytes."""
    flat = np.asarray(value, dtype=object).reshape(-1)
    out = []
    for item in flat:
        if isinstance(item, str):
            out.append(item.encode("utf-8"))
        elif isinstance(item, (bytes, np.bytes_)):
            out.append(bytes(item))
        else:
            out.append(repr(item).encode("utf-8"))
    return out


def _format_element(array, index):
    """Render one element of a fixed-width array with its raw bits."""
    flat = array.reshape(-1)
    element = flat[index]
    raw = np.ascontiguousarray(flat[index:index + 1]).view(np.uint8).tobytes()
    return f"{element!r} (bits 0x{raw[::-1].hex()})"


def _compare_values(label, value_a, value_b, dtype_a, report):
    """Compare two values of the same datatype and shape, byte for byte.

    Returns the finding text, or ``None`` when identical.
    """
    if _is_string_type(dtype_a):
        list_a = _as_bytes_list(value_a)
        list_b = _as_bytes_list(value_b)
        report.bytes_compared += sum(len(item) for item in list_a)
        for index, (item_a, item_b) in enumerate(zip(list_a, list_b)):
            if item_a != item_b:
                return (
                    f"{label} VALUE differs at flat index {index}: "
                    f"{item_a!r} vs {item_b!r}"
                )
        if len(list_a) != len(list_b):
            return f"{label} VALUE differs: {len(list_a)} vs {len(list_b)} strings"
        return None
    array_a = np.ascontiguousarray(np.asarray(value_a, dtype=dtype_a))
    array_b = np.ascontiguousarray(np.asarray(value_b, dtype=dtype_a))
    itemsize = array_a.dtype.itemsize
    report.bytes_compared += array_a.size * itemsize
    rows_a = array_a.reshape(-1).view(np.uint8).reshape(-1, max(itemsize, 1))
    rows_b = array_b.reshape(-1).view(np.uint8).reshape(-1, max(itemsize, 1))
    if rows_a.shape != rows_b.shape:
        return f"{label} VALUE differs: {rows_a.shape} vs {rows_b.shape} bytes"
    unequal = np.any(rows_a != rows_b, axis=1)
    if not unequal.any():
        return None
    index = int(np.argmax(unequal))
    count = int(unequal.sum())
    return (
        f"{label} BYTES differ at {count} of {unequal.size} elements; first at "
        f"flat index {index}: {_format_element(array_a, index)} vs "
        f"{_format_element(array_b, index)}"
    )


def _read_dataset(dataset):
    if dataset.shape is None:
        return None
    return dataset[()]


def compare_files(path_a, path_b, allow_list=None):
    """Compare two HDF5 files in full and return a :class:`Report`.

    ``allow_list`` defaults to :data:`ALLOW_LIST`; pass ``{}`` to exempt
    nothing.
    """
    allow = ALLOW_LIST if allow_list is None else allow_list
    report = Report(findings=[], allow_listed=[])
    with h5py.File(path_a, "r") as h5_a, h5py.File(path_b, "r") as h5_b:
        objects_a, attrs_a = _walk(h5_a)
        objects_b, attrs_b = _walk(h5_b)
        for path in sorted(set(objects_a) ^ set(objects_b)):
            side = "A" if path in objects_a else "B"
            kind = (objects_a.get(path) or objects_b.get(path))[0]
            report.findings.append(f"{path}: {kind} present in {side} only")
        for path in sorted(set(attrs_a) ^ set(attrs_b)):
            side = "A" if path in attrs_a else "B"
            report.findings.append(f"{path}: attribute present in {side} only")

        for path in sorted(set(objects_a) & set(objects_b)):
            report.objects += 1
            kind_a, obj_a = objects_a[path]
            kind_b, obj_b = objects_b[path]
            if kind_a != kind_b:
                report.findings.append(f"{path}: KIND {kind_a} vs {kind_b}")
                continue
            if kind_a in ("softlink", "extlink"):
                if obj_a != obj_b:
                    report.findings.append(
                        f"{path}: LINK TARGET {obj_a!r} vs {obj_b!r}"
                    )
                continue
            if kind_a != "dataset":
                continue
            if obj_a.id.get_type() != obj_b.id.get_type():
                report.findings.append(
                    f"{path}: DTYPE {obj_a.dtype!r} vs {obj_b.dtype!r}"
                )
                continue
            if obj_a.shape != obj_b.shape:
                report.findings.append(
                    f"{path}: SHAPE {obj_a.shape} vs {obj_b.shape}"
                )
                continue
            if path in allow:
                report.allow_listed.append(path)
                continue
            if obj_a.shape is None:
                continue
            finding = _compare_values(
                path, _read_dataset(obj_a), _read_dataset(obj_b),
                obj_a.dtype, report,
            )
            if finding:
                report.findings.append(finding)

        for path in sorted(set(attrs_a) & set(attrs_b)):
            report.attributes += 1
            owner_a, name = attrs_a[path]
            owner_b, _ = attrs_b[path]
            aid_a = owner_a.attrs.get_id(name)
            aid_b = owner_b.attrs.get_id(name)
            if aid_a.get_type() != aid_b.get_type():
                report.findings.append(
                    f"{path}: DTYPE {aid_a.dtype!r} vs {aid_b.dtype!r}"
                )
                continue
            if aid_a.shape != aid_b.shape:
                report.findings.append(
                    f"{path}: SHAPE {aid_a.shape} vs {aid_b.shape}"
                )
                continue
            if path in allow:
                report.allow_listed.append(path)
                continue
            if aid_a.shape is None:
                continue
            finding = _compare_values(
                path, owner_a.attrs[name], owner_b.attrs[name],
                aid_a.dtype, report,
            )
            if finding:
                report.findings.append(finding)
    return report


def print_report(label, report, stream=sys.stdout):
    status = "IDENTICAL" if report.identical else "DIFFERENT"
    print(
        f"result_bitdiff compare [{label}]: {status} -- "
        f"{report.objects} objects, {report.attributes} attributes, "
        f"{report.bytes_compared} bytes compared, "
        f"{len(report.findings)} finding(s)",
        file=stream,
    )
    for path in report.allow_listed:
        print(f"  allow-listed (value not compared): {path}", file=stream)
    for finding in report.findings:
        print(f"  {finding}", file=stream)


def cmd_compare(args):
    report = compare_files(args.a, args.b)
    print_report(f"{args.a} vs {args.b}", report)
    return 0 if report.identical else 1


def cmd_allow_list(_args):
    print(f"result_bitdiff allow-list: {len(ALLOW_LIST)} entr"
          f"{'y' if len(ALLOW_LIST) == 1 else 'ies'}")
    for path, reason in ALLOW_LIST.items():
        print(f"  {path}: {reason}")
    return 0


# ----------------------------------------------------------------------
# The matrix
# ----------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class MatrixEntry:
    """One matrix run: a committed configuration and its step count.

    ``spec`` is a committed stance name or a configuration file path
    relative to the tree root. ``param_overrides`` layer last; only the
    allow-list evidence probe uses them.
    """

    name: str
    spec: str
    steps: int
    param_overrides: tuple = ()


MATRIX = (
    # The reference configuration at the golden's route, over the digest
    # gate's horizon.
    MatrixEntry("g1atrim", "g1atrim", 4000),
    # The prescribed-measured drive: the ES1 measured trace takes the drive
    # over at 0.07 ms, a cathode solver model the reference never runs.
    MatrixEntry(
        "es1_reference",
        "scripts/stances/examples/g1atrim_es1_reference.toml",
        500,
    ),
    # The emitting end face's sheath debit: three cathode-cell electron
    # energy rows the reference does not carry.
    MatrixEntry(
        "cathode_face_full_debit",
        "scripts/stances/examples/g1atrim_cathode_face_full_debit.toml",
        500,
    ),
)
MATRIX_BY_NAME = {entry.name: entry for entry in MATRIX}

#: The ignition-budget probe used as allow-list evidence: the reference at the
#: golden route with the accepted-step budget set so the guard opens the
#: switch after one step, before breakdown.
IGNITION_ABORT_PROBE = MatrixEntry(
    "ignition_abort_probe", "g1atrim", 50,
    (("ignition_accepted_step_cap", 1),),
)


def _leg_main(args):
    """Run one matrix leg in THIS process against ``args.tree``.

    Called only through the ``_leg`` subcommand, in a subprocess whose
    ``PYTHONPATH`` and working directory are the tree.
    """
    tree = Path(args.tree).resolve()
    own_dir = str(THIS_FILE.parent)
    sys.path[:] = [p for p in sys.path if p != own_dir]
    for sub in SCRIPT_SUBDIRS:
        sys.path.insert(0, str(tree / "scripts" / sub))

    import cablp
    from cablp.cathode import kernels as kernel_selector
    from cablp.solvers._sim1d import LAPDSim1D, default_config, save_result_hdf5
    import baseline_sim1d
    import golden_digest_gate
    import stance_config

    for module in (cablp, baseline_sim1d, golden_digest_gate, stance_config):
        origin = Path(module.__file__).resolve()
        if tree not in origin.parents:
            raise RuntimeError(
                f"leg import escaped its tree: {module.__name__} from "
                f"{origin}, tree {tree}"
            )
    provenance = str(kernel_selector.PROVENANCE)
    if args.compiled != (provenance != "pure"):
        raise RuntimeError(
            f"kernel provenance {provenance!r} does not match the requested "
            f"{'compiled' if args.compiled else 'pure'} path"
        )
    print(f"leg: tree={tree} cablp={Path(cablp.__file__).resolve()} "
          f"kernels={provenance}", flush=True)

    digest_overrides = golden_digest_gate.DIGEST_PARAM_OVERRIDES
    if args.spec == baseline_sim1d.PRODUCTION_STANCE:
        params, flags = baseline_sim1d.build_baseline_config(digest_overrides)
    else:
        stance = stance_config.load_named_configuration(args.spec)
        stance_params, stance_flags = stance_config.without_mesh_sized_package(
            dict(stance.params), dict(stance.flags)
        )
        params, flags = default_config()
        params.update(stance_params)
        flags.update(stance_flags)
        params.update(baseline_sim1d.BASELINE_PARAM_OVERRIDES)
        flags.update(baseline_sim1d.BASELINE_FLAG_OVERRIDES)
        params.update(digest_overrides)
    params.update(json.loads(args.param_overrides))
    _, _, lineage = stance_config.load_configuration(args.spec)
    lineage = lineage.with_identity(params, flags)

    sim = LAPDSim1D(params, flags, configuration=lineage)
    run_kwargs = dict(baseline_sim1d.BASELINE_RUN_KWARGS)
    run_kwargs["max_steps"] = int(args.steps)
    started = time.perf_counter()
    sim.start_simulation(**run_kwargs)
    result = sim.get_results()
    save_result_hdf5(args.out, result)
    print(
        f"leg: spec={args.spec} steps={result.steps} "
        f"final_time={result.final_time:.6e} saves={len(result.time)} "
        f"identity={lineage.identity} solve_wall_s="
        f"{time.perf_counter() - started:.1f} out={args.out}",
        flush=True,
    )
    return 0


def _refuse_outdir_in_tree(outdir, trees):
    outdir = Path(outdir).resolve()
    for tree in trees:
        tree = Path(tree).resolve()
        if outdir == tree or tree in outdir.parents:
            raise SystemExit(
                f"result_bitdiff: --outdir {outdir} is inside the code tree "
                f"{tree}; run artifacts live outside the repository"
            )
    return outdir


def _git(*args, cwd=THIS_TREE):
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _build_extension(tree):
    subprocess.run(
        [sys.executable, "build_ext.py", "--inplace"], cwd=tree, check=True,
        capture_output=True,
    )


def resolve_tree(spec, outdir, compiled):
    """Return ``(tree_path, label)`` for a directory or a git revision."""
    candidate = Path(spec).expanduser()
    if candidate.is_dir():
        tree = candidate.resolve()
        if not (tree / "cablp").is_dir() or not (tree / "scripts").is_dir():
            raise SystemExit(f"result_bitdiff: {tree} is not a code tree")
        return tree, f"{spec} (directory)"
    sha = _git("rev-parse", "--verify", f"{spec}^{{commit}}")
    tree = Path(outdir) / "trees" / sha[:12]
    if not (tree / "cablp").is_dir():
        tree.mkdir(parents=True, exist_ok=True)
        archive = subprocess.Popen(
            ["git", "archive", sha], cwd=THIS_TREE, stdout=subprocess.PIPE
        )
        subprocess.run(["tar", "-x", "-C", str(tree)], stdin=archive.stdout,
                       check=True)
        if archive.wait() != 0:
            raise SystemExit(f"result_bitdiff: git archive {sha} failed")
        adas_src = THIS_TREE / "cablp" / "atomic" / "data" / "adas"
        masters = sorted(adas_src.glob("*.dat"))
        if not masters:
            raise SystemExit(
                f"result_bitdiff: no OPEN-ADAS masters in {adas_src} to copy "
                "into the exported tree"
            )
        for master in masters:
            shutil.copy2(master, tree / "cablp" / "atomic" / "data" / "adas")
    if compiled and not list((tree / "cablp").rglob("*.so")):
        _build_extension(tree)
    return tree.resolve(), f"{spec} = {sha} (exported)"


def run_leg(tree, entry, out_path, log_path, compiled):
    """Run one leg in a subprocess; return its wall time in seconds."""
    env = dict(os.environ)
    env.pop("CABLP_COMPILED_KERNELS", None)
    if compiled:
        env["CABLP_COMPILED_KERNELS"] = "1"
    env["PYTHONPATH"] = str(tree)
    cmd = [
        sys.executable, str(THIS_FILE), "_leg",
        "--tree", str(tree),
        "--spec", entry.spec,
        "--steps", str(entry.steps),
        "--out", str(out_path),
        "--param-overrides", json.dumps(dict(entry.param_overrides)),
    ]
    if compiled:
        cmd.append("--compiled")
    started = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("$ PYTHONPATH=" + str(tree) + " " + " ".join(cmd) + "\n")
        log.flush()
        code = subprocess.run(cmd, cwd=tree, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - started
    if code != 0:
        raise SystemExit(
            f"result_bitdiff: leg {entry.name} in {tree} failed (exit {code}); "
            f"see {log_path}"
        )
    return wall


def cmd_matrix(args):
    # Refuse BEFORE anything is created: every tree knowable without side
    # effects -- this tree (which a revision is exported from) and any
    # directory given as --base/--head.
    known_trees = [THIS_TREE] + [
        Path(spec).expanduser() for spec in (args.base, args.head)
        if Path(spec).expanduser().is_dir()
    ]
    outdir = _refuse_outdir_in_tree(Path(args.outdir).expanduser(),
                                    known_trees)
    outdir.mkdir(parents=True, exist_ok=True)
    base_tree, base_label = resolve_tree(args.base, outdir, args.compiled)
    head_tree, head_label = resolve_tree(args.head, outdir, args.compiled)
    entries = [MATRIX_BY_NAME[name] for name in args.entries] if args.entries \
        else list(MATRIX)
    print(f"result_bitdiff matrix: base={base_label}", flush=True)
    print(f"result_bitdiff matrix: head={head_label}", flush=True)
    print(f"result_bitdiff matrix: kernels="
          f"{'compiled' if args.compiled else 'pure'}", flush=True)
    cmd_allow_list(args)
    started = time.perf_counter()
    failures = 0
    for entry in entries:
        paths = {}
        for side, tree in (("base", base_tree), ("head", head_tree)):
            out = outdir / f"{entry.name}.{side}.h5"
            wall = run_leg(tree, entry, out, outdir / f"{entry.name}.{side}.log",
                           args.compiled)
            paths[side] = out
            print(f"  leg {entry.name}/{side}: {entry.steps} steps, "
                  f"{wall:.1f} s wall", flush=True)
        report = compare_files(paths["base"], paths["head"])
        print_report(f"{entry.name}: base vs head", report)
        failures += not report.identical
    wall = time.perf_counter() - started
    print(f"result_bitdiff matrix: {len(entries) - failures}/{len(entries)} "
          f"identical, wall {wall:.1f} s")
    return 0 if failures == 0 else 1


# ----------------------------------------------------------------------
# The self-test
# ----------------------------------------------------------------------
def _find_dataset(h5, predicate, exclude=()):
    """Return the first float64 dataset path (sorted) with an element matching."""
    objects, _ = _walk(h5)
    for path in sorted(objects):
        kind, obj = objects[path]
        if kind != "dataset" or path in exclude or obj.shape in (None, ()):
            continue
        if obj.dtype != np.float64 or obj.size == 0:
            continue
        values = obj[()].reshape(-1)
        hits = np.flatnonzero(predicate(values))
        if hits.size:
            return path, int(hits[0])
    raise RuntimeError("no float64 dataset matches the negative-control predicate")


def _expect_exactly(label, report, path_fragment):
    ok = (len(report.findings) == 1
          and report.findings[0].startswith(path_fragment))
    print_report(label, report)
    print(f"  negative control {label}: "
          f"{'DETECTED (exactly one finding)' if ok else 'NOT AS EXPECTED'}")
    return ok


def _negative_controls(result, outdir):
    ok = True
    # (a) one sign of zero.
    copy = outdir / "negctl_signed_zero.h5"
    shutil.copy2(result, copy)
    with h5py.File(copy, "r+") as h5:
        path, index = _find_dataset(
            h5, lambda v: (v == 0.0) & ~np.signbit(v))
        values = h5[path][()]
        flat = values.reshape(-1)
        flat[index] = -0.0
        h5[path][...] = values
    print(f"negative control: +0.0 -> -0.0 at {path} flat index {index}")
    ok &= _expect_exactly("signed-zero", compare_files(result, copy),
                          f"{path} BYTES differ at 1 of")
    # (b) one byte: the lowest-order byte of a nonzero finite float.
    copy = outdir / "negctl_one_byte.h5"
    shutil.copy2(result, copy)
    with h5py.File(copy, "r+") as h5:
        path2, index2 = _find_dataset(
            h5, lambda v: np.isfinite(v) & (v != 0.0), exclude=(path,))
        values = h5[path2][()]
        raw = values.reshape(-1).view(np.uint8)
        raw[index2 * 8] ^= 0x01
        h5[path2][...] = values
    print(f"negative control: low byte flipped at {path2} flat index {index2}")
    ok &= _expect_exactly("one-byte", compare_files(result, copy),
                          f"{path2} BYTES differ at 1 of")
    # (c) one attribute value.
    copy = outdir / "negctl_attribute.h5"
    shutil.copy2(result, copy)
    with h5py.File(copy, "r+") as h5:
        h5.attrs["steps"] = h5.attrs["steps"] + 1
    ok &= _expect_exactly("attribute", compare_files(result, copy), "/@steps")
    # (d) one extra dataset.
    copy = outdir / "negctl_extra_dataset.h5"
    shutil.copy2(result, copy)
    with h5py.File(copy, "r+") as h5:
        h5.create_dataset("negative_control_extra", data=np.zeros(1))
    ok &= _expect_exactly("extra-dataset", compare_files(result, copy),
                          "/negative_control_extra")
    return ok


def _qualified_captures(tree, result, outdir, log_path):
    """Write two qualified captures of one saved result; return their paths.

    The capture writer is the tree's own; the two captures differ only in the
    per-execution identity and the clock stamps the workflow supplies. The
    size cap is lifted: the probe is about the stamps, not the budget.
    """
    shutil.rmtree(Path(outdir) / "qualified", ignore_errors=True)
    code = f"""
import json, sys, time, uuid
from datetime import datetime, timezone
from pathlib import Path
from cablp.solvers._sim1d import load_result_hdf5
from cablp.solvers._sim1d.results.phase3_capture import (
    _source_state_rows, configuration_identity, reserve_run_id,
    write_qualified_capture)
root = Path({str(outdir)!r}) / "qualified"
out = root / "phase3_rhs"
result = load_result_hdf5({str(result)!r})
# A loaded result's RHS rows come back in the file's (alphabetical) order;
# the capture writer requires the packed-state order.
rows = _source_state_rows(result)
result.rhs_terms = {{term: {{row: fields[row] for row in rows}}
                     for term, fields in result.rhs_terms.items()}}
identity = configuration_identity(result.params, result.flags)
paths = []
for _ in range(2):
    run_id = "urn:uuid:" + str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    reserve_run_id(out, run_id, {{"allocated_at": started}})
    result.run_id = run_id
    time.sleep(0.01)
    completed = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    h5_path, _ = write_qualified_capture(
        out, result, run_id=run_id, capture_revision="probe",
        producer_path="scripts/gates/result_bitdiff.py",
        started_at=started, completed_at=completed,
        configuration_identity_sha256=identity,
        recipe_identity="result-bitdiff-allow-list-probe",
        run_controls={{}}, invocation=["result_bitdiff.py", "--self-test"],
        producer_blobs={{}}, environment_lock={{}}, repository_root=root,
        maximum_bytes=1 << 40)
    paths.append(str(h5_path))
print(json.dumps(paths))
"""
    env = dict(os.environ)
    env.pop("CABLP_COMPILED_KERNELS", None)
    env["PYTHONPATH"] = str(tree)
    with open(log_path, "w") as log:
        proc = subprocess.run([sys.executable, "-c", code], cwd=tree, env=env,
                              stdout=subprocess.PIPE, stderr=log, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"result_bitdiff: qualified-capture probe failed; "
                         f"see {log_path}")
    return [Path(p) for p in json.loads(proc.stdout.strip().splitlines()[-1])]


def _evidence(label, path_a, path_b, expected_paths):
    """Compare with NO allow-list; the differences must be exactly expected."""
    report = compare_files(path_a, path_b, allow_list={})
    differing = sorted(f.split(" ", 1)[0].rstrip(":") for f in report.findings)
    ok = differing == sorted(expected_paths)
    print_report(f"{label} (allow-list disabled)", report)
    print(f"  allow-list evidence {label}: differs exactly at "
          f"{differing} -- {'CONFIRMED' if ok else 'NOT AS EXPECTED'}")
    return ok, differing


def cmd_self_test(args):
    tree = Path(args.tree).expanduser().resolve() if args.tree else THIS_TREE
    outdir = _refuse_outdir_in_tree(Path(args.outdir).expanduser(),
                                    (tree, THIS_TREE))
    outdir.mkdir(parents=True, exist_ok=True)
    entry = MATRIX_BY_NAME[args.entry]
    started = time.perf_counter()
    ok = True

    print(f"result_bitdiff self-test: tree={tree} entry={entry.name} "
          f"({entry.spec}, {entry.steps} steps), kernels=pure", flush=True)
    cmd_allow_list(args)
    # (1) the same entry twice on the same tree.
    runs = []
    for tag in ("run1", "run2"):
        out = outdir / f"selftest_{entry.name}.{tag}.h5"
        wall = run_leg(tree, entry, out,
                       outdir / f"selftest_{entry.name}.{tag}.log", False)
        print(f"  leg {entry.name}/{tag}: {wall:.1f} s wall", flush=True)
        runs.append(out)
    report = compare_files(*runs)
    print_report(f"{entry.name}: rerun on the same tree", report)
    ok &= report.identical
    strict = compare_files(*runs, allow_list={})
    print(f"  rerun with the allow-list disabled: "
          f"{'IDENTICAL' if strict.identical else 'DIFFERENT'} "
          f"({len(strict.findings)} finding(s))")

    # (2) negative controls.
    ok &= _negative_controls(runs[0], outdir)

    # (3) allow-list evidence: each entry shown to differ between two
    # identically produced files, and nothing else to differ.
    covered = set()
    probe = []
    for tag in ("run1", "run2"):
        out = outdir / f"evidence_{IGNITION_ABORT_PROBE.name}.{tag}.h5"
        run_leg(tree, IGNITION_ABORT_PROBE, out,
                outdir / f"evidence_{IGNITION_ABORT_PROBE.name}.{tag}.log",
                False)
        probe.append(out)
    good, differing = _evidence("ignition-abort probe run twice", *probe,
                                ["/ignition_abort@wall_clock_s"])
    ok &= good
    covered.update(differing)
    captures = _qualified_captures(tree, runs[0], outdir,
                                   outdir / "evidence_qualified_capture.log")
    good, differing = _evidence(
        "two qualified captures of one result", *captures,
        ["/@run_id", "/@started_at", "/@completed_at"])
    ok &= good
    covered.update(differing)
    uncovered = sorted(set(ALLOW_LIST) - covered)
    print(f"  allow-list entries without evidence: {uncovered or 'none'}")
    ok &= not uncovered

    wall = time.perf_counter() - started
    print(f"result_bitdiff self-test: {'PASS' if ok else 'FAIL'}, "
          f"wall {wall:.1f} s")
    return 0 if ok else 1


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Full-result bit-diff gate for LAPDSim1D HDF5 results."
    )
    parser.add_argument("--self-test", action="store_true",
                        help="Run the self-test (needs --outdir).")
    parser.add_argument("--outdir", help="Artifact directory outside the repo.")
    parser.add_argument("--tree", help="Self-test tree (default: this tree).")
    parser.add_argument("--entry", default=MATRIX[0].name,
                        choices=sorted(MATRIX_BY_NAME),
                        help="Self-test matrix entry.")
    sub = parser.add_subparsers(dest="mode")

    p = sub.add_parser("compare", help="Compare two result files.")
    p.add_argument("a")
    p.add_argument("b")

    sub.add_parser("allow-list", help="Print the allow-list.")

    p = sub.add_parser("matrix", help="Run the matrix under two trees.")
    p.add_argument("--base", required=True, help="Git revision or tree path.")
    p.add_argument("--head", required=True, help="Git revision or tree path.")
    p.add_argument("--outdir", required=True,
                   help="Artifact directory outside the repo.")
    p.add_argument("--compiled", action="store_true",
                   help="Run both legs on the compiled kernels.")
    p.add_argument("--entries", nargs="+", choices=sorted(MATRIX_BY_NAME),
                   help="Run only these entries (default: all).")

    p = sub.add_parser("_leg", help=argparse.SUPPRESS)
    p.add_argument("--tree", required=True)
    p.add_argument("--spec", required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--param-overrides", default="{}")
    p.add_argument("--compiled", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.self_test:
        if not args.outdir:
            raise SystemExit("result_bitdiff: --self-test needs --outdir")
        return cmd_self_test(args)
    if args.mode == "compare":
        return cmd_compare(args)
    if args.mode == "allow-list":
        return cmd_allow_list(args)
    if args.mode == "matrix":
        return cmd_matrix(args)
    if args.mode == "_leg":
        return _leg_main(args)
    raise SystemExit("result_bitdiff: name a mode (compare, matrix, "
                     "allow-list) or --self-test")


if __name__ == "__main__":
    raise SystemExit(main())
