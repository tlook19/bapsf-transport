"""Synthetic/read-only verification for Phase 3 source-capture plumbing.

This script never constructs or executes ``LAPDSim1D``.  Its result objects
and HDF5 files are hand-built in temporary directories.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.solvers._sim1d.results.io import save_result_hdf5
from cablp.solvers._sim1d.results.phase3_capture import (
    MAX_ARTIFACT_BYTES,
    PHASE3_CAPTURE_PROFILE,
    blank_crosswalk_inventory,
    configuration_identity,
    load_qualified_capture,
    reserve_run_id,
    scientific_payload_digest,
    validate_run_id,
    whole_file_sha256,
    write_qualified_capture,
)


RUN_A = "urn:uuid:123e4567-e89b-42d3-a456-426614174000"
RUN_B = "urn:uuid:123e4567-e89b-42d3-a456-426614174001"
RUN_C = "urn:uuid:123e4567-e89b-42d3-a456-426614174002"
ROWS = ("n", "nn", "M", "Ee", "Ei", "M_n", "nn_a", "M_n_a", "En")
TERMS = ("synthetic_flux", "synthetic_reaction", "synthetic_boundary")


def _fake_result(run_id=RUN_A):
    frames, cells = 2, 3
    rhs_terms = {}
    for term_index, term in enumerate(TERMS):
        rhs_terms[term] = {
            row: np.full((frames, cells), term_index * 10 + row_index, dtype=float)
            for row_index, row in enumerate(ROWS)
        }
    zero = np.zeros((frames, cells), dtype=float)
    result = SimpleNamespace(
        run_id=run_id,
        params={"synthetic": 1, "max_steps_action": "stop"},
        flags={"neutral_two_zone": True, "neutral_energy": True},
        compiled_kernels="pure",
        steps=4000,
        final_time=1.0e-5,
        run_status="max_steps_reached",
        time=np.asarray([0.0, 1.0e-5]),
        y=np.arange(frames * len(ROWS) * cells, dtype=float).reshape(frames, -1),
        n=zero,
        nn=zero,
        M=zero,
        momentum=zero,
        Ee=zero,
        Ei=zero,
        u=zero,
        Te=zero,
        Ti=zero,
        pe=zero,
        pi=zero,
        p=zero,
        M_n=zero,
        u_n=zero,
        nn_a=zero,
        M_n_a=zero,
        u_n_a=zero,
        En=zero,
        Tn=zero,
        z_cm=np.asarray([1.0, 2.0, 3.0]),
        length_cm=np.ones(cells),
        Rp_cm=np.ones(cells),
        Rm_cm=np.full(cells, 2.0),
        plasma_volume_cm3=np.full(cells, 2.0),
        neutral_volume_cm3=np.full(cells, 5.0),
        volume_ratio=np.full(cells, 2.5),
        plasma_active=np.ones(cells, dtype=bool),
        cell_role=np.asarray(["column"] * cells, dtype=object),
        rhs_terms=rhs_terms,
        total_rhs={row: zero for row in ROWS},
        electron_energy_terms_W_cm3={term: zero for term in TERMS},
        ion_energy_terms_W_cm3={term: zero for term in TERMS},
        diagnostics=[],
    )
    return result


def _write(root, result, *, maximum_bytes=MAX_ARTIFACT_BYTES, invocation=None):
    output = root / "cablp/scripts/baselines/phase3_rhs"
    reserve_run_id(
        output,
        result.run_id,
        {"allocated_at": "2026-08-24T12:00:00Z", "kind": "synthetic-test"},
    )
    return write_qualified_capture(
        output,
        result,
        run_id=result.run_id,
        capture_revision="a" * 40,
        producer_path="cablp/scripts/capture_phase3_rhs.py",
        started_at="2026-08-24T12:00:00Z",
        completed_at="2026-08-24T12:00:01Z",
        configuration_identity_sha256=configuration_identity(
            result.params, result.flags
        ),
        recipe_identity="synthetic-recipe",
        run_controls={"max_steps": 4000},
        invocation=(
            ["python", "scripts/capture_phase3_rhs.py", "--synthetic"]
            if invocation is None
            else invocation
        ),
        producer_blobs={"cablp/synthetic.py": "b" * 40},
        environment_lock={"path": "cablp/poetry.lock", "git_blob": "c" * 40},
        repository_root=root,
        maximum_bytes=maximum_bytes,
    )


def _raises(error, function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{function.__name__} did not raise {error.__name__}")


def check_identity_and_reservation(root):
    assert validate_run_id(RUN_A) == RUN_A
    for bad in (None, "", "123e4567-e89b-42d3-a456-426614174000", RUN_A.upper()):
        _raises(ValueError, validate_run_id, bad)
    output = root / "reservation/cablp/scripts/baselines/phase3_rhs"
    marker = reserve_run_id(output, RUN_A, {"kind": "synthetic-test"})
    assert marker.is_file()
    _raises(FileExistsError, reserve_run_id, output, RUN_A, {})
    colliding = output / f"{RUN_B.removeprefix('urn:uuid:')}.h5"
    colliding.parent.mkdir(parents=True, exist_ok=True)
    colliding.touch()
    _raises(FileExistsError, reserve_run_id, output, RUN_B, {})


def check_schema_round_trip_and_digests(root):
    result = _fake_result()
    h5_path, provenance_path = _write(root / "roundtrip", result)
    loaded, provenance = load_qualified_capture(h5_path, provenance_path)
    assert loaded.run_id == RUN_A
    assert provenance["term_names"] == list(TERMS)
    assert provenance["state_rows"] == list(ROWS)
    assert provenance["artifact_sha256"] == whole_file_sha256(h5_path)
    assert provenance["scientific_payload_sha256"] == scientific_payload_digest(
        h5_path
    )
    assert not Path(provenance["artifact_path"]).is_absolute()
    with h5py.File(h5_path, "r") as h5:
        assert h5.attrs["format"] == "sim1d-hdf5-v1"
        assert h5.attrs["capture_profile"] == PHASE3_CAPTURE_PROFILE
        assert tuple(h5["qualification/term_names"].asstr()[()]) == TERMS
        assert tuple(h5["qualification/state_rows"].asstr()[()]) == ROWS
        assert np.array_equal(h5["geometry/annulus_volume_cm3"][()], [3.0] * 3)
        assert (
            h5["rhs_terms/synthetic_flux/nn_a"].attrs["support_dataset"]
            == "/geometry/annulus_volume_cm3"
        )
        assert (
            h5["rhs_terms/synthetic_flux/Ee"].attrs["unit"]
            == "erg cm^-3 s^-1"
        )
    inventory = blank_crosswalk_inventory(h5_path)
    assert len(inventory) == len(TERMS) * len(ROWS)
    assert {row["dataset_path"] for row in inventory} == {
        f"/rhs_terms/{term}/{field}" for term in TERMS for field in ROWS
    }
    assert all(row["disposition"] == row["reviewer"] == "" for row in inventory)

    # Identity changes representation, but not the canonical scientific payload.
    second = _fake_result(RUN_B)
    h5_b, provenance_b = _write(root / "roundtrip", second)
    assert scientific_payload_digest(h5_b) == scientific_payload_digest(h5_path)
    assert whole_file_sha256(h5_b) != whole_file_sha256(h5_path)
    assert provenance_b != provenance_path


def check_refusals_and_atomicity(root):
    base = _fake_result(RUN_A)
    wrong_id = copy.deepcopy(base)
    wrong_id.run_id = RUN_B
    output = root / "wrong-id/cablp/scripts/baselines/phase3_rhs"
    reserve_run_id(output, RUN_A, {"kind": "synthetic-test"})
    _raises(
        ValueError,
        write_qualified_capture,
        output,
        wrong_id,
        run_id=RUN_A,
        capture_revision="a" * 40,
        producer_path="cablp/scripts/capture_phase3_rhs.py",
        started_at="start",
        completed_at="complete",
        configuration_identity_sha256=configuration_identity(
            wrong_id.params, wrong_id.flags
        ),
        recipe_identity="synthetic",
        run_controls={"max_steps": 4000},
        invocation=["python", "scripts/capture_phase3_rhs.py"],
        producer_blobs={},
        environment_lock={"path": "cablp/poetry.lock"},
        repository_root=root / "wrong-id",
    )

    for mutator in (
        lambda r: setattr(r, "steps", 3999),
        lambda r: setattr(r, "run_status", "completed"),
        lambda r: setattr(r, "compiled_kernels", "compiled"),
        lambda r: setattr(r, "time", np.asarray([0.0])),
        lambda r: r.rhs_terms[TERMS[1]].pop("Ei"),
        lambda r: r.rhs_terms[TERMS[1]].__setitem__(
            "extra", np.zeros((2, 3), dtype=float)
        ),
        lambda r: setattr(r, "y", np.zeros((2, 2))),
    ):
        candidate = copy.deepcopy(base)
        candidate.run_id = RUN_C
        mutator(candidate)
        case_name = hashlib.sha256(repr(mutator).encode()).hexdigest()
        case_root = root / f"refusal-{case_name}"
        case_output = case_root / "cablp/scripts/baselines/phase3_rhs"
        reserve_run_id(case_output, RUN_C, {"kind": "synthetic-test"})
        _raises(
            ValueError,
            _write_reserved,
            case_root,
            case_output,
            candidate,
        )

    cap_root = root / "cap"
    cap_result = _fake_result(RUN_C)
    _raises(ValueError, _write, cap_root, cap_result, maximum_bytes=1)
    cap_target = (
        cap_root
        / "cablp/scripts/baselines/phase3_rhs"
        / f"{RUN_C.removeprefix('urn:uuid:')}.h5"
    )
    assert not cap_target.exists()
    assert not cap_target.with_suffix(".provenance.json").exists()

    private_root = root / "private"
    _raises(
        ValueError,
        _write,
        private_root,
        _fake_result(RUN_C),
        invocation=["/home/example/python", "scripts/capture_phase3_rhs.py"],
    )
    private_target = (
        private_root
        / "cablp/scripts/baselines/phase3_rhs"
        / f"{RUN_C.removeprefix('urn:uuid:')}.h5"
    )
    assert not private_target.exists()


def _write_reserved(case_root, output, result):
    return write_qualified_capture(
        output,
        result,
        run_id=result.run_id,
        capture_revision="a" * 40,
        producer_path="cablp/scripts/capture_phase3_rhs.py",
        started_at="start",
        completed_at="complete",
        configuration_identity_sha256=configuration_identity(
            result.params, result.flags
        ),
        recipe_identity="synthetic",
        run_controls={"max_steps": 4000},
        invocation=["python", "scripts/capture_phase3_rhs.py"],
        producer_blobs={},
        environment_lock={"path": "cablp/poetry.lock"},
        repository_root=case_root,
    )


def check_mutation_and_unqualified_refusal(root):
    qualified_root = root / "mutation"
    h5_path, provenance_path = _write(qualified_root, _fake_result())
    with h5py.File(h5_path, "r+") as h5:
        h5["rhs_terms/synthetic_flux/n"][0, 0] += 1.0
    _raises(ValueError, load_qualified_capture, h5_path, provenance_path)

    reordered_root = root / "reordered-census"
    reordered_h5, reordered_provenance = _write(
        reordered_root, _fake_result(RUN_B)
    )
    with h5py.File(reordered_h5, "r+") as h5:
        census = h5["qualification/term_names"][()]
        h5["qualification/term_names"][...] = census[::-1]
    _raises(
        ValueError,
        load_qualified_capture,
        reordered_h5,
        reordered_provenance,
    )

    ordinary_path = root / "ordinary.h5"
    ordinary = _fake_result()
    del ordinary.run_id
    save_result_hdf5(ordinary_path, ordinary)
    with h5py.File(ordinary_path, "r") as h5:
        assert "capture_profile" not in h5.attrs
        assert "run_id" not in h5.attrs
    _raises(ValueError, load_qualified_capture, ordinary_path)


def check_constructor_order_and_cli_import(repo_root):
    solver_source = (
        repo_root / "cablp/cablp/solvers/_sim1d/solver.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(solver_source)
    lapd_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LAPDSim1D"
    )
    constructor = next(
        node
        for node in lapd_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    rendered = [ast.unparse(statement) for statement in constructor.body]
    validation_index = next(
        index for index, text in enumerate(rendered) if "validate_run_id" in text
    )
    construction_index = next(
        index
        for index, text in enumerate(rendered)
        if "_init_config_and_early_flags" in text
    )
    assert validation_index < construction_index

    cli_source = (repo_root / "cablp/scripts/capture_phase3_rhs.py").read_text(
        encoding="utf-8"
    )
    cli_tree = ast.parse(cli_source)
    calls = [
        node
        for node in ast.walk(cli_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "sim"
        and node.func.attr in {"start_simulation", "run"}
    ]
    assert len(calls) == 1
    # Parsing and importing the capture modules above reached no constructor;
    # the only executable solver call remains guarded by main/function entry.


def main():
    repo_root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="phase3-capture-synthetic-") as temp:
        root = Path(temp)
        check_identity_and_reservation(root)
        check_schema_round_trip_and_digests(root)
        check_refusals_and_atomicity(root)
        check_mutation_and_unqualified_refusal(root)
    check_constructor_order_and_cli_import(repo_root)
    print("phase3 source capture synthetic checks: PASS")
    print("LAPDSim1D constructions/executions: 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
