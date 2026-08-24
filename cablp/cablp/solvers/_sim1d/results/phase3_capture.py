"""Qualified Phase 3 run/RHS capture without implicit execution identity.

The ordinary ``sim1d-hdf5-v1`` path remains the general result format.  This
module adds the stricter ``sim1d-phase3-rhs-qualification-v1`` profile used by
the separately authorized qualification workflow.  It deliberately contains
no UUID generator and no solver construction.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import uuid
from pathlib import Path

import h5py
import numpy as np

from ..core.state import (
    NEUTRAL_ANNULUS_MOMENTUM_NAME,
    NEUTRAL_ANNULUS_NAME,
    NEUTRAL_ENERGY_NAME,
    NEUTRAL_MOMENTUM_NAME,
    STATE_NAMES_1D,
)
from .io import RESULT_VERSION, load_result_hdf5, save_result_hdf5


PHASE3_CAPTURE_PROFILE = "sim1d-phase3-rhs-qualification-v1"
MAX_ARTIFACT_BYTES = 50_331_648

_ROW_UNITS = {
    "n": "cm^-3 s^-1",
    "nn": "cm^-3 s^-1",
    "nn_a": "cm^-3 s^-1",
    "M": "g cm^-2 s^-2",
    "M_n": "g cm^-2 s^-2",
    "M_n_a": "g cm^-2 s^-2",
    "Ee": "erg cm^-3 s^-1",
    "Ei": "erg cm^-3 s^-1",
    "En": "erg cm^-3 s^-1",
}
_PLASMA_ROWS = {"n", "M", "Ee", "Ei"}
_ANNULUS_ROWS = {"nn_a", "M_n_a"}
_QUALIFICATION_REQUIRED_ATTRS = (
    "capture_profile",
    "run_id",
    "capture_revision",
    "producer_path",
    "started_at",
    "completed_at",
    "kernel_provenance",
    "configuration_identity",
    "recipe_identity",
    "invocation_json",
    "scientific_payload_sha256",
)


def validate_run_id(run_id):
    """Return a canonical RFC 9562 UUID URN or raise ``ValueError``.

    No allocation occurs here.  In particular, malformed or omitted identity
    is never replaced by a save-time generated value.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id is required and must be an RFC 9562 UUID URN")
    if not run_id.startswith("urn:uuid:"):
        raise ValueError("run_id must use the canonical 'urn:uuid:' prefix")
    text = run_id.removeprefix("urn:uuid:")
    try:
        value = uuid.UUID(text)
    except (AttributeError, ValueError) as exc:
        raise ValueError("run_id must contain a valid UUID") from exc
    if value.variant != uuid.RFC_4122 or value.version not in range(1, 9):
        raise ValueError("run_id must carry RFC 9562 variant and version bits")
    canonical = f"urn:uuid:{value}"
    if run_id != canonical:
        raise ValueError(f"run_id is not canonical; expected {canonical!r}")
    return canonical


def artifact_paths(output_directory, run_id):
    """Return the fixed HDF5/provenance/allocation paths for ``run_id``."""
    run_id = validate_run_id(run_id)
    output_directory = Path(output_directory)
    stem = run_id.removeprefix("urn:uuid:")
    return (
        output_directory / f"{stem}.h5",
        output_directory / f"{stem}.provenance.json",
        output_directory / ".allocations" / f"{stem}.json",
    )


def reserve_run_id(output_directory, run_id, allocation_record):
    """Persist an execution allocation exactly once, before construction.

    The exclusive create is the reuse lock.  It is intentionally retained if
    later construction or execution fails: attempted execution IDs are never
    recycled.
    """
    run_id = validate_run_id(run_id)
    h5_path, provenance_path, allocation_path = artifact_paths(
        output_directory, run_id
    )
    for collision in (h5_path, provenance_path, allocation_path):
        if collision.exists():
            raise FileExistsError(
                f"run_id is reused or output-colliding: {collision.name}"
            )
    record = dict(allocation_record)
    record["run_id"] = run_id
    _reject_private_or_absolute_values(record)
    allocation_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_bytes(record) + b"\n"
    try:
        fd = os.open(allocation_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"run_id has already been allocated: {run_id}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return allocation_path


def configuration_identity(params, flags):
    """Return the canonical resolved configuration SHA-256 identity."""
    # This deliberately matches golden_digest_gate.config_identity byte for
    # byte, including json.dumps' default item separators.  The selected
    # configuration identity is already committed in that representation.
    payload = json.dumps(
        {"params": params, "flags": flags},
        sort_keys=True,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scientific_payload_digest(path):
    """Digest ordered scientific datasets as path/dtype/shape/array bytes."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        dataset_paths = _scientific_dataset_paths(h5)
        digest = hashlib.sha256()
        for dataset_path in dataset_paths:
            dataset = h5[dataset_path]
            values = np.asarray(dataset[()])
            if values.dtype.kind in {"O", "U"}:
                raise ValueError(
                    f"scientific payload dataset is not fixed-width: "
                    f"{dataset_path}"
                )
            dtype = values.dtype.str.encode("ascii")
            shape = tuple(int(value) for value in values.shape)
            array_bytes = np.ascontiguousarray(values).tobytes(order="C")
            for part in (
                dataset_path.encode("utf-8"),
                dtype,
                json.dumps(shape, separators=(",", ":")).encode("ascii"),
                array_bytes,
            ):
                digest.update(struct.pack(">Q", len(part)))
                digest.update(part)
        return digest.hexdigest()


def whole_file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_qualified_capture(
    output_directory,
    result,
    *,
    run_id,
    capture_revision,
    producer_path,
    started_at,
    completed_at,
    configuration_identity_sha256,
    recipe_identity,
    run_controls,
    invocation,
    producer_blobs,
    environment_lock,
    repository_root,
    maximum_bytes=MAX_ARTIFACT_BYTES,
):
    """Validate, write, close, digest, and atomically install one capture."""
    run_id = validate_run_id(run_id)
    output_directory = Path(output_directory)
    repository_root = Path(repository_root).resolve()
    h5_path, provenance_path, allocation_path = artifact_paths(
        output_directory, run_id
    )
    if not allocation_path.is_file():
        raise ValueError("run_id must be persistently allocated before capture")
    for target in (h5_path, provenance_path):
        if target.exists():
            raise FileExistsError(f"capture target already exists: {target.name}")
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_rel = _repository_relative(h5_path, repository_root)
    provenance_rel = _repository_relative(provenance_path, repository_root)
    allocation_rel = _repository_relative(allocation_path, repository_root)
    producer_path = _validate_relative_path(producer_path)
    environment_lock = dict(environment_lock)
    producer_blobs = dict(producer_blobs)
    invocation = list(invocation)
    run_controls = dict(run_controls)
    _reject_private_or_absolute_values(
        {
            "producer_path": producer_path,
            "invocation": invocation,
            "producer_blobs": producer_blobs,
            "environment_lock": environment_lock,
        }
    )

    census = _validate_result(
        result,
        run_id=run_id,
        configuration_identity_sha256=configuration_identity_sha256,
    )
    temp_directory = output_directory.parent
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{h5_path.stem}.", suffix=".tmp", dir=temp_directory
    )
    os.close(fd)
    temp_path = Path(temp_name)
    temp_provenance = temp_path.with_suffix(".provenance.tmp")
    try:
        save_result_hdf5(temp_path, result)
        _validate_serialized_rhs_surface(temp_path, result, census)
        _qualify_hdf5(
            temp_path,
            result=result,
            census=census,
            run_id=run_id,
            capture_revision=capture_revision,
            producer_path=producer_path,
            started_at=started_at,
            completed_at=completed_at,
            configuration_identity_sha256=configuration_identity_sha256,
            recipe_identity=recipe_identity,
            run_controls=run_controls,
            invocation=invocation,
        )
        scientific_sha = scientific_payload_digest(temp_path)
        with h5py.File(temp_path, "r+") as h5:
            h5.attrs["scientific_payload_sha256"] = scientific_sha
            h5.flush()
        _validate_qualified_hdf5(temp_path)
        byte_size = temp_path.stat().st_size
        if byte_size > int(maximum_bytes):
            raise ValueError(
                f"closed capture is {byte_size} bytes, exceeding hard cap "
                f"{int(maximum_bytes)}; target was not installed"
            )
        file_sha = whole_file_sha256(temp_path)
        provenance = {
            "capture_profile": PHASE3_CAPTURE_PROFILE,
            "run_id": run_id,
            "capture_revision": str(capture_revision),
            "artifact_path": artifact_rel,
            "provenance_path": provenance_rel,
            "allocation_path": allocation_rel,
            "artifact_sha256": file_sha,
            "artifact_bytes": byte_size,
            "scientific_payload_sha256": scientific_sha,
            "configuration_identity": configuration_identity_sha256,
            "recipe_identity": str(recipe_identity),
            "kernel_provenance": "pure",
            "started_at": str(started_at),
            "completed_at": str(completed_at),
            "producer_path": producer_path,
            "producer_blobs": producer_blobs,
            "environment_lock": environment_lock,
            "invocation": invocation,
            "run_controls": run_controls,
            "term_names": list(census["term_names"]),
            "state_rows": list(census["state_rows"]),
        }
        _reject_private_or_absolute_values(provenance)
        with temp_provenance.open("xb") as stream:
            stream.write(_canonical_json_bytes(provenance) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Hard links provide atomic, no-overwrite installation on the target
        # filesystem.  The allocation lock and preflight collision checks make
        # the two-file finalize single-writer; HDF5 is installed last so a
        # visible artifact always has its adjacent provenance record.
        os.link(temp_provenance, provenance_path)
        try:
            os.link(temp_path, h5_path)
        except Exception:
            provenance_path.unlink()
            raise
        return h5_path, provenance_path
    finally:
        temp_path.unlink(missing_ok=True)
        temp_provenance.unlink(missing_ok=True)


def load_qualified_capture(path, provenance_path=None):
    """Validate a qualified file and adjacent provenance, then load it."""
    path = Path(path)
    if provenance_path is None:
        provenance_path = path.with_suffix(".provenance.json")
    provenance_path = Path(provenance_path)
    _validate_qualified_hdf5(path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    _reject_private_or_absolute_values(provenance)
    if provenance.get("artifact_sha256") != whole_file_sha256(path):
        raise ValueError("whole-file SHA-256 mismatch")
    scientific_sha = scientific_payload_digest(path)
    if provenance.get("scientific_payload_sha256") != scientific_sha:
        raise ValueError("scientific-payload SHA-256 mismatch")
    with h5py.File(path, "r") as h5:
        if _text(h5.attrs["run_id"]) != provenance.get("run_id"):
            raise ValueError("HDF5/provenance run_id mismatch")
        if _text(h5.attrs["scientific_payload_sha256"]) != scientific_sha:
            raise ValueError("embedded scientific-payload digest mismatch")
        comparisons = {
            "capture_profile": _text(h5.attrs["capture_profile"]),
            "capture_revision": _text(h5.attrs["capture_revision"]),
            "configuration_identity": _text(
                h5.attrs["configuration_identity"]
            ),
            "recipe_identity": _text(h5.attrs["recipe_identity"]),
            "kernel_provenance": _text(h5.attrs["kernel_provenance"]),
            "started_at": _text(h5.attrs["started_at"]),
            "completed_at": _text(h5.attrs["completed_at"]),
            "producer_path": _text(h5.attrs["producer_path"]),
            "invocation": json.loads(_text(h5.attrs["invocation_json"])),
            "run_controls": json.loads(_text(h5.attrs["run_controls_json"])),
            "term_names": list(
                _read_text_dataset(h5["qualification/term_names"])
            ),
            "state_rows": list(
                _read_text_dataset(h5["qualification/state_rows"])
            ),
        }
        differing = [
            key for key, value in comparisons.items() if provenance.get(key) != value
        ]
        if differing:
            raise ValueError(
                f"HDF5/provenance metadata mismatch: {sorted(differing)}"
            )
    result = load_result_hdf5(path)
    return result, provenance


def blank_crosswalk_inventory(path):
    """Return one disposition-blank record for every captured term/row path."""
    _validate_qualified_hdf5(path)
    with h5py.File(path, "r") as h5:
        run_id = _text(h5.attrs["run_id"])
        capture_revision = _text(h5.attrs["capture_revision"])
        producer_path = _text(h5.attrs["producer_path"])
        term_names = _read_text_dataset(h5["qualification/term_names"])
        state_rows = _read_text_dataset(h5["qualification/state_rows"])
        units = _read_text_dataset(h5["qualification/row_units"])
        supports = _read_text_dataset(h5["qualification/row_support_paths"])
        row_metadata = dict(zip(state_rows, zip(units, supports), strict=True))
        return [
            {
                "run_id": run_id,
                "capture_profile": PHASE3_CAPTURE_PROFILE,
                "capture_revision": capture_revision,
                "producer_path": producer_path,
                "producer_anchor": (
                    "LAPDSim1D._trajectory_result -> "
                    f"result.rhs_terms[{term!r}][{row!r}]"
                ),
                "source_term": term,
                "source_row": row,
                "dataset_path": f"/rhs_terms/{term}/{row}",
                "unit": row_metadata[row][0],
                "support_path": row_metadata[row][1],
                "dataset_shape": list(h5[f"rhs_terms/{term}/{row}"].shape),
                "dataset_dtype": h5[f"rhs_terms/{term}/{row}"].dtype.str,
                "proposed_equation_term": "",
                "disposition": "",
                "reviewer": "",
                "review_time": "",
                "rationale": "",
                "finding": "",
            }
            for term in term_names
            for row in state_rows
        ]


def _validate_result(result, *, run_id, configuration_identity_sha256):
    if getattr(result, "run_id", None) != run_id:
        raise ValueError("result run_id does not match supplied run_id")
    if getattr(result, "compiled_kernels", None) != "pure":
        raise ValueError("qualification profile requires kernel provenance 'pure'")
    if int(getattr(result, "steps", -1)) != 4000:
        raise ValueError("qualification result must contain exactly 4000 steps")
    if getattr(result, "run_status", None) != "max_steps_reached":
        raise ValueError("qualification result must end with max_steps_reached")
    time = np.asarray(getattr(result, "time", []))
    if time.ndim != 1 or time.size < 2 or float(time[0]) != 0.0:
        raise ValueError("qualification result requires t=0 and at least two frames")
    params = getattr(result, "params", None)
    flags = getattr(result, "flags", None)
    if params is None or flags is None:
        raise ValueError("qualification result lacks resolved params/flags")
    actual_config = configuration_identity(params, flags)
    if actual_config != configuration_identity_sha256:
        raise ValueError(
            f"configuration identity mismatch: {actual_config} != "
            f"{configuration_identity_sha256}"
        )
    rhs_terms = getattr(result, "rhs_terms", None)
    if not isinstance(rhs_terms, dict) or not rhs_terms:
        raise ValueError("qualification result has no rhs_terms surface")
    term_names = tuple(rhs_terms)
    state_rows = _source_state_rows(result)
    cells = len(np.asarray(result.z_cm))
    expected_shape = (time.size, cells)
    for term_name, fields in rhs_terms.items():
        if "/" in term_name or not term_name:
            raise ValueError(f"invalid RHS term name {term_name!r}")
        if tuple(fields) != state_rows:
            raise ValueError(
                f"nonuniform or reordered RHS rows for {term_name!r}: "
                f"{tuple(fields)!r} != {state_rows!r}"
            )
        for row, values in fields.items():
            if "/" in row or np.asarray(values).shape != expected_shape:
                raise ValueError(
                    f"RHS dataset {term_name}/{row} has shape "
                    f"{np.asarray(values).shape}, expected {expected_shape}"
                )
    y = np.asarray(getattr(result, "y", []))
    if y.shape != (time.size, len(state_rows) * cells):
        raise ValueError(
            f"packed state shape {y.shape} disagrees with "
            f"{len(state_rows)} rows x {cells} cells"
        )
    _validate_geometry(result, state_rows, cells)
    return {"term_names": term_names, "state_rows": state_rows}


def _source_state_rows(result):
    """Derive exact packed row order from ``core.state`` and live config.

    The RHS mappings are checked against this order; they are never allowed to
    establish their own schema merely by agreeing with each other.
    """
    params = dict(result.params)
    flags = dict(result.flags)
    present = {
        name for name in _optional_state_names() if hasattr(result, name)
    }
    return _canonical_state_rows(params, flags, present)


def _canonical_state_rows(params, flags, present_optional):
    expectations = (
        (NEUTRAL_MOMENTUM_NAME, bool(flags.get("neutral_momentum", False))),
        (NEUTRAL_ANNULUS_NAME, bool(flags.get("neutral_two_zone", False))),
        (
            NEUTRAL_ANNULUS_MOMENTUM_NAME,
            params.get("neutral_momentum_radial", "uniform")
            == "kinetic_two_moment",
        ),
        (NEUTRAL_ENERGY_NAME, bool(flags.get("neutral_energy", False))),
    )
    present_optional = set(present_optional)
    rows = list(STATE_NAMES_1D)
    for name, configured in expectations:
        present = name in present_optional
        if present != configured:
            raise ValueError(
                f"source state row {name!r} presence {present} disagrees "
                f"with resolved configuration expectation {configured}"
            )
        if configured:
            rows.append(name)
    unknown = present_optional - set(_optional_state_names())
    if unknown:
        raise ValueError(f"unknown optional source state rows: {sorted(unknown)}")
    return tuple(rows)


def _optional_state_names():
    return (
        NEUTRAL_MOMENTUM_NAME,
        NEUTRAL_ANNULUS_NAME,
        NEUTRAL_ANNULUS_MOMENTUM_NAME,
        NEUTRAL_ENERGY_NAME,
    )


def _validate_geometry(result, state_rows, cells):
    for name in (
        "z_cm",
        "length_cm",
        "Rp_cm",
        "Rm_cm",
        "plasma_volume_cm3",
        "neutral_volume_cm3",
        "volume_ratio",
        "plasma_active",
        "cell_role",
    ):
        values = np.asarray(getattr(result, name, []))
        if values.shape != (cells,):
            raise ValueError(
                f"geometry {name} has shape {values.shape}, expected {(cells,)}"
            )
    if set(state_rows) & _ANNULUS_ROWS:
        annulus = np.asarray(result.neutral_volume_cm3) - np.asarray(
            result.plasma_volume_cm3
        )
        if np.any(annulus < 0.0):
            raise ValueError("annulus control volume is negative")


def _validate_serialized_rhs_surface(path, result, census):
    """Compare the source result census and arrays to the just-written HDF5."""
    with h5py.File(path, "r") as h5:
        term_names = census["term_names"]
        state_rows = census["state_rows"]
        actual_terms = set(h5["rhs_terms"].keys())
        if actual_terms != set(term_names) or len(actual_terms) != len(term_names):
            raise ValueError("written HDF5 term surface differs from source result")
        for term in term_names:
            actual_rows = set(h5[f"rhs_terms/{term}"].keys())
            if actual_rows != set(state_rows) or len(actual_rows) != len(state_rows):
                raise ValueError(
                    f"written HDF5 row surface differs for source term {term!r}"
                )
            for row in state_rows:
                source = np.asarray(result.rhs_terms[term][row])
                written = np.asarray(h5[f"rhs_terms/{term}/{row}"][()])
                if source.dtype != written.dtype or source.shape != written.shape:
                    raise ValueError(
                        f"written HDF5 dtype/shape differs for {term}/{row}"
                    )
                if not np.array_equal(source, written, equal_nan=True):
                    raise ValueError(
                        f"written HDF5 values differ for {term}/{row}"
                    )


def _qualify_hdf5(path, **metadata):
    result = metadata.pop("result")
    census = metadata.pop("census")
    state_rows = census["state_rows"]
    supports = tuple(_support_path(row, state_rows) for row in state_rows)
    with h5py.File(path, "r+") as h5:
        h5.attrs["capture_profile"] = PHASE3_CAPTURE_PROFILE
        h5.attrs["run_id"] = metadata["run_id"]
        h5.attrs["capture_revision"] = str(metadata["capture_revision"])
        h5.attrs["producer_path"] = metadata["producer_path"]
        h5.attrs["started_at"] = str(metadata["started_at"])
        h5.attrs["completed_at"] = str(metadata["completed_at"])
        h5.attrs["kernel_provenance"] = "pure"
        h5.attrs["configuration_identity"] = metadata[
            "configuration_identity_sha256"
        ]
        h5.attrs["recipe_identity"] = str(metadata["recipe_identity"])
        h5.attrs["run_controls_json"] = _canonical_json_text(
            metadata["run_controls"]
        )
        h5.attrs["invocation_json"] = _canonical_json_text(metadata["invocation"])
        qualification = h5.create_group("qualification", track_order=True)
        _write_text_dataset(qualification, "term_names", census["term_names"])
        _write_text_dataset(qualification, "state_rows", state_rows)
        _write_text_dataset(
            qualification,
            "row_units",
            tuple(_ROW_UNITS[row] for row in state_rows),
        )
        _write_text_dataset(qualification, "row_support_paths", supports)
        _write_text_dataset(
            qualification, "axis_names", ("saved_time", "solver_cell")
        )
        _write_text_dataset(
            qualification, "axis_paths", ("/time", "/geometry/z_cm")
        )
        qualification.create_dataset(
            "packed_state_shape",
            data=np.asarray(
                [len(np.asarray(result.time)), len(state_rows), len(result.z_cm)],
                dtype=np.int64,
            ),
        )
        if set(state_rows) & _ANNULUS_ROWS:
            h5["geometry"].create_dataset(
                "annulus_volume_cm3",
                data=np.asarray(result.neutral_volume_cm3)
                - np.asarray(result.plasma_volume_cm3),
            )
        h5["time"].attrs["axis"] = "saved_time"
        h5["y"].attrs["axis_0"] = "saved_time:/time"
        h5["y"].attrs["axis_1"] = (
            "packed_conservative_state:row-major(state_rows,solver_cell)"
        )
        for term in census["term_names"]:
            for row, support in zip(state_rows, supports, strict=True):
                dataset = h5[f"rhs_terms/{term}/{row}"]
                dataset.attrs["axis_0"] = "saved_time:/time"
                dataset.attrs["axis_1"] = "solver_cell:/geometry/z_cm"
                dataset.attrs["unit"] = _ROW_UNITS[row]
                dataset.attrs["support_dataset"] = support


def _validate_qualified_hdf5(path):
    with h5py.File(path, "r") as h5:
        if _text(h5.attrs.get("format", "")) != RESULT_VERSION:
            raise ValueError("qualified artifact has wrong source HDF5 format")
        for name in _QUALIFICATION_REQUIRED_ATTRS:
            if name not in h5.attrs:
                raise ValueError(f"qualified artifact lacks root attribute {name!r}")
        if _text(h5.attrs["capture_profile"]) != PHASE3_CAPTURE_PROFILE:
            raise ValueError("HDF5 is not Phase 3 qualified evidence")
        validate_run_id(_text(h5.attrs["run_id"]))
        if _text(h5.attrs["kernel_provenance"]) != "pure":
            raise ValueError("qualified artifact does not record the pure kernel")
        term_names = _read_text_dataset(h5["qualification/term_names"])
        state_rows = _read_text_dataset(h5["qualification/state_rows"])
        units = _read_text_dataset(h5["qualification/row_units"])
        supports = _read_text_dataset(h5["qualification/row_support_paths"])
        if len(units) != len(state_rows) or len(supports) != len(state_rows):
            raise ValueError("row census/unit/support lengths differ")
        if state_rows != _serialized_state_rows(h5):
            raise ValueError(
                "qualified row census differs from source-owned conservative order"
            )
        if len(term_names) != len(set(term_names)):
            raise ValueError("qualified term census contains duplicates")
        actual_terms = set(h5["rhs_terms"].keys())
        if actual_terms != set(term_names) or len(actual_terms) != len(term_names):
            raise ValueError("RHS term census differs from captured surface")
        expected_shape = (len(h5["time"]), len(h5["geometry/z_cm"]))
        for term in term_names:
            actual_rows = set(h5[f"rhs_terms/{term}"].keys())
            if actual_rows != set(state_rows) or len(actual_rows) != len(state_rows):
                raise ValueError(f"RHS row census differs for term {term!r}")
            for row, unit, support in zip(state_rows, units, supports, strict=True):
                dataset = h5[f"rhs_terms/{term}/{row}"]
                if dataset.shape != expected_shape:
                    raise ValueError(f"RHS shape differs for {term}/{row}")
                if _text(dataset.attrs.get("unit", "")) != unit:
                    raise ValueError(f"RHS unit differs for {term}/{row}")
                if _text(dataset.attrs.get("support_dataset", "")) != support:
                    raise ValueError(f"RHS support differs for {term}/{row}")
                if support not in h5:
                    raise ValueError(f"RHS support dataset is absent: {support}")
        embedded = _text(h5.attrs["scientific_payload_sha256"])
    if embedded != scientific_payload_digest(path):
        raise ValueError("embedded scientific-payload SHA-256 mismatch")


def _serialized_state_rows(h5):
    params = json.loads(_text(h5.attrs["params_json"]))
    flags = json.loads(_text(h5.attrs["flags_json"]))
    present = {name for name in _optional_state_names() if name in h5}
    return _canonical_state_rows(params, flags, present)


def _support_path(row, state_rows):
    if row in _PLASMA_ROWS:
        return "/geometry/plasma_volume_cm3"
    if row in _ANNULUS_ROWS:
        return "/geometry/annulus_volume_cm3"
    if row in {"nn", "En"}:
        return (
            "/geometry/plasma_volume_cm3"
            if "nn_a" in state_rows
            else "/geometry/neutral_volume_cm3"
        )
    if row == "M_n":
        # The selected uniform-radial closure keeps M_n as chamber-mean
        # momentum even when nn/En are split onto the plasma column.
        return "/geometry/neutral_volume_cm3"
    raise ValueError(f"no support contract for row {row!r}")


def _scientific_dataset_paths(h5):
    paths = ["/time", "/y"]
    for name, value in h5["geometry"].items():
        if isinstance(value, h5py.Dataset) and value.dtype.kind not in {"O", "S", "U"}:
            paths.append(f"/geometry/{name}")
    for term in h5["qualification/term_names"].asstr()[()]:
        for row in h5["qualification/state_rows"].asstr()[()]:
            paths.append(f"/rhs_terms/{term}/{row}")
    for name, value in h5["qualification"].items():
        if isinstance(value, h5py.Dataset):
            paths.append(f"/qualification/{name}")
    return tuple(sorted(paths))


def _repository_relative(path, root):
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"artifact path must be inside repository root: {path}"
        ) from exc


def _validate_relative_path(value):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"path must be repository-relative: {value!r}")
    return path.as_posix()


def _reject_private_or_absolute_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_private_or_absolute_values(key)
            _reject_private_or_absolute_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_private_or_absolute_values(item)
    elif isinstance(value, Path):
        _reject_private_or_absolute_values(str(value))
    elif isinstance(value, str):
        lowered = value.lower()
        if (
            value.startswith(("/", "~"))
            or ":\\" in value
            or "file://" in lowered
            or "http://" in lowered
            or "https://" in lowered
            or "x-amz-signature" in lowered
            or "sig=" in lowered
            or "password=" in lowered
            or "token=" in lowered
            or "secret=" in lowered
            or "authorization:" in lowered
        ):
            raise ValueError(
                f"absolute/private/mutable locator is forbidden: {value!r}"
            )


def _write_text_dataset(group, name, values):
    encoded = tuple(str(value).encode("utf-8") for value in values)
    width = max((len(value) for value in encoded), default=1)
    group.create_dataset(
        name,
        data=np.asarray(encoded, dtype=f"S{width}"),
    )


def _read_text_dataset(dataset):
    values = dataset.asstr()[()]
    return tuple(str(value) for value in np.atleast_1d(values))


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _canonical_json_bytes(value):
    return _canonical_json_text(value).encode("utf-8")


def _canonical_json_text(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")
