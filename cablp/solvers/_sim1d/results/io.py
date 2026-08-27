import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.cathode.kernels import PURE_PROVENANCE as PURE_KERNEL_PROVENANCE

from ..core.config import resolve_config
from ..core.timestep import TimestepDiagnostics
from ..physics.hot_neutrals import HOT_CHANNEL_DIAGNOSTIC_FIELDS
from ..physics.sources import (
    IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS,
)
from .compat import add_sim3_compat_aliases


RESULT_VERSION = "sim1d-hdf5-v1"

# The optional per-sample arrays a run may or may not carry. ``_write_arrays``
# skips the ones the result lacks and ``_read_arrays`` skips the ones the file
# lacks, so this list is additive: adding a name here cannot change any file a
# previous run wrote, and a file written before a name existed still loads with
# that attribute simply absent.
#
# HOT-CHANNEL DIAGNOSTICS, 2026-08-14. The hot channel's per-cell readings were
# computed and attached to the live result but were NOT in this list, so
# ``save_result_hdf5`` silently dropped them and every artifact written before
# this date is missing them. Those files are not migrated: absence means "never
# persisted", never zero.
_OPTIONAL_ARRAY_FIELDS = (
    # Present only when the run evolved them (the neutral_momentum /
    # neutral_two_zone / neutral_energy flags); readers tolerate their
    # absence, so the 5-field format is unchanged. With nn_a present, nn is
    # the COLUMN density. En / Tn ride the same volume as nn.
    "M_n",
    "u_n",
    "nn_a",
    "M_n_a",
    "u_n_a",
    "En",
    "Tn",
) + HOT_CHANNEL_DIAGNOSTIC_FIELDS + IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS

_ARRAY_FIELDS = (
    "time",
    "phase",
    "phase_elapsed",
    "phase_cathode_enabled",
    "phase_gas_puff_enabled",
    "phase_floating",
    "y",
    "n",
    "nn",
    "M",
    "momentum",
    "Ee",
    "Ei",
    "u",
    "Te",
    "Ti",
    "pe",
    "pi",
    "p",
) + _OPTIONAL_ARRAY_FIELDS

# LABEL-SEMANTICS BOUNDARY, 2026-08-05. ``diagnostics/active_constraint`` now
# always names the bound that actually minimized. Files written BEFORE this
# date instead carry the literal "dt_min" on every step whose timestep was
# clamped up to ``dt_min``, overwriting the true bound's name. Those files are
# historical records and are NOT migrated: a "dt_min" entry in one of them
# means "this step was clamped, by an unrecorded bound". From this date the
# clamp is carried by the separate ``clamped_to_dt_min`` flag (0.0/1.0)
# alongside ``dt_raw``, the unclamped request. Both are defaulted fields, so
# older files still load, reading 0.0 and NaN respectively.


def _resolve_config_namespace(kind, supplied):
    """Resolve one namespace the way ``LAPDSim1D.__init__`` resolves it.

    ``resolve_config`` owns both namespaces; the untouched one falls back to
    its template and is discarded here, so ``params`` and ``flags`` stay
    independent.
    """
    if kind == "params":
        return resolve_config(params=supplied)[0]
    return resolve_config(flags=supplied)[1]


def _check_config_metadata(kind, supplied, constructed):
    """Raise unless ``supplied`` would construct ``constructed``'s config.

    Callers naturally hold the PRE-resolution override mapping they passed to
    ``LAPDSim1D``, while ``result.params``/``result.flags`` are the resolved
    config. Compare post-resolution so any input that would build the same
    solver config passes, and a genuinely different one still raises.
    """
    supplied = dict(supplied)
    constructed = dict(constructed)
    if supplied == constructed:
        return
    resolved_supplied = _resolve_config_namespace(kind, supplied)
    resolved_constructed = _resolve_config_namespace(kind, constructed)
    if resolved_supplied == resolved_constructed:
        return
    differing = sorted(
        key
        for key in set(resolved_supplied) | set(resolved_constructed)
        if resolved_supplied.get(key) != resolved_constructed.get(key)
    )
    raise ValueError(
        f"{kind} metadata differs from the constructed LAPDSim1D config "
        f"after resolution; differing keys: {differing}"
    )


def save_result_hdf5(path, result, params=None, flags=None):
    """Write a ``LAPDSim1D.run`` result namespace to an HDF5 file."""
    result_params = getattr(result, "params", None)
    result_flags = getattr(result, "flags", None)
    if result_params is not None:
        if params is not None:
            _check_config_metadata("params", params, result_params)
        params = result_params
    if result_flags is not None:
        if flags is not None:
            _check_config_metadata("flags", flags, result_flags)
        flags = result_flags
    path = Path(path)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = RESULT_VERSION
        h5.attrs["solver"] = "LAPDSim1D"
        h5.attrs["steps"] = int(result.steps)
        h5.attrs["final_time"] = float(result.final_time)
        h5.attrs["t_prebreakdown_trigger"] = float(
            getattr(result, "t_prebreakdown_trigger", np.nan)
        )
        h5.attrs["t_breakdown_trigger"] = float(
            getattr(result, "t_breakdown_trigger", np.nan)
        )
        # Present only on opt-in max_steps_action="stop" runs; default runs
        # never carry it, keeping their files byte-identical to before.
        run_status = getattr(result, "run_status", None)
        if run_status is not None:
            h5.attrs["run_status"] = str(run_status)
        # Kernel provenance (D3/D4): which arithmetic produced this
        # trajectory -- "pure" for the default Python path, otherwise the
        # compiled kernel module's own id. Written whenever the result carries
        # it, so no new artifact is anonymous about its kernel path; an
        # artifact with NO compiled_kernels attribute predates the selector
        # and is pure by construction.
        compiled_kernels = getattr(result, "compiled_kernels", None)
        if compiled_kernels is not None:
            h5.attrs["compiled_kernels"] = str(compiled_kernels)
        # Presence-gated execution identity. Ordinary runs and their files do
        # not acquire an identity implicitly; qualified Phase 3 runs supply it
        # before solver construction and preserve it here.
        run_id = getattr(result, "run_id", None)
        if run_id is not None:
            from .phase3_capture import validate_run_id

            h5.attrs["run_id"] = validate_run_id(run_id)
        if params is not None:
            h5.attrs["params_json"] = _json_dumps(params)
        if flags is not None:
            h5.attrs["flags_json"] = _json_dumps(flags)

        _write_arrays(h5, result, _ARRAY_FIELDS)
        _write_geometry(h5.create_group("geometry"), result)
        _write_nested_fields(h5.create_group("rhs_terms"), result.rhs_terms)
        _write_term_arrays(
            h5.create_group("electron_energy_terms_W_cm3"),
            result.electron_energy_terms_W_cm3,
        )
        _write_term_arrays(
            h5.create_group("ion_energy_terms_W_cm3"),
            result.ion_energy_terms_W_cm3,
        )
        _write_field_arrays(h5.create_group("total_rhs"), result.total_rhs)
        if hasattr(result, "floor_ledger"):
            _write_field_arrays(
                h5.create_group("floor_ledger"), result.floor_ledger
            )
        # K2d transfer-ledger census. Written only by a run that built the
        # DVM arm, so a moment-model file's dataset layout is unchanged and
        # the absence of the group means "never recorded", never "zero".
        #
        # The group grows ADDITIVELY and is read by name, so a reader must
        # not assume a fixed key set. The exponential transfer hold added
        # ``transfer_hold`` (a string attr naming the selector),
        # ``{Ei,M}_hold_debt`` (per-cell) with their ``_total`` /
        # ``_max_abs`` scalars, and ``sample_{Ei,M}_hold_debt_total``; on a
        # file written before it, their absence means the run predates the
        # hold, i.e. it ran the zero-order hold with no hold debt to carry.
        # The residual the ``*_residual_rel`` scalars report changed meaning
        # with them: it is now ``applied + debt + hold_debt - booked``.
        dvm_ledger = getattr(result, "dvm_transfer_ledger", None)
        if dvm_ledger:
            _write_census(h5.create_group("dvm_transfer_ledger"), dvm_ledger)
        if hasattr(result, "atomic_rate_domain"):
            _write_field_arrays(
                h5.create_group("atomic_rate_domain"),
                result.atomic_rate_domain,
            )
        if hasattr(result, "phase_events"):
            _write_field_arrays(h5.create_group("phase_events"), result.phase_events)
        if hasattr(result, "timestep_rejection_events"):
            _write_field_arrays(
                h5.create_group("timestep_rejection_events"),
                result.timestep_rejection_events,
            )
        if hasattr(result, "current_trigger_samples"):
            _write_field_arrays(
                h5.create_group("current_trigger_samples"),
                result.current_trigger_samples,
            )
        if hasattr(result, "cathode_diagnostics"):
            _write_field_arrays(
                h5.create_group("cathode_diagnostics"),
                result.cathode_diagnostics,
            )
        if hasattr(result, "ignition_diagnostics"):
            _write_field_arrays(
                h5.create_group("ignition_diagnostics"),
                result.ignition_diagnostics,
            )
        if hasattr(result, "gas_puff_diagnostics"):
            _write_field_arrays(
                h5.create_group("gas_puff_diagnostics"),
                result.gas_puff_diagnostics,
            )
        # Present only on a run whose cathode switch was opened by an ignition
        # guard; a normal run never carries it, so its file is unchanged.
        ignition_abort = getattr(result, "ignition_abort", None)
        if ignition_abort is not None:
            group = h5.create_group("ignition_abort")
            for key, value in ignition_abort.items():
                if isinstance(value, str):
                    group.attrs[key] = value
                else:
                    group.attrs[key] = float(value)
        _write_diagnostics(h5.create_group("diagnostics"), result.diagnostics)
    return path


def load_result_hdf5(path):
    """Load a ``save_result_hdf5`` file into a run-result namespace."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        file_format = h5.attrs.get("format", "")
        if _decode_string(file_format) != RESULT_VERSION:
            raise ValueError(
                f"unsupported sim1d result format {file_format!r}; "
                f"expected {RESULT_VERSION!r}"
            )

        arrays = _read_arrays(h5, _ARRAY_FIELDS)
        geometry = _read_arrays(
            h5["geometry"],
            (
                "z_cm",
                "length_cm",
                "Rp_cm",
                "Rm_cm",
                "plasma_volume_cm3",
                "neutral_volume_cm3",
                "volume_ratio",
                "plasma_active",
            ),
        )
        geometry["cell_role"] = _read_string_array(h5["geometry/cell_role"])
        params = _read_json_attr(h5, "params_json")
        flags = _read_json_attr(h5, "flags_json")

        result = SimpleNamespace(
            **arrays,
            **geometry,
            rhs_terms=_read_nested_fields(h5["rhs_terms"]),
            total_rhs=_read_field_arrays(h5["total_rhs"]),
            floor_ledger=(
                _read_field_arrays(h5["floor_ledger"])
                if "floor_ledger" in h5
                else {}
            ),
            atomic_rate_domain=(
                _read_field_arrays(h5["atomic_rate_domain"])
                if "atomic_rate_domain" in h5
                else {}
            ),
            phase_events=(
                _read_field_arrays(h5["phase_events"])
                if "phase_events" in h5
                else _empty_phase_events()
            ),
            timestep_rejection_events=(
                _read_field_arrays(h5["timestep_rejection_events"])
                if "timestep_rejection_events" in h5
                else _empty_timestep_rejection_events()
            ),
            current_trigger_samples=(
                _read_field_arrays(h5["current_trigger_samples"])
                if "current_trigger_samples" in h5
                else _empty_current_trigger_samples()
            ),
            cathode_diagnostics=(
                _read_field_arrays(h5["cathode_diagnostics"])
                if "cathode_diagnostics" in h5
                else {}
            ),
            # Defaulted dataset: runs saved before 2026-07-28 have no
            # ignition_diagnostics group and load with NaN/zero-defaulted
            # per-sample records of the right length.
            ignition_diagnostics=(
                _read_field_arrays(h5["ignition_diagnostics"])
                if "ignition_diagnostics" in h5
                else _empty_ignition_diagnostics(
                    len(_read_dataset(h5["time"]))
                )
            ),
            # Defaulted dataset: runs saved before 2026-07-29 have no
            # gas_puff_diagnostics group and load with NaN-defaulted
            # per-sample records of the right length.
            gas_puff_diagnostics=(
                _read_field_arrays(h5["gas_puff_diagnostics"])
                if "gas_puff_diagnostics" in h5
                else _empty_gas_puff_diagnostics(
                    len(_read_dataset(h5["time"]))
                )
            ),
            electron_energy_terms_W_cm3=_read_term_arrays(
                h5["electron_energy_terms_W_cm3"]
            ),
            ion_energy_terms_W_cm3=_read_term_arrays(
                h5["ion_energy_terms_W_cm3"]
            ),
            diagnostics=_read_diagnostics(h5["diagnostics"]),
            steps=int(h5.attrs["steps"]),
            final_time=float(h5.attrs["final_time"]),
            t_prebreakdown_trigger=_read_float_attr(
                h5,
                "t_prebreakdown_trigger",
                np.nan,
            ),
            t_breakdown_trigger=_read_float_attr(
                h5,
                "t_breakdown_trigger",
                np.nan,
            ),
            params=params,
            flags=flags,
            path=path,
        )
        if "run_status" in h5.attrs:
            result.run_status = _decode_string(h5.attrs["run_status"])
        # Absent on artifacts written before the kernel selector existed;
        # those ran the pure path, which is what the default resolves to.
        result.compiled_kernels = (
            _decode_string(h5.attrs["compiled_kernels"])
            if "compiled_kernels" in h5.attrs
            else PURE_KERNEL_PROVENANCE
        )
        if "run_id" in h5.attrs:
            from .phase3_capture import validate_run_id

            result.run_id = validate_run_id(_decode_string(h5.attrs["run_id"]))
        # Set only when the file carries it: a reader must be able to tell a
        # run whose census was never persisted from one whose census is zero.
        if "dvm_transfer_ledger" in h5:
            result.dvm_transfer_ledger = _read_census(h5["dvm_transfer_ledger"])
        if "ignition_abort" in h5:
            result.ignition_abort = {
                key: (
                    _decode_string(value)
                    if isinstance(value, (bytes, str))
                    else float(value)
                )
                for key, value in h5["ignition_abort"].attrs.items()
            }
        return add_sim3_compat_aliases(result)


def _write_arrays(group, owner, names):
    for name in names:
        if hasattr(owner, name):
            _write_array_dataset(group, name, getattr(owner, name))


def _write_geometry(group, result):
    numeric_names = (
        "z_cm",
        "length_cm",
        "Rp_cm",
        "Rm_cm",
        "plasma_volume_cm3",
        "neutral_volume_cm3",
        "volume_ratio",
        "plasma_active",
    )
    _write_arrays(group, result, numeric_names)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "cell_role",
        data=np.asarray(result.cell_role, dtype=object),
        dtype=str_dtype,
    )


def _write_nested_fields(group, terms):
    for term_name, fields in terms.items():
        term_group = group.create_group(term_name)
        _write_field_arrays(term_group, fields)


def _write_field_arrays(group, fields):
    for field_name, values in fields.items():
        _write_array_dataset(group, field_name, values)


def _write_array_dataset(group, name, values):
    values = np.asarray(values)
    if values.dtype.kind in {"U", "S", "O"}:
        group.create_dataset(
            name,
            data=np.asarray(values, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        group.create_dataset(name, data=values)


def _write_census(group, census):
    """Write a scalar/array census mapping: arrays as datasets, scalars as attrs.

    Integer counts are written as integers so the round trip returns the count,
    not a float that a report would have to re-cast, and STRING entries (a
    census that records which selector produced it -- the DVM transfer
    ledger's ``transfer_hold``) are written as string attrs and read back as
    ``str``, not coerced through float.
    """
    for name, value in census.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(name, data=np.asarray(value, dtype=float))
        elif isinstance(value, str):
            group.attrs[name] = value
        elif isinstance(value, (bool, int, np.integer)):
            group.attrs[name] = int(value)
        else:
            group.attrs[name] = float(value)


def _read_census(group):
    """Read a :func:`_write_census` group back into its scalar/array mapping."""
    census = {name: _read_dataset(dataset) for name, dataset in group.items()}
    for name, value in group.attrs.items():
        if isinstance(value, (str, bytes)):
            census[name] = (
                value.decode("utf-8") if isinstance(value, bytes) else value
            )
            continue
        value = np.asarray(value)
        census[name] = int(value) if value.dtype.kind in "iub" else float(value)
    return census


def _write_term_arrays(group, terms):
    for term_name, values in terms.items():
        group.create_dataset(term_name, data=np.asarray(values))


def _write_diagnostics(group, diagnostics):
    group.attrs["count"] = len(diagnostics)
    if not diagnostics:
        return
    field_names = tuple(diagnostics[0].__dataclass_fields__)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    for field_name in field_names:
        values = [getattr(diag, field_name) for diag in diagnostics]
        if isinstance(values[0], str):
            group.create_dataset(
                field_name,
                data=np.asarray(values, dtype=object),
                dtype=str_dtype,
            )
        else:
            group.create_dataset(field_name, data=np.asarray(values, dtype=float))


def _read_arrays(group, names):
    return {name: _read_dataset(group[name]) for name in names if name in group}


def _read_nested_fields(group):
    return {
        term_name: _read_field_arrays(term_group)
        for term_name, term_group in group.items()
    }


def _read_field_arrays(group):
    return {
        field_name: _read_dataset(dataset)
        for field_name, dataset in group.items()
    }


def _read_term_arrays(group):
    return {term_name: dataset[()] for term_name, dataset in group.items()}


def _read_dataset(dataset):
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return _read_string_array(dataset)
    return dataset[()]


def _read_diagnostics(group):
    count = int(group.attrs.get("count", 0))
    if count == 0:
        return []
    diagnostics = []
    field_names = tuple(TimestepDiagnostics.__dataclass_fields__)
    loaded = {}
    defaults = {
        field_name: field.default
        for field_name, field in TimestepDiagnostics.__dataclass_fields__.items()
    }
    for field_name in field_names:
        if field_name in group:
            loaded[field_name] = group[field_name][()]
        else:
            loaded[field_name] = [defaults[field_name]] * count
    for i in range(count):
        kwargs = {}
        for field_name, values in loaded.items():
            value = values[i]
            if field_name in {
                "active_constraint",
                "phase",
                "step_cap",
                "rejection_reason",
            }:
                value = _decode_string(value)
            else:
                value = float(value)
            kwargs[field_name] = value
        diagnostics.append(TimestepDiagnostics(**kwargs))
    return diagnostics


def _read_string_array(dataset):
    return np.asarray([_decode_string(value) for value in dataset[()]], dtype=object)


def _read_json_attr(group, name):
    if name not in group.attrs:
        return None
    return json.loads(_decode_string(group.attrs[name]))


def _read_float_attr(group, name, default):
    if name not in group.attrs:
        return float(default)
    return float(group.attrs[name])


def _empty_phase_events():
    return {
        "time": np.asarray([], dtype=float),
        "phase": np.asarray([], dtype=object),
        "reason": np.asarray([], dtype=object),
    }


def _empty_ignition_diagnostics(samples):
    """Return NaN/zero-defaulted ignition diagnostics for a pre-2026-07-28 file."""
    from ..core.ignition import IGNITION_DIAGNOSTIC_FIELDS

    zero_fields = {"armed", "joint_negative", "stalled"}
    return {
        name: np.zeros(samples, dtype=float)
        if name in zero_fields
        else np.full(samples, np.nan, dtype=float)
        for name in IGNITION_DIAGNOSTIC_FIELDS
    }


def _empty_gas_puff_diagnostics(samples):
    """Return NaN-defaulted puff waveform records for a pre-2026-07-29 file.

    NaN, not zero: an old file does not record that the puff was off, it
    records nothing, and a reader must not read that silence as no fuel.
    """
    from ..physics.neutrals import GAS_PUFF_DIAGNOSTIC_FIELDS

    return {
        name: np.full(samples, np.nan, dtype=float)
        for name in GAS_PUFF_DIAGNOSTIC_FIELDS
    }


def _empty_current_trigger_samples():
    return {
        "time": np.asarray([], dtype=float),
        "I_tot": np.asarray([], dtype=float),
    }


def _empty_timestep_rejection_events():
    return {
        "time": np.asarray([], dtype=float),
        "attempted_dt": np.asarray([], dtype=float),
        "retry_index": np.asarray([], dtype=float),
        "reason": np.asarray([], dtype=object),
        "phase": np.asarray([], dtype=object),
        "active_constraint": np.asarray([], dtype=object),
    }


def _decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_dumps(value):
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
