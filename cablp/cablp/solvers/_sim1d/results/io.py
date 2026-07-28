import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from ..core.config import resolve_config
from ..core.timestep import TimestepDiagnostics
from .compat import add_sim3_compat_aliases


RESULT_VERSION = "sim1d-hdf5-v1"


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
        if params is not None:
            h5.attrs["params_json"] = _json_dumps(params)
        if flags is not None:
            h5.attrs["flags_json"] = _json_dumps(flags)

        _write_arrays(
            h5,
            result,
            (
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
                # Optional fields, present only when the run evolved them
                # (the neutral_momentum / neutral_two_zone flags); readers
                # tolerate their absence, so the 5-field format is
                # unchanged. With nn_a present, nn is the COLUMN density.
                "M_n",
                "u_n",
                "nn_a",
                "M_n_a",
                "u_n_a",
            ),
        )
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

        arrays = _read_arrays(
            h5,
            (
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
                "M_n",
                "u_n",
                "nn_a",
                "M_n_a",
                "u_n_a",
            ),
        )
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
