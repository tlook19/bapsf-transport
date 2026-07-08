import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from ..core.timestep import TimestepDiagnostics


RESULT_VERSION = "sim1d-hdf5-v1"


def save_result_hdf5(path, result, params=None, flags=None):
    """Write a ``LAPDSim1D.run`` result namespace to an HDF5 file."""
    path = Path(path)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = RESULT_VERSION
        h5.attrs["solver"] = "LAPDSim1D"
        h5.attrs["steps"] = int(result.steps)
        h5.attrs["final_time"] = float(result.final_time)
        if params is not None:
            h5.attrs["params_json"] = _json_dumps(params)
        if flags is not None:
            h5.attrs["flags_json"] = _json_dumps(flags)

        _write_arrays(
            h5,
            result,
            (
                "time",
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
            ),
        )
        geometry["cell_role"] = _read_string_array(h5["geometry/cell_role"])
        params = _read_json_attr(h5, "params_json")
        flags = _read_json_attr(h5, "flags_json")

        return SimpleNamespace(
            **arrays,
            **geometry,
            rhs_terms=_read_nested_fields(h5["rhs_terms"]),
            total_rhs=_read_field_arrays(h5["total_rhs"]),
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
            params=params,
            flags=flags,
            path=path,
        )


def _write_arrays(group, owner, names):
    for name in names:
        if hasattr(owner, name):
            group.create_dataset(name, data=np.asarray(getattr(owner, name)))


def _write_geometry(group, result):
    numeric_names = (
        "z_cm",
        "length_cm",
        "Rp_cm",
        "Rm_cm",
        "plasma_volume_cm3",
        "neutral_volume_cm3",
        "volume_ratio",
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
    str_dtype = h5py.string_dtype(encoding="utf-8")
    for field_name, values in fields.items():
        values = np.asarray(values)
        if values.dtype.kind in {"U", "S", "O"}:
            group.create_dataset(
                field_name,
                data=np.asarray(values, dtype=object),
                dtype=str_dtype,
            )
        else:
            group.create_dataset(field_name, data=values)


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
    return {name: group[name][()] for name in names if name in group}


def _read_nested_fields(group):
    return {
        term_name: _read_field_arrays(term_group)
        for term_name, term_group in group.items()
    }


def _read_field_arrays(group):
    return {
        field_name: (
            _read_string_array(dataset)
            if h5py.check_string_dtype(dataset.dtype) is not None
            else dataset[()]
        )
        for field_name, dataset in group.items()
    }


def _read_term_arrays(group):
    return {term_name: dataset[()] for term_name, dataset in group.items()}


def _read_diagnostics(group):
    count = int(group.attrs.get("count", 0))
    if count == 0:
        return []
    diagnostics = []
    field_names = tuple(TimestepDiagnostics.__dataclass_fields__)
    loaded = {field_name: group[field_name][()] for field_name in field_names}
    for i in range(count):
        kwargs = {}
        for field_name, values in loaded.items():
            value = values[i]
            if field_name == "active_constraint":
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
