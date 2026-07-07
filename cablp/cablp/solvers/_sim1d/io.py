import json
from pathlib import Path

import h5py
import numpy as np


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
        _write_diagnostics(h5.create_group("diagnostics"), result.diagnostics)
    return path


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
    for field_name, values in fields.items():
        group.create_dataset(field_name, data=np.asarray(values))


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


def _json_dumps(value):
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
