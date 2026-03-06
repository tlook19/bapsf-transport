"""
HDF5 database read/write for LAPDSim sweep results.

Schema
------
sweep.h5
├── attrs: {created, description}
├── runs/
│   ├── run_0000/
│   │   ├── attrs: {param_*, flag_*, timestamp, status}
│   │   ├── <result arrays>          # all keys from get_results()
│   │   └── stats_10_20ms/
│   │       └── attrs: {ne_var, ne_min, ...}
│   └── run_0001/ ...
└── index/
    ├── run_ids          resizable str dataset
    ├── status           resizable str dataset
    ├── n_cells          resizable int dataset
    ├── params/{name}    resizable float dataset per param
    ├── flags/{name}     resizable int (0/1) dataset per flag
    └── stats_10_20ms/{name}  resizable float dataset per stat
"""
import contextlib
import datetime

import h5py
import numpy as np


@contextlib.contextmanager
def open_db(path, mode="r"):
    """
    Context manager returning an open h5py.File.

    Parameters
    ----------
    path : str or path-like
    mode : str
        'r'  read-only, 'r+' read-write, 'a' append/create, 'w' truncate+create.
    """
    import pathlib
    p = pathlib.Path(path).expanduser()
    if mode in ("w", "a"):
        p.parent.mkdir(parents=True, exist_ok=True)
    db = h5py.File(p, mode)
    try:
        if mode in ("w", "a"):
            db.require_group("runs")
            db.require_group("index")
            if "created" not in db.attrs:
                db.attrs["created"] = datetime.datetime.now().isoformat()
        yield db
    finally:
        db.close()


# ── Internal helpers ──────────────────────────────────────────────────────────

def _str_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _append_dataset(grp, key, val):
    """
    Append a single value to a resizable dataset in `grp`, creating it if needed.

    Strings are stored as variable-length UTF-8.
    Booleans are stored as int8.
    Floats and ints are stored as float64 and int32 respectively.
    """
    if isinstance(val, bool):
        np_val = np.int8(val)
        dtype = "i1"
    elif isinstance(val, (int, np.integer)):
        np_val = np.int32(val)
        dtype = "i4"
    elif isinstance(val, str):
        np_val = val
        dtype = _str_dtype()
    else:
        np_val = np.float64(val)
        dtype = "f8"

    if key not in grp:
        if isinstance(np_val, str):
            grp.create_dataset(
                key,
                data=np.array([np_val], dtype=object),
                maxshape=(None,),
                dtype=dtype,
            )
        else:
            grp.create_dataset(
                key,
                data=np.array([np_val]),
                maxshape=(None,),
                dtype=dtype,
            )
    else:
        ds = grp[key]
        n = ds.shape[0]
        ds.resize((n + 1,))
        ds[n] = np_val


# ── Public API ────────────────────────────────────────────────────────────────

def save_run(db, run_id, params, flags, results, stats):
    """
    Write one simulation run to ``db['runs/{run_id}']``.

    If the run_id already exists it is overwritten.

    Parameters
    ----------
    db : h5py.File
        Opened in 'a' or 'w' mode.
    run_id : str
    params : dict
        Input parameter dict (from ``input_dict_template``).
    flags : dict
        Input flags dict (from ``input_flags_template``).
    results : dict
        Output of ``sim.get_results()``.
    stats : dict
        Output of ``compute_window_stats()``.
    """
    runs = db.require_group("runs")
    if run_id in runs:
        del runs[run_id]

    grp = runs.create_group(run_id)
    grp.attrs["timestamp"] = datetime.datetime.now().isoformat()
    grp.attrs["status"] = "ok"

    # Store params as attrs
    for k, v in params.items():
        try:
            grp.attrs[f"param_{k}"] = v
        except TypeError:
            grp.attrs[f"param_{k}"] = str(v)

    # Store flags as attrs
    for k, v in flags.items():
        grp.attrs[f"flag_{k}"] = bool(v)

    # Store result arrays
    for key, val in results.items():
        arr = np.asarray(val)
        grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)

    # Store pre-computed stats as attrs on a subgroup
    sg = grp.create_group("stats_10_20ms")
    for k, v in stats.items():
        sg.attrs[k] = float(v)


def load_run(db, run_id, keys=None):
    """
    Load result arrays for one run.

    Parameters
    ----------
    db : h5py.File
    run_id : str
    keys : list of str or None
        Which result arrays to load.  ``None`` loads every array.

    Returns
    -------
    params : dict
    flags  : dict
    results : dict of {key: np.ndarray}
    """
    grp = db["runs"][run_id]

    params = {}
    flags = {}
    for attr_key, attr_val in grp.attrs.items():
        if attr_key.startswith("param_"):
            params[attr_key[6:]] = attr_val
        elif attr_key.startswith("flag_"):
            flags[attr_key[5:]] = bool(attr_val)

    all_keys = [k for k in grp.keys() if k != "stats_10_20ms"]
    load_keys = keys if keys is not None else all_keys

    results = {}
    for k in load_keys:
        if k in grp:
            results[k] = grp[k][:]

    return params, flags, results


def load_run_stats(db, run_id):
    """Load the pre-computed window stats for a single run."""
    sg = db["runs"][run_id]["stats_10_20ms"]
    return {k: float(v) for k, v in sg.attrs.items()}


def list_runs(db):
    """Return sorted list of run_ids present in the database."""
    return sorted(db.get("runs", {}).keys())


def update_index(db, run_id, params, flags, stats, n_cells, status="ok"):
    """
    Append one row to the index datasets.

    Creates datasets on the first call; resizes and appends on subsequent calls.
    """
    idx = db.require_group("index")

    _append_dataset(idx, "run_ids", run_id)
    _append_dataset(idx, "status", status)
    _append_dataset(idx, "n_cells", int(n_cells))

    p_grp = idx.require_group("params")
    n_rows = idx["run_ids"].shape[0]
    # Union of existing param keys and this run's param keys; pad missing entries with NaN
    # (same pattern as stats) so all numeric param arrays stay aligned with run_ids.
    all_param_keys = set(p_grp.keys()) | set(params.keys())
    for k in all_param_keys:
        if k in params:
            v = params[k]
            if isinstance(v, bool):
                val = int(v)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                val = float(v)
            else:
                val = str(v)
        else:
            # Key exists from a prior run but not this one — pad with NaN for numeric,
            # empty string for string datasets.
            if k in p_grp and p_grp[k].dtype.kind in ("S", "O", "U"):
                val = ""
            else:
                val = float("nan")
        current_len = p_grp[k].shape[0] if k in p_grp else 0
        for _ in range(n_rows - 1 - current_len):
            pad = "" if k in p_grp and p_grp[k].dtype.kind in ("S", "O", "U") else float("nan")
            _append_dataset(p_grp, k, pad)
        _append_dataset(p_grp, k, val)

    f_grp = idx.require_group("flags")
    for k, v in flags.items():
        _append_dataset(f_grp, k, int(bool(v)))

    s_grp = idx.require_group("stats_10_20ms")
    # n_rows = total runs including this one (run_ids was already appended above)
    n_rows = idx["run_ids"].shape[0]
    # Union of existing stat keys and this run's stat keys
    all_stat_keys = set(s_grp.keys()) | set(stats.keys())
    for k in all_stat_keys:
        v = float(stats[k]) if k in stats else float("nan")
        # Pad any missing entries from prior runs that didn't have this key (e.g. failures
        # before the first success, or a new key introduced mid-sweep).
        current_len = s_grp[k].shape[0] if k in s_grp else 0
        for _ in range(n_rows - 1 - current_len):
            _append_dataset(s_grp, k, float("nan"))
        _append_dataset(s_grp, k, v)


def load_index(db):
    """
    Return a dict-of-arrays summarising all runs.

    Returns
    -------
    dict with keys:
        run_ids       : list of str
        status        : list of str
        n_cells       : np.ndarray int
        params        : dict of {name: np.ndarray}
        flags         : dict of {name: np.ndarray bool}
        stats_10_20ms : dict of {name: np.ndarray float}
    """
    if "index" not in db:
        return {
            "run_ids": [],
            "status": [],
            "n_cells": np.array([], dtype=int),
            "params": {},
            "flags": {},
            "stats_10_20ms": {},
        }

    idx = db["index"]

    if "run_ids" not in idx:
        return {
            "run_ids": [],
            "status": [],
            "n_cells": np.array([], dtype=int),
            "params": {},
            "flags": {},
            "stats_10_20ms": {},
        }

    def _decode(arr):
        return [s.decode() if isinstance(s, bytes) else s for s in arr]

    result = {
        "run_ids": _decode(idx["run_ids"][:]),
        "status": _decode(idx["status"][:]),
        "n_cells": idx["n_cells"][:].astype(int),
        "params": {},
        "flags": {},
        "stats_10_20ms": {},
    }

    for k in idx.get("params", {}).keys():
        ds = idx["params"][k][:]
        result["params"][k] = _decode(ds) if ds.dtype.kind in ("O", "S", "U") else ds

    for k in idx.get("flags", {}).keys():
        result["flags"][k] = idx["flags"][k][:].astype(bool)

    for k in idx.get("stats_10_20ms", {}).keys():
        result["stats_10_20ms"][k] = idx["stats_10_20ms"][k][:]

    return result


def rebuild_index(db):
    """
    Rebuild the ``index/`` group from the ``runs/`` group.

    Useful after partial failures or manual edits.
    """
    if "index" in db:
        del db["index"]
    db.require_group("index")

    for run_id in sorted(db.get("runs", {}).keys()):
        grp = db["runs"][run_id]
        status = grp.attrs.get("status", "ok")

        n_cells = int(grp["ne"].shape[1]) if "ne" in grp else 0
        params = {k[6:]: v for k, v in grp.attrs.items() if k.startswith("param_")}
        flags = {k[5:]: bool(v) for k, v in grp.attrs.items() if k.startswith("flag_")}
        stats = {}
        if "stats_10_20ms" in grp:
            stats = {k: float(v) for k, v in grp["stats_10_20ms"].attrs.items()}

        update_index(db, run_id, params, flags, stats, n_cells, status)
