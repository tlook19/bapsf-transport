"""Pre/post census of the g1atrim kinetic_dvm arm across the transfer hold.

Reads two saved runs and reports, over the window the pre-fix arm ground in,
the step budget, the dt distribution, the clamp census by signal, the
active-constraint and step-cap census, the transfer ledger, and the state of
the collector-end cells. Read-only; it opens the diagnostics and ledger
groups and the saved trajectory, never the per-step term arrays.

Usage:
    python scripts/dvmhold_arm_census.py PRE.h5 POST.h5 [--window MS MS]
"""

import argparse
import collections
import json

import h5py
import numpy as np

CELLS = (277, 278, 279)


def _s(values):
    return np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in values]
    )


def census(path, lo, hi):
    out = {"path": path}
    with h5py.File(path, "r") as h:
        params = json.loads(h.attrs["params_json"])
        out["dt_min"] = float(params.get("dt_min", 1e-12))
        out["transfer_hold"] = params.get(
            "neutral_kinetic_dvm_transfer_hold", "<unset -> exponential>"
        )
        out["steps"] = int(h.attrs["steps"])
        out["final_time_ms"] = 1e3 * float(h.attrs["final_time"])
        out["run_status"] = h.attrs.get("run_status", "<absent>")
        d = h["diagnostics"]
        t = np.asarray(d["time"])
        adt = np.asarray(d["accepted_dt"])
        raw = np.asarray(d["dt_raw"])
        clamp = np.asarray(d["clamped_to_dt_min"])
        acc_clamp = (
            np.asarray(d["clamped_to_dt_min_accepted"])
            if "clamped_to_dt_min_accepted" in d
            else np.full(t.size, np.nan)
        )
        ac = _s(d["active_constraint"][...])
        cap = _s(d["step_cap"][...])
        dt_sl = np.asarray(d["dt_surface_loss"])
        m = (t >= lo) & (t < hi)
        out["steps_in_window"] = int(m.sum())
        out["window_ms"] = (1e3 * lo, 1e3 * hi)
        if m.any():
            out["dt_min_acc"] = float(adt[m].min())
            out["dt_med_acc"] = float(np.median(adt[m]))
            out["dt_max_acc"] = float(adt[m].max())
            out["dt_surface_loss_min"] = float(dt_sl[m].min())
            out["dt_surface_loss_p05"] = float(np.percentile(dt_sl[m], 5))
            out["raw_clamps"] = int(clamp[m].sum())
            out["accepted_below_dt_min"] = int(
                np.count_nonzero(adt[m] <= out["dt_min"])
            )
            out["accepted_clamp_flag"] = (
                int(np.nansum(acc_clamp[m]))
                if np.isfinite(acc_clamp[m]).any()
                else None
            )
            out["constraints"] = collections.Counter(ac[m]).most_common(5)
            out["caps"] = collections.Counter(cap[m]).most_common(5)
            # dt_min clamps attributable to the DVM's own bundle.
            sl = m & (ac == "surface_loss")
            out["surface_loss_steps"] = int(sl.sum())
            out["surface_loss_clamps"] = int(clamp[sl].sum())
        g = h.get("dvm_transfer_ledger")
        if g is not None:
            out["ledger"] = {
                k: float(g.attrs[k])
                for k in (
                    "relax_steps", "relax_limited_steps", "limited_cells",
                    "Ei_debt_total", "Ei_debt_max_abs", "Ei_residual_rel",
                    "M_residual_rel",
                )
                if k in g.attrs
            }
            for k in ("Ei_hold_debt_total", "Ei_hold_debt_max_abs"):
                if k in g.attrs:
                    out["ledger"][k] = float(g.attrs[k])
            if "Ei_hold_debt" in g:
                hold = np.asarray(g["Ei_hold_debt"])
                out["hold_debt_cells"] = {
                    c: float(hold[c]) for c in CELLS if c < hold.size
                }
            if "sample_Ei_hold_debt_total" in g:
                series = np.asarray(g["sample_Ei_hold_debt_total"])
                out["hold_debt_series"] = (
                    float(series[0]), float(series[series.size // 2]),
                    float(series[-1]), float(np.max(np.abs(series))),
                )
            rcs = np.asarray(g["relax_cell_steps"])
            nz = np.flatnonzero(rcs)
            out["limited_cell_index"] = nz.tolist()[:12]
        tt = np.asarray(h["time"])
        Ti = np.asarray(h["Ti"])
        n = np.asarray(h["n"])
        j = int(np.argmin(np.abs(tt - min(hi, tt[-1]))))
        out["state_t_ms"] = 1e3 * float(tt[j])
        out["Ti_floor"] = float(params.get("Ti_floor", 0.02585))
        out["end_cells"] = {
            c: (float(Ti[j, c]), float(n[j, c]))
            for c in CELLS if c < Ti.shape[1]
        }
        floor = out["Ti_floor"]
        at_floor = Ti <= floor * (1.0 + 1e-9)
        out["frames_with_end_cell_at_floor"] = {
            c: int(np.count_nonzero(at_floor[:, c]))
            for c in CELLS if c < Ti.shape[1]
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pre")
    ap.add_argument("post")
    ap.add_argument("--window", nargs=2, type=float, default=(11.5, 12.24),
                    metavar="MS", help="census window [ms]")
    args = ap.parse_args()
    lo, hi = (1e-3 * v for v in args.window)
    for label, path in (("PRE  (zoh)", args.pre), ("POST (hold)", args.post)):
        c = census(path, lo, hi)
        print("=" * 78)
        print(f"{label}: {path}")
        for key, value in c.items():
            if key == "path":
                continue
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
