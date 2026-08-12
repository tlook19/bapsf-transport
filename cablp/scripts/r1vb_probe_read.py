"""R1 ON-path acceptance read over the bounded conducting-phase probe(s).

Every assertion the R1 brief pre-registered, evaluated from the SAVED cathode
diagnostics of one or more probe runs:

  A1  V_b <= the circuit-available voltage at every solve where the bound was
      in force, and reported (never asserted) over all solves -- the two are
      different statements and the difference is the inductive back-EMF the
      bound deliberately does not count as supply.
  A2  beam birth energy (= the returned phi_c) <= ~180 eV throughout the build.
  A3  the active-bound census is populated (0 none / 1 data cap / 2 circuit).
  A4  zero escaped solves: phi_c never above the composed ceiling.

plus the REPORTED, non-gating observables: the build-leg e-fold (estimator
carried over verbatim from ``covcal_efold_read.tau_us``) and an F4-style axial
flatness read of the beam deposition.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/r1vb_probe_read.py r1vb_probe_off r1vb_probe_on
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
EPS = 1e-9  # relative slack on the voltage comparisons (float assembly noise)


def sdec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def load(stem):
    path = HERE / f"{stem}.h5"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    d = {"stem": stem}
    with h5py.File(path, "r") as f:
        d["params"] = json.loads(f.attrs["params_json"])
        d["flags"] = json.loads(f.attrs["flags_json"])
        g = f["geometry"]
        act = np.asarray(g["plasma_active"], bool)
        d["active"] = act
        d["V"] = np.asarray(g["plasma_volume_cm3"], float)
        d["length"] = np.asarray(g["length_cm"], float)
        d["t"] = np.asarray(f["time"], float)
        d["meann"] = np.asarray(f["n"], float)[:, act].mean(axis=1)
        d["Te_mean"] = np.asarray(f["Te"], float)[:, act].mean(axis=1)
        ev = f["phase_events"]
        d["ev_t"] = np.asarray(ev["time"], float)
        d["ev_phase"] = [sdec(v) for v in np.asarray(ev["phase"])]
        cd = f["cathode_diagnostics"]
        d["regime"] = [sdec(v) for v in np.asarray(cd["source_regime"])]
        for k in (
            "source_phi_c", "source_V_b", "source_V_p", "source_phi_a",
            "source_I_tot", "source_phi_c_at_cap", "source_phi_c_ceiling_V",
            "source_circuit_V_avail_V", "source_bound_active",
            "circuit_I_loop", "circuit_V_dis_step", "has_solution",
        ):
            if k in cd:
                d[k] = np.asarray(cd[k], float)
        if "beam_heat_anomalous_W" in cd:
            d["beam_heat"] = sum(
                np.asarray(cd[b], float)
                for b in ("beam_heat_coulomb_W", "beam_heat_anomalous_W",
                          "beam_heat_secondary_W", "beam_heat_terminal_W")
                if b in cd
            )
    return d


def leg(d):
    """The build leg (a, b): 0 < t <= breakdown, or the whole window."""
    t = d["t"]
    bd = [float(w) for w, p in zip(d["ev_t"], d["ev_phase"])
          if p == "breakdown"]
    t_bd = bd[0] if bd else None
    pre = (np.flatnonzero((t > 0) & (t <= t_bd)) if t_bd is not None
           else np.flatnonzero(t > 0))
    return int(pre[0]), int(pre[-1]), t_bd


def tau_us(d, a, b):
    m, t = d["meann"], d["t"]
    return 1.0e6 * (t[b] - t[a]) / np.log(m[b] / m[a])


def report(d):
    print("=" * 78)
    bound = bool(d["flags"].get("cathode_circuit_voltage_bound", False))
    print(f"{d['stem']}   cathode_circuit_voltage_bound = {bound}")
    print("=" * 78)
    t = d["t"]
    live = np.asarray(d.get("has_solution", np.ones_like(t)), float) > 0.0
    a, b, t_bd = leg(d)
    print(f"  window 0 -> {t[-1] * 1e3:.6f} ms, {t.size} saves, "
          f"{int(live.sum())} with a solve; "
          f"breakdown {'at %.6f ms' % (t_bd * 1e3) if t_bd else 'not reached'}")

    failures = []
    have_census = "source_bound_active" in d
    if not have_census:
        print("  NO CENSUS DATASETS -- run predates R1")
        return ["no census datasets"]

    code = d["source_bound_active"]
    ceil = d["source_phi_c_ceiling_V"]
    avail = d["source_circuit_V_avail_V"]
    phi = d["source_phi_c"]
    V_b = d["source_V_b"]
    sel = live & np.isfinite(code)

    # A3 -- census populated.
    counts = {int(c): int(np.sum(code[sel] == c)) for c in (0.0, 1.0, 2.0)}
    print(f"  A3 census over {int(sel.sum())} solves: "
          f"none={counts[0]}  data_cap={counts[1]}  circuit={counts[2]}")
    if not np.all(np.isin(code[sel], (0.0, 1.0, 2.0))):
        failures.append("A3: census carries a code outside {0,1,2}")
    if bound and counts[2] == 0 and counts[1] == 0:
        print("  A3 NOTE: the ceiling never bound on a SAVED frame")

    # A4 -- zero escaped solves against the COMPOSED ceiling.
    esc = sel & (phi > ceil * (1.0 + EPS))
    print(f"  A4 escaped solves (phi_c above the composed ceiling): "
          f"{int(esc.sum())}")
    if np.any(esc):
        failures.append(f"A4: {int(esc.sum())} escaped solves")

    # A2 -- beam birth energy.
    print(f"  A2 phi_c over the build leg: max {np.nanmax(phi[sel]):.4f} V, "
          f"median {np.nanmedian(phi[sel]):.4f} V   "
          f"(ceiling median {np.nanmedian(ceil[sel]):.4f} V)")
    if bound and np.nanmax(phi[sel]) > 180.0 * (1.0 + EPS):
        failures.append(
            f"A2: phi_c reached {np.nanmax(phi[sel]):.4f} V > 180 V"
        )

    # A1 -- V_b vs the circuit-available voltage.
    if bound:
        ok_all = sel & np.isfinite(avail)
        viol_all = ok_all & (V_b > avail * (1.0 + EPS))
        bnd = ok_all & (code == 2.0)
        viol_bnd = bnd & (V_b > avail * (1.0 + EPS))
        print(f"  A1 V_b <= V_avail: violations {int(viol_all.sum())}/"
              f"{int(ok_all.sum())} over ALL solves, "
              f"{int(viol_bnd.sum())}/{int(bnd.sum())} over BOUND solves")
        if np.any(viol_bnd):
            failures.append(
                f"A1: {int(viol_bnd.sum())} bound solves above V_avail"
            )
        if np.any(viol_all):
            worst = np.nanmax((V_b / avail)[viol_all])
            idx = np.flatnonzero(viol_all)
            print(f"     REPORTED (not gated): worst V_b/V_avail = "
                  f"{worst:.4f} on an unbounded solve; regimes "
                  f"{sorted({d['regime'][i] for i in idx})}")
    ratio = np.full(t.shape, np.nan)
    vdis = d.get("circuit_V_dis_step")
    if vdis is not None:
        good = sel & (np.abs(vdis) > 1e-12)
        ratio[good] = V_b[good] / vdis[good]
        if np.any(good):
            print(f"  V_b/V_dis over the build leg: median "
                  f"{np.nanmedian(ratio[good]):.4f}, max "
                  f"{np.nanmax(ratio[good]):.4f}  (R0 measured 5.09-5.64 "
                  f"median unbounded)")

    # REPORTED: build-leg e-fold.
    if b > a and d["meann"][b] > 0 and d["meann"][a] > 0:
        print(f"  e-fold over the build leg [{t[a] * 1e3:.6f} -> "
              f"{t[b] * 1e3:.6f} ms]: tau = {tau_us(d, a, b):.4f} us   "
              f"(<n> {d['meann'][a]:.6e} -> {d['meann'][b]:.6e} cm^-3)")
    # REPORTED: F4-style axial flatness of the beam deposition.
    heat = d.get("beam_heat")
    if heat is not None and heat.ndim == 2:
        act = d["active"]
        rows = np.flatnonzero(heat[:, act].sum(axis=1) > 0.0)
        if rows.size:
            i = int(rows[-1])
            row = heat[i][act]
            print(f"  F4-style flatness at t={t[i] * 1e3:.6f} ms: "
                  f"min/max = {row.min() / row.max():.4f} over "
                  f"{row.size} active cells")
        else:
            print("  F4-style flatness: no beam power deposited in the window")
    print()
    return failures


def main(argv=None):
    stems = list(argv if argv is not None else sys.argv[1:])
    if not stems:
        raise SystemExit(__doc__)
    failures = []
    runs = {}
    for stem in stems:
        d = load(stem)
        runs[stem] = d
        failures += [f"{stem}: {f}" for f in report(d)]

    on = [d for d in runs.values()
          if d["flags"].get("cathode_circuit_voltage_bound", False)]
    off = [d for d in runs.values()
           if not d["flags"].get("cathode_circuit_voltage_bound", False)]
    if on and off:
        print("=" * 78)
        print("PRE-REGISTERED DIRECTION (reported, not gated): the drive-power")
        print("correction x0.177-0.20 should SLOW the pedestal build.")
        print("=" * 78)
        for d in (off[0], on[0]):
            a, b, _ = leg(d)
            tag = "ON " if d["flags"].get(
                "cathode_circuit_voltage_bound", False) else "OFF"
            if b > a:
                print(f"  {tag}  tau = {tau_us(d, a, b):10.4f} us   "
                      f"<n>(end) = {d['meann'][b]:.6e} cm^-3   "
                      f"t_end = {d['t'][b] * 1e3:.6f} ms")
        # Matched-time comparison: same save index, so the two are read at the
        # same elapsed time rather than at whatever each window reached.
        k = min(off[0]["t"].size, on[0]["t"].size) - 1
        if k > 0 and abs(off[0]["t"][k] - on[0]["t"][k]) < 1e-12:
            r = on[0]["meann"][k] / off[0]["meann"][k]
            print(f"  matched-time <n> ratio ON/OFF at t="
                  f"{on[0]['t'][k] * 1e3:.6f} ms: {r:.6f} "
                  f"({'SLOWER' if r < 1.0 else 'FASTER -- STOP AND REPORT'})")
        else:
            print("  matched-time comparison unavailable (save grids differ)")
        print()

    print("=" * 78)
    if failures:
        print(f"R1 ON-PATH ACCEPTANCE: FAIL -- {len(failures)} problem(s)")
        for f in failures:
            print(f"  {f}")
        return 1
    print("R1 ON-PATH ACCEPTANCE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
