"""fa1 -- A1 anchored-fill READ instrument: arm vs REF neutral/fuel diagnostics.

READ-ONLY. Pure h5 reads of two saved runs; no solver is constructed, no
config is rebuilt, nothing in the repo is mutated. Written for the A1 read
because the fa0 probe it descends from lived in an ephemeral session
scratchpad (see fa0_neutral_budget.cmd: "Probe scripts live in the session
scratchpad (never the repo)") and no longer exists; the volume/inventory
conventions below are reconstructed to reproduce fa0_neutral_budget.txt on
sp1_ref.h5, which is the acceptance check for this rebuild.

Conventions (verified against sp1_ref.h5 geometry):
    V_col = geometry/plasma_volume_cm3      (nn lives here)
    V_ann = neutral_volume_cm3 - plasma_volume_cm3   (nn_a lives here)
Neutral inventory = sum(nn * V_col) + sum(nn_a * V_ann)   [particles]

Usage:
    python scripts/fa1_diag.py --arm ARM.h5 --ref REF.h5
"""

import argparse

import h5py
import numpy as np

# Axial windows in machine coordinates [cm]. The mid-column window is the
# sp1 leg-1 "required-source family" band (z ~ 790-1045); the source window
# covers the plenum/obstruction/cathode/gap/puff cluster.
SOURCE_Z = (-1e9, 100.0)
MID_Z = (790.0, 1045.0)
FAR_Z = (1045.0, 1900.0)

WINDOWS_MS = [
    ("whole run", None, None),
    ("discharge plateau", 5.0, 19.5),
    ("late plateau", 15.0, 19.5),
]

SNAP_MS = [5.0, 10.0, 19.0]


class Run:
    """Lazy handle on one saved run plus its derived volumes."""

    def __init__(self, path):
        self.path = path
        self.h5 = h5py.File(path, "r")
        g = self.h5["geometry"]
        self.z = g["z_cm"][:]
        self.V_col = g["plasma_volume_cm3"][:]
        self.V_ann = g["neutral_volume_cm3"][:] - self.V_col
        self.role = np.array([_s(x) for x in g["cell_role"][:]])
        self.t = self.h5["time"][:]
        self.t_ms = self.t * 1e3

    def close(self):
        self.h5.close()

    # -- inventories -----------------------------------------------------
    def neutral_inventory(self, mask=None):
        """Neutral particle inventory per saved sample [particles]."""
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        nn = self.h5["nn"][:, m]
        nna = self.h5["nn_a"][:, m]
        return nn @ self.V_col[m] + nna @ self.V_ann[m]

    def plasma_inventory(self, mask=None):
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        return self.h5["n"][:, m] @ self.V_col[m]

    def zmask(self, lo, hi):
        return (self.z >= lo) & (self.z <= hi)

    # -- rhs term volume integrals --------------------------------------
    def term_rate(self, term, channel, mask=None):
        """Volume-integrated rate of one rhs term/channel [particles/s]."""
        m = np.ones_like(self.z, dtype=bool) if mask is None else mask
        grp = self.h5["rhs_terms"][term]
        if channel not in grp:
            return None
        vol = self.V_col[m] if channel in ("n", "nn") else self.V_ann[m]
        return grp[channel][:, m] @ vol

    def neutral_term_rate(self, term, mask=None):
        """nn+nn_a volume-integrated rate of one rhs term [particles/s]."""
        a = self.term_rate(term, "nn", mask)
        b = self.term_rate(term, "nn_a", mask)
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return a + b

    def phase_events(self):
        ph = np.array([_s(x) for x in self.h5["phase"][:]])
        out = []
        prev = None
        for i, p in enumerate(ph):
            if p != prev:
                out.append((i, self.t_ms[i], p))
                prev = p
        return out


def _s(x):
    if isinstance(x, bytes):
        return x.decode()
    s = str(x)
    if s.startswith("b'") and s.endswith("'"):
        return s[2:-1]
    return s


def window_slice(run, lo_ms, hi_ms):
    if lo_ms is None:
        return slice(None)
    i0 = int(np.searchsorted(run.t_ms, lo_ms))
    i1 = int(np.searchsorted(run.t_ms, hi_ms))
    return slice(i0, i1 + 1)


def budget(run, label):
    print(f"\n### NEUTRAL BUDGET -- {label} ({run.path})")
    inv_n = run.neutral_inventory()
    inv_p = run.plasma_inventory()
    puff = run.h5["gas_puff_diagnostics"]["puff_particles_per_s"][:]
    sgp = run.h5["gas_puff_diagnostics"]["S_gp_sccm"][:]
    terms = list(run.h5["rhs_terms"].keys())

    for name, lo, hi in WINDOWS_MS:
        sl = window_slice(run, lo, hi)
        tt = run.t[sl]
        if tt.size < 2:
            continue
        span = tt[-1] - tt[0]
        lo_s = run.t_ms[sl][0]
        hi_s = run.t_ms[sl][-1]
        print(f"\n  === window {name}: {lo_s:.2f}-{hi_s:.2f} ms "
              f"({tt.size} samples) ===")
        print(f"    puff_particles_per_s   {puff[sl].mean():.4e}   "
              f"(S_gp {sgp[sl].mean():.0f} sccm mean, "
              f"{sgp[sl].max():.0f} peak)")
        d_n = (inv_n[sl][-1] - inv_n[sl][0]) / span
        d_p = (inv_p[sl][-1] - inv_p[sl][0]) / span
        print(f"    d/dt neutral inv       {d_n:.4e} /s   "
              f"(level {inv_n[sl].mean():.4e})")
        print(f"    d/dt plasma inv        {d_p:.4e} /s   "
              f"(level {inv_p[sl].mean():.4e})")
        print(f"    d/dt total inv         {d_n + d_p:.4e} /s")
        rows = []
        for term in terms:
            r = run.neutral_term_rate(term)
            if r is None:
                continue
            v = float(r[sl].mean())
            if abs(v) > 1e17:
                rn = run.term_rate(term, "n")
                rows.append((abs(v), term, v,
                             float(rn[sl].mean()) if rn is not None else 0.0))
        rows.sort(reverse=True)
        print("    -- per-term neutral inventory rates (particles/s), "
              "|mean| > 1e17 --")
        for _, term, v, vn in rows:
            print(f"      {term:32s} nn+nn_a {v:11.4e}   n {vn:11.4e}")
        src = run.neutral_term_rate("neutral_sources")
        src_m = float(src[sl].mean())
        pump = src_m - puff[sl].mean()
        print(f"    neutral_sources total    {src_m:.4e}")
        print(f"    implied PUMP rate        {pump:.4e}   = "
              f"{100 * abs(pump) / puff[sl].mean():.1f}% of puff")


def seed_check(arm, ref):
    print("\n### 7. EQUILIBRATED-SEED LINEARITY CHECK (t = 0 initial state)")
    for run, tag in ((ref, "REF "), (arm, "ARM ")):
        nn0 = run.h5["nn"][0]
        nna0 = run.h5["nn_a"][0]
        inv = float(nn0 @ run.V_col + nna0 @ run.V_ann)
        print(f"  {tag} seed: nn min/max {nn0.min():.4e} / {nn0.max():.4e} "
              f"cm^-3 | inventory {inv:.6e} particles")
        run._seed_inv = inv
        run._seed_nn = nn0
    ratio = arm._seed_inv / ref._seed_inv
    linear = 9010.0 / 5200.0
    print(f"  arm/REF seed inventory ratio = {ratio:.5f}")
    print(f"  linear expectation S_gp ratio = {linear:.5f}")
    print(f"  DEVIATION FROM LINEAR = {100 * (ratio / linear - 1):+.3f} %")
    # per-region seed ratio
    for name, (lo, hi) in (("source z<=100", SOURCE_Z),
                           ("mid 790-1045", MID_Z),
                           ("far 1045-1900", FAR_Z)):
        ma = arm.zmask(*(lo, hi))
        mr = ref.zmask(*(lo, hi))
        ia = float(arm.h5["nn"][0][ma] @ arm.V_col[ma]
                   + arm.h5["nn_a"][0][ma] @ arm.V_ann[ma])
        ir = float(ref.h5["nn"][0][mr] @ ref.V_col[mr]
                   + ref.h5["nn_a"][0][mr] @ ref.V_ann[mr])
        print(f"    {name:16s} arm/REF = {ia / ir:.5f}")


def axial_excess(arm, ref):
    print("\n### 6b. AXIAL nn(z) AT MATCHED TIMES -- ARM vs REF, and the EXCESS")
    print("  (nn = column-zone neutral density [cm^-3]; excess = arm - REF)")
    for t_ms in SNAP_MS:
        ia = int(np.argmin(np.abs(arm.t_ms - t_ms)))
        ir = int(np.argmin(np.abs(ref.t_ms - t_ms)))
        na = arm.h5["nn"][ia]
        nr = ref.h5["nn"][ir]
        naa = arm.h5["nn_a"][ia]
        nra = ref.h5["nn_a"][ir]
        # per-cell excess INVENTORY (particles), both zones
        exc = (na - nr) * arm.V_col + (naa - nra) * arm.V_ann
        tot = exc.sum()
        print(f"\n  --- t = {t_ms:.2f} ms (arm sample {ia} @ "
              f"{arm.t_ms[ia]:.3f} ms, REF sample {ir} @ "
              f"{ref.t_ms[ir]:.3f} ms) ---")
        print(f"    total excess neutral inventory = {tot:.4e} particles")
        print(f"    {'z [cm]':>9} {'nn_ARM':>11} {'nn_REF':>11} "
              f"{'ratio':>7} {'excess_inv':>12} {'cum%':>7}")
        cum = 0.0
        for i in range(len(arm.z)):
            zz = arm.z[i]
            if not (i < 12 or (i - 12) % 18 == 0 or i >= len(arm.z) - 3):
                continue
            cum_here = exc[:i + 1].sum()
            print(f"    {zz:9.1f} {na[i]:11.4e} {nr[i]:11.4e} "
                  f"{na[i] / nr[i]:7.3f} {exc[i]:12.4e} "
                  f"{100 * cum_here / tot:7.2f}")
        # regional split of the excess
        print("    -- regional split of the EXCESS inventory --")
        for name, (lo, hi) in (("source z<=100", SOURCE_Z),
                               ("100<z<790", (100.0, 790.0)),
                               ("mid 790-1045", MID_Z),
                               ("far 1045-1900", FAR_Z),
                               ("end z>1900", (1900.0, 1e9))):
            m = arm.zmask(lo, hi)
            print(f"      {name:16s} {exc[m].sum():11.4e}  "
                  f"({100 * exc[m].sum() / tot:6.2f}% of excess)")
        # centroid of excess vs centroid of the REF fill
        w = np.clip(exc, 0, None)
        zc_exc = float((arm.z * w).sum() / w.sum())
        base = nr * ref.V_col + nra * ref.V_ann
        zc_ref = float((ref.z * base).sum() / base.sum())
        print(f"    centroid of POSITIVE excess  z = {zc_exc:8.1f} cm")
        print(f"    centroid of REF fill         z = {zc_ref:8.1f} cm")


def region_trajectories(arm, ref):
    print("\n### 6a/6c. REGIONAL nn(t) TRAJECTORIES AND IONIZATION SOURCE")
    regions = (("SOURCE z<=100", SOURCE_Z),
               ("MID 790-1045", MID_Z),
               ("FAR 1045-1900", FAR_Z))
    for name, (lo, hi) in regions:
        ma = arm.zmask(lo, hi)
        mr = ref.zmask(lo, hi)
        inv_a = arm.neutral_inventory(ma)
        inv_r = ref.neutral_inventory(mr)
        # volume-mean nn over the region (column zone)
        Va = arm.V_col[ma].sum() + arm.V_ann[ma].sum()
        Vr = ref.V_col[mr].sum() + ref.V_ann[mr].sum()
        print(f"\n  --- {name} ---")
        print(f"    {'t [ms]':>8} {'nn_ARM':>11} {'nn_REF':>11} {'ratio':>7} "
              f"{'ioniz_ARM':>12} {'ioniz_REF':>12} {'i_ratio':>8}")
        ion_a = arm.term_rate("ionization_birth", "n", ma)
        ion_r = ref.term_rate("ionization_birth", "n", mr)
        for t_ms in (1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 17.5, 19.0, 20.0):
            ia = int(np.argmin(np.abs(arm.t_ms - t_ms)))
            ir = int(np.argmin(np.abs(ref.t_ms - t_ms)))
            va = inv_a[ia] / Va
            vr = inv_r[ir] / Vr
            print(f"    {t_ms:8.2f} {va:11.4e} {vr:11.4e} {va / vr:7.3f} "
                  f"{ion_a[ia]:12.4e} {ion_r[ir]:12.4e} "
                  f"{ion_a[ia] / ion_r[ir]:8.3f}")


def health(run, label):
    print(f"\n### 8. RUN HEALTH -- {label} ({run.path})")
    print(f"  attrs: {dict((k, v) for k, v in run.h5.attrs.items() if k not in ('params_json', 'flags_json'))}")
    d = run.h5["diagnostics"]
    dt = d["accepted_dt"][:]
    print(f"  accepted steps = {dt.size}")
    print(f"  dt  min {dt.min():.4e}  median {np.median(dt):.4e}  "
          f"max {dt.max():.4e}")
    clamp = d["clamped_to_dt_min"][:]
    print(f"  steps clamped to dt_min: {int(clamp.sum())} "
          f"({100 * clamp.mean():.3f}%)")
    ac = np.array([_s(x) for x in d["active_constraint"][:]])
    vals, cnts = np.unique(ac, return_counts=True)
    order = np.argsort(-cnts)
    print("  active timestep constraint census:")
    for i in order[:10]:
        print(f"    {vals[i]:28s} {cnts[i]:8d}  ({100 * cnts[i] / ac.size:5.2f}%)")
    rr = np.array([_s(x) for x in d["rejection_reason"][:]])
    vals, cnts = np.unique(rr, return_counts=True)
    print("  rejection reasons:")
    for v, c in zip(vals, cnts):
        print(f"    {v:28s} {c:8d}")
    for name in ("n", "nn", "nn_a", "Te", "Ti"):
        arr = run.h5[name]
        chunk = arr[::37]
        print(f"  finiteness {name:5s}: all finite = "
              f"{bool(np.isfinite(chunk).all())}  "
              f"min {np.nanmin(chunk):.4e} max {np.nanmax(chunk):.4e}")
    print("  phase events:")
    for i, t_ms, p in run.phase_events():
        print(f"    sample {i:6d}  t = {t_ms:9.4f} ms   {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--ref", required=True)
    args = ap.parse_args()
    arm = Run(args.arm)
    ref = Run(args.ref)
    print("=" * 78)
    print("fa1 A1 ANCHORED-FILL READ -- neutral/fuel diagnostics, arm vs REF")
    print(f"  ARM = {args.arm}")
    print(f"  REF = {args.ref}")
    print("=" * 78)
    seed_check(arm, ref)
    region_trajectories(arm, ref)
    axial_excess(arm, ref)
    budget(ref, "REF")
    budget(arm, "ARM")
    health(ref, "REF")
    health(arm, "ARM")
    arm.close()
    ref.close()


if __name__ == "__main__":
    main()
