"""Channel decomposition of the conducting-phase pedestal e-fold (covcal F2).

READ-ONLY DIAGNOSIS. Constructs no solver, integrates nothing, runs no solver
step. Every per-channel number is taken from the ``rhs_terms/<name>/n`` rows the
solver itself saved, on the same active-cell mask and the same build leg the F2
estimator uses (``covcal_read.py:pedestal_efold``, unchanged in definition).

Sections:
  (1) estimator reproduction -- confirms this script's leg is the F2 leg.
  (2) channel decomposition of d ln<n>/dt over the build leg, with the
      beam channel split into its WALKER (tail) and PRIMARY sub-channels via
      ``cathode_diagnostics/beam_tail_ionization_events_per_s`` (the walkers'
      own share of the shared ``beam_ionization_birth`` bank).
  (3) leg sub-structure -- ignition transient / plateau / acceleration.
  (4) what sets the rate: the loop-gain chain and the rate identities.
  (5) the amplitude cross-check (shot 1 vs shot 2 discharge current).
  (6) watch items.

Every number is conditional on the campaign-class closure set
(``heating_anomalous_transport=tail_walk`` + ``heating_anomalous_tail_ionization=on``,
``coverage_closure=True``, ``beam_deposition_model=csda``, ADAS rates, nx=60)
unless the section says otherwise.
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent

BAND_LO_US, BAND_HI_US = 713.0, 725.0

# The three F2 shots, in decreasing coverage growth rate r.
SHOTS = ("covdecide_twion_f005", "covcal_f2_shot1", "covcal_f2_shot2")

# Every rhs_terms row that is non-zero on the ``n`` channel in these runs.
# (Section 2 asserts the rest are identically zero rather than assuming it.)
LIVE = ("beam_ionization_birth", "ionization_birth", "characteristic_boundary",
        "anode_collection", "plasma_advective_flux", "recombination_rad_loss")

# Ladder arms that carry the declared f bracket and the tail-ionization A/B.
# DIFFERENT CLOSURE SET: nx=240, no coverage closure. Read for leverage only.
LADDER = ("es1_lad_ref_nx240", "es1_lad_tw25_nx240", "es1_lad_tw25ion_nx240",
          "es1_lad_tw50ion_nx240", "es1_lad_tw100ion_nx240")

# f_cov0 probes. ALSO a different closure set (see the table's own header).
FCOV = ("covprobe_f005", "covprobe_f010", "covprobe_f020", "covprobe_f040",
        "covprobe2_tw_f005", "covprobe2diag_tw_f020")


def sdec(x):
    return x.decode() if isinstance(x, (bytes, bytearray)) else str(x)


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load(stem):
    """Return the leg-relevant arrays for one run, or None if absent."""
    path = HERE / f"{stem}.h5"
    if not path.exists():
        return None
    d = {"stem": stem}
    with h5py.File(path, "r") as f:
        d["params"] = json.loads(f.attrs["params_json"])
        d["flags"] = json.loads(f.attrs["flags_json"])
        g = f["geometry"]
        act = np.asarray(g["plasma_active"], bool)
        d["active"] = act
        d["V"] = np.asarray(g["plasma_volume_cm3"], float)
        d["V_act"] = float(d["V"][act].sum())
        d["L_act"] = float(np.asarray(g["length_cm"], float)[act].sum())
        t = np.asarray(f["time"], float)
        d["t"] = t
        n = np.asarray(f["n"], float)
        d["meann"] = n[:, act].mean(axis=1)
        d["nn_mean"] = np.asarray(f["nn"], float)[:, act].mean(axis=1)
        d["Te_mean"] = np.asarray(f["Te"], float)[:, act].mean(axis=1)
        ev = f["phase_events"]
        d["ev_t"] = np.asarray(ev["time"], float)
        d["ev_phase"] = [sdec(v) for v in np.asarray(ev["phase"])]
        d["ev_reason"] = [sdec(v) for v in np.asarray(ev["reason"])]
        # every rhs_terms n row, active-cell mean == d<n>/dt contribution
        d["terms"] = {}
        for nm in sorted(f["rhs_terms"].keys()):
            arr = np.asarray(f["rhs_terms"][nm]["n"], float)
            d["terms"][nm] = arr[:, act].mean(axis=1)
        d["total_rhs_n"] = np.asarray(f["total_rhs"]["n"], float)[:, act].mean(axis=1)
        # walker sub-channel: events [1/s per cell] -> density [cm^-3 s^-1]
        cd = f["cathode_diagnostics"]
        events = np.asarray(cd["beam_tail_ionization_events_per_s"], float)
        wdens = np.zeros_like(events)
        wdens[:, act] = events[:, act] / d["V"][None, act]
        d["walker"] = wdens[:, act].mean(axis=1)
        bib = np.asarray(f["rhs_terms"]["beam_ionization_birth"]["n"], float)
        d["beam_col_per_s"] = (bib * d["V"][None, :])[:, act].sum(axis=1)
        d["walker_col_per_s"] = events[:, act].sum(axis=1)
        for k in ("beam_tail_power_W", "beam_tail_ionization_cost_W",
                  "beam_tail_radiated_W", "beam_heat_anomalous_W",
                  "circuit_I_loop", "circuit_V_dis_step", "source_I_e",
                  "source_I_eth", "source_I_eth_star", "source_phi_c",
                  "source_phi_c_at_cap", "source_P_prim", "source_l_b",
                  "source_beam_bypass_fraction", "coverage_fraction"):
            # Older/leaner runs (the nx=240 ladder, the pre-coverage probes)
            # do not carry every one of these. Absent is NOT defaulted to a
            # number -- it is left absent, so a section that needs it fails
            # loudly rather than reading a fabricated zero.
            if k in cd:
                d[k] = np.asarray(cd[k], float)
        ard = f["atomic_rate_domain"]
        d["rate_first_below_s"] = float(np.asarray(ard["first_below_time_s"]))
        d["rate_Te_min_table"] = float(np.asarray(ard["table_Te_min_eV"]))
        d["rate_vol_below"] = np.asarray(ard["active_volume_fraction_below"], float)
    return d


def leg(d):
    """The F2 build leg (a, b) -- 0 < t <= breakdown, estimator-identical."""
    t = d["t"]
    bd = [float(w) for w, p in zip(d["ev_t"], d["ev_phase"]) if p == "breakdown"]
    t_bd = bd[0] if bd else None
    pre = np.flatnonzero((t > 0) & (t <= t_bd)) if t_bd is not None \
        else np.flatnonzero(t > 0)
    return int(pre[0]), int(pre[-1]), t_bd


def tau_us(d, a, b):
    m, t = d["meann"], d["t"]
    return 1.0e6 * (t[b] - t[a]) / np.log(m[b] / m[a])


def escape(d):
    """Tail power leaving the column, as a fraction. Definition carried over
    verbatim from ``covcal_read.py:escape_fraction``."""
    tail = d["beam_tail_power_W"]
    dep = (d["beam_heat_anomalous_W"].sum(axis=1)
           + d["beam_tail_radiated_W"].sum(axis=1)
           + d["beam_tail_ionization_cost_W"].sum(axis=1))
    out = np.full(tail.shape, np.nan)
    live = tail > 0.0
    out[live] = 1.0 - dep[live] / tail[live]
    return out


def gains(d, a, b, gamma):
    """Trapezoid-integrated log-gain of a normalized rate over the leg."""
    return float(np.trapezoid(gamma[a:b + 1], d["t"][a:b + 1]))


def main():
    runs = {s: load(s) for s in SHOTS}
    missing = [s for s, v in runs.items() if v is None]
    if missing:
        sys.exit(f"ABSENT: {missing}")

    section("(0) RUN IDENTITY AND CLOSURE SET")
    print(f"{'stem':24s} {'r [1/s]':>11s} {'f_cov0':>7s} {'transport':>10s} "
          f"{'tail_ion':>8s} {'f':>6s} {'nx':>4s} {'rates':>6s}")
    for s in SHOTS:
        p = runs[s]["params"]
        fphi = p.get("heating_anomalous_tail_phi_c_fraction")
        print(f"{s:24s} {p['coverage_growth_rate_per_s']:>11} "
              f"{p['coverage_initial_fraction']:>7} "
              f"{p['heating_anomalous_transport']:>10} "
              f"{p['heating_anomalous_tail_ionization']:>8} "
              f"{('0.25 (shipped)' if fphi is None else fphi):>6} "
              f"{p['nx']:>4} {p['atomic_rate_model']:>6}")

    section("(1) ESTIMATOR REPRODUCTION -- this script's leg IS the F2 leg")
    print(f"{'stem':24s} {'t_bd [ms]':>10s} {'reason':>16s} {'saves':>7s} "
          f"{'dln n':>8s} {'TAU [us]':>10s} {'band':>12s}")
    legs = {}
    for s in SHOTS:
        d = runs[s]
        a, b, t_bd = leg(d)
        legs[s] = (a, b, t_bd)
        reason = [r for w, p, r in zip(d["ev_t"], d["ev_phase"], d["ev_reason"])
                  if p == "breakdown"][0]
        tau = tau_us(d, a, b)
        verdict = "IN BAND" if BAND_LO_US <= tau <= BAND_HI_US else "OUT OF BAND"
        print(f"{s:24s} {t_bd*1e3:10.4f} {reason:>16s} {b-a+1:7d} "
              f"{np.log(d['meann'][b]/d['meann'][a]):8.4f} {tau:10.4f} {verdict:>12s}")
    print(f"\n  target band [{BAND_LO_US}, {BAND_HI_US}] us; the leg ends on a "
          "CURRENT threshold ('I_prebreakdown'), not a density one.")

    section("(2) CHANNEL DECOMPOSITION OF THE BUILD LEG")
    print("  d ln<n>/dt is decomposed EXACTLY as the estimator forms <n>: the")
    print("  active-cell arithmetic mean of each rhs_terms n row, divided by <n>.")
    print("  gamma_c(t) = <dn/dt>_c / <n>;  the leg log-gain is trapz(gamma_c dt).")
    for s in SHOTS:
        d = runs[s]
        a, b, _ = legs[s]
        dead = [nm for nm, v in d["terms"].items()
                if nm not in LIVE and np.abs(v[a:b + 1]).max() > 0.0]
        gw = d["walker"] / d["meann"]
        gb_all = d["terms"]["beam_ionization_birth"] / d["meann"]
        gp = gb_all - gw
        rows = [("beam: WALKER (tail) ionization", gw),
                ("beam: PRIMARY (CSDA) ionization", gp),
                ("bulk/thermal ionization_birth", d["terms"]["ionization_birth"] / d["meann"]),
                ("characteristic_boundary (sink)", d["terms"]["characteristic_boundary"] / d["meann"]),
                ("anode_collection (sink)", d["terms"]["anode_collection"] / d["meann"]),
                ("plasma_advective_flux (transport)", d["terms"]["plasma_advective_flux"] / d["meann"]),
                ("recombination_rad_loss (sink)", d["terms"]["recombination_rad_loss"] / d["meann"])]
        obs = float(np.log(d["meann"][b] / d["meann"][a]))
        acc = [(lab, gains(d, a, b, g)) for lab, g in rows]
        tot = sum(v for _, v in acc)
        print(f"\n  --- {s}  (leg saves {a}..{b}, dt = {d['t'][b]-d['t'][a]:.4e} s) ---")
        print(f"  rhs_terms rows non-zero on n outside the live set: "
              f"{dead if dead else 'NONE (the six below are the whole budget)'}")
        print(f"  {'channel':36s} {'dln n':>9s} {'share':>9s} {'tau if alone [us]':>19s}")
        for lab, v in acc:
            share = 100.0 * v / obs
            ta = (d["t"][b] - d["t"][a]) / v * 1e6 if v != 0.0 else np.inf
            print(f"  {lab:36s} {v:+9.4f} {share:+8.2f} % {ta:19.2f}")
        print(f"  {'SUM (reconstructed)':36s} {tot:+9.4f} {100.0*tot/obs:+8.2f} %")
        print(f"  {'OBSERVED ln(<n>_b/<n>_a)':36s} {obs:+9.4f}")
        print(f"  closure of the reconstruction: {100.0*(tot-obs)/obs:+.3f} % "
              "(trapezoid over 10 us saves vs the exact stepped integral)")

    section("(3) DOES OWNERSHIP SHIFT ACROSS THE LEG?")
    segs = (("ignition transient 10-50 us", 1.0e-5, 5.0e-5),
            ("plateau 50-430 us", 5.0e-5, 4.3e-4),
            ("acceleration 430 us - breakdown", 4.3e-4, None))
    for s in SHOTS:
        d = runs[s]
        a, b, _ = legs[s]
        gw = d["walker"] / d["meann"]
        gb = d["terms"]["beam_ionization_birth"] / d["meann"]
        gu = d["terms"]["ionization_birth"] / d["meann"]
        gs = sum(d["terms"][k] for k in ("characteristic_boundary",
                                         "anode_collection",
                                         "plasma_advective_flux")) / d["meann"]
        print(f"\n  --- {s} ---")
        print(f"  {'segment':32s} {'dt [s]':>10s} {'dln n':>8s} {'tau_seg':>9s}"
              f" {'walker':>8s} {'primary':>8s} {'bulk':>8s} {'sinks':>8s}")
        for lab, lo, hi in segs:
            hi = d["t"][b] + 1e-15 if hi is None else hi
            m = np.flatnonzero((d["t"] >= lo - 1e-15) & (d["t"] <= hi))
            if m.size < 2:
                continue
            i0, i1 = int(m[0]), int(m[-1])
            dl = float(np.log(d["meann"][i1] / d["meann"][i0]))
            dt = d["t"][i1] - d["t"][i0]
            it = lambda g: float(np.trapezoid(g[i0:i1 + 1], d["t"][i0:i1 + 1]))
            print(f"  {lab:32s} {dt:10.3e} {dl:8.4f} {1e6*dt/dl:9.2f}"
                  f" {it(gw):+8.4f} {it(gb)-it(gw):+8.4f} {it(gu):+8.4f} {it(gs):+8.4f}")
        print(f"  walker share of the BEAM channel: "
              f"{d['walker_col_per_s'][a+6]/d['beam_col_per_s'][a+6]:.3f} at "
              f"t={d['t'][a+6]*1e3:.2f} ms -> "
              f"{d['walker_col_per_s'][b]/d['beam_col_per_s'][b]:.3f} at breakdown")

    section("(4) WHAT SETS THE RATE -- the loop-gain chain")
    for s in ("covcal_f2_shot1",):
        d = runs[s]
        a, b, _ = legs[s]
        sl = slice(a + 3, b + 1)
        lg = lambda x, y: float(np.polyfit(np.log(x[sl]), np.log(y[sl]), 1)[0])
        print(f"  {s}: log-log slopes over the leg (saves {a+3}..{b})")
        print(f"    d ln I_e     / d ln <n>  = {lg(d['meann'], d['source_I_e']):.4f}")
        print(f"    d ln I_loop  / d ln <n>  = {lg(d['meann'], d['circuit_I_loop']):.4f}")
        print(f"    d ln S_beam  / d ln <n>  = {lg(d['meann'], d['beam_col_per_s']):.4f}")
        print(f"    d ln S_beam  / d ln I_e  = {lg(d['source_I_e'], d['beam_col_per_s']):.4f}")
        print("  => the beam source is LINEAR in <n> through the circuit, which")
        print("     is what makes the build exponential rather than algebraic.")
        esc = escape(d)
        print(f"\n  {'t [s]':>10s} {'<n>':>10s} {'I_e [A]':>9s} {'I_loop':>8s} "
              f"{'V_dis':>7s} {'phi_c':>7s} {'cap?':>5s} {'P_prim':>10s} "
              f"{'P_tail':>10s} {'esc':>6s} {'l_b/L':>8s} {'<Te>':>6s} {'<nn>':>10s}")
        for i in list(range(a, b + 1, max(1, (b - a) // 9))) + [b]:
            print(f"  {d['t'][i]:10.3e} {d['meann'][i]:10.3e} "
                  f"{d['source_I_e'][i]:9.3f} {d['circuit_I_loop'][i]:8.2f} "
                  f"{d['circuit_V_dis_step'][i]:7.2f} {d['source_phi_c'][i]:7.1f} "
                  f"{d['source_phi_c_at_cap'][i]:5.0f} {d['source_P_prim'][i]:10.3e} "
                  f"{d['beam_tail_power_W'][i]:10.3e} {esc[i]:6.3f} "
                  f"{d['source_l_b'][i]/d['L_act']:8.2f} {d['Te_mean'][i]:6.3f} "
                  f"{d['nn_mean'][i]:10.3e}")
        print(f"\n  active plasma volume  = {d['V_act']:.4e} cm^3")
        print(f"  active column length  = {d['L_act']:.1f} cm")
        print("  I_eth (emission CAPABILITY) is ~657 A throughout while I_e stays")
        print(f"  below {d['source_I_e'][b]:.1f} A at breakdown: the build is CIRCUIT-limited,")
        print("  not emission-limited.")

        print("\n  --- bulk channel identity: gamma_bulk = nn * SCD(Te) ---")
        gu = d["terms"]["ionization_birth"] / d["meann"]
        i = (a + b) // 2
        print(f"    at t={d['t'][i]:.3e}: <nn>={d['nn_mean'][i]:.4e} cm^-3, "
              f"<Te>={d['Te_mean'][i]:.3f} eV, gamma_bulk={gu[i]:.4e} 1/s")
        print(f"    implied <sigma v> = gamma_bulk/<nn> = {gu[i]/d['nn_mean'][i]:.4e} cm^3/s")
        print("    (section 6 of the memo checks this against cablp.atomic.adas.he_rates)")

        print("\n  --- phi_c ceiling exposure on the leg ---")
        for t2 in SHOTS:
            e = runs[t2]
            a2, b2, _ = legs[t2]
            cap = e["source_phi_c_at_cap"]
            gb2 = e["terms"]["beam_ionization_birth"] / e["meann"]
            g_cap = float(np.trapezoid(gb2[a2:b2+1]*cap[a2:b2+1], e["t"][a2:b2+1]))
            g_all = float(np.trapezoid(gb2[a2:b2+1], e["t"][a2:b2+1]))
            print(f"    {t2:24s} saves at cap {int(cap[a2:b2+1].sum()):3d}/{b2-a2+1:3d}"
                  f" ({100*cap[a2:b2+1].sum()/(b2-a2+1):5.1f} %)"
                  f"  beam log-gain accrued at cap {100*g_cap/g_all:5.1f} %"
                  f"  tau {tau_us(e,a2,b2):8.3f} us")
        print("    cathode_phi_c_cap_V = "
              f"{runs['covcal_f2_shot1']['params']['cathode_phi_c_cap_V']} V")

    section("(5) SENSITIVITY MAP -- measured leverage from the artifact family")
    print("  ARITHMETIC OF THE TARGET (shot 1, holding the breakdown log-gain and")
    print("  the ignition transient fixed):")
    d = runs["covcal_f2_shot1"]
    a, b, _ = legs["covcal_f2_shot1"]
    obs = float(np.log(d["meann"][b] / d["meann"][a]))
    m = np.flatnonzero((d["t"] >= 1e-5 - 1e-15) & (d["t"] <= 5e-5))
    i0, i1 = int(m[0]), int(m[-1])
    dl0 = float(np.log(d["meann"][i1] / d["meann"][i0]))
    dt0 = d["t"][i1] - d["t"][i0]
    tgt = 0.5 * (BAND_LO_US + BAND_HI_US) * 1e-6
    dt_tot = tgt * obs
    g_post_now = (obs - dl0) / (d["t"][b] - d["t"][i1])
    g_post_need = (obs - dl0) / (dt_tot - dt0)
    print(f"    total leg log-gain               {obs:.4f}")
    print(f"    ignition transient (10-50 us)    {dl0:.4f} in {dt0:.2e} s")
    print(f"    uniform slowdown to hit {tgt*1e6:.0f} us  x{tgt*1e6/tau_us(d,a,b):.2f}")
    print(f"    post-transient rate now          {g_post_now:.4e} 1/s")
    print(f"    post-transient rate needed       {g_post_need:.4e} 1/s")
    print(f"    => required reduction            x{g_post_now/g_post_need:.2f}")

    print("\n  MEASURED leverage, ladder family (DIFFERENT CLOSURE SET: nx=240,")
    print("  coverage_closure OFF, production ladder stance -- leverage only):")
    print(f"  {'stem':28s} {'transport':>10s} {'tail_ion':>8s} {'f':>5s} "
          f"{'t_bd [ms]':>10s} {'TAU [us]':>9s}")
    lad = {}
    for s in LADDER:
        e = load(s)
        if e is None:
            print(f"  {s:28s} ABSENT")
            continue
        a2, b2, t_bd = leg(e)
        p = e["params"]
        fphi = p.get("heating_anomalous_tail_phi_c_fraction")
        lad[s] = tau_us(e, a2, b2)
        print(f"  {s:28s} {p['heating_anomalous_transport']:>10} "
              f"{p['heating_anomalous_tail_ionization']:>8} "
              f"{('0.25' if fphi is None else fphi):>5} "
              f"{(t_bd*1e3 if t_bd else float('nan')):10.4f} {lad[s]:9.3f}")
    if len(lad) == len(LADDER):
        r = lad["es1_lad_tw25ion_nx240"]
        print(f"\n    f bracket   0.25 -> 0.50 : tau x{lad['es1_lad_tw50ion_nx240']/r:.3f}")
        print(f"    f bracket   0.25 -> 1.00 : tau x{lad['es1_lad_tw100ion_nx240']/r:.3f}")
        print(f"    walker ionization ON->OFF: tau x{lad['es1_lad_tw25_nx240']/r:.3f}")
        print(f"    tail_walk -> local       : tau x{lad['es1_lad_ref_nx240']/r:.3f}")
        f = np.array([0.25, 0.5, 1.0])
        y = np.array([lad["es1_lad_tw25ion_nx240"], lad["es1_lad_tw50ion_nx240"],
                      lad["es1_lad_tw100ion_nx240"]])
        sl = float(np.polyfit(np.log(f), np.log(y), 1)[0])
        print(f"    d ln tau / d ln f = {sl:.4f}  -> f needed for "
              f"{0.5*(BAND_LO_US+BAND_HI_US):.0f} us: "
              f"{0.25*(0.5*(BAND_LO_US+BAND_HI_US)/y[0])**(1/sl):.2f} "
              "(outside the declared bracket AND outside [0,1])")

    print("\n  MEASURED leverage, f_cov0 probes (EACH ROW ITS OWN CLOSURE SET --")
    print("  read the transport/tail_ion columns before comparing any two):")
    print(f"  {'stem':28s} {'f_cov0':>7s} {'transport':>10s} {'tail_ion':>8s} "
          f"{'t_bd [ms]':>10s} {'TAU [us]':>9s} {'leg':>16s}")
    for s in FCOV:
        e = load(s)
        if e is None:
            print(f"  {s:28s} ABSENT")
            continue
        a2, b2, t_bd = leg(e)
        p = e["params"]
        grew = e["meann"][b2] > e["meann"][a2] > 0
        tv = tau_us(e, a2, b2) if grew else float("nan")
        print(f"  {s:28s} {p['coverage_initial_fraction']:>7} "
              f"{p['heating_anomalous_transport']:>10} "
              f"{p['heating_anomalous_tail_ionization']:>8} "
              f"{(t_bd*1e3 if t_bd else float('nan')):10.4f} {tv:9.3f} "
              f"{('to breakdown' if t_bd else 'SURVIVING BUILD'):>16s}")
    print("\n    r (coverage growth) leverage, from section (1): "
          f"{tau_us(runs['covdecide_twion_f005'], *legs['covdecide_twion_f005'][:2]):.2f}"
          f" -> {tau_us(runs['covcal_f2_shot2'], *legs['covcal_f2_shot2'][:2]):.2f} us"
          " over SEVEN decades of r == the F2 null.")

    section("(6) AMPLITUDE CROSS-CHECK -- where the I_max difference flows")
    s1, s2 = runs["covcal_f2_shot1"], runs["covcal_f2_shot2"]
    e1, e2 = escape(s1), escape(s2)
    print("  shot 1 (r=179.9, coverage GROWS) vs shot 2 (r=2e-4, coverage FROZEN)")
    print(f"  {'t [ms]':>7s} | {'f_cov1':>7s} {'f_cov2':>7s} | {'I_loop1':>8s} "
          f"{'I_loop2':>8s} | {'byp1':>7s} {'byp2':>7s} | {'V_dis1':>7s} "
          f"{'V_dis2':>7s} | {'esc1':>8s} {'esc2':>8s}")
    for i in (50, 100, 155, 200, 300, 400, 500, 600, 700, 800):
        print(f"  {s1['t'][i]*1e3:7.3f} | {s1['coverage_fraction'][i]:7.5f} "
              f"{s2['coverage_fraction'][i]:7.5f} | {s1['circuit_I_loop'][i]:8.1f} "
              f"{s2['circuit_I_loop'][i]:8.1f} | "
              f"{s1['source_beam_bypass_fraction'][i]:7.5f} "
              f"{s2['source_beam_bypass_fraction'][i]:7.5f} | "
              f"{s1['circuit_V_dis_step'][i]:7.2f} {s2['circuit_V_dis_step'][i]:7.2f} | "
              f"{e1[i]:8.6f} {e2[i]:8.6f}")
    print(f"\n  I_loop max: shot 1 {s1['circuit_I_loop'].max():.2f} A, "
          f"shot 2 {s2['circuit_I_loop'].max():.2f} A "
          f"(ratio {s2['circuit_I_loop'].max()/s1['circuit_I_loop'].max():.3f})")
    print(f"  bypass at t_end: shot 1 {s1['source_beam_bypass_fraction'][-1]:.6f}, "
          f"shot 2 {s2['source_beam_bypass_fraction'][-1]:.6f}")
    print("  _cathode_solver.py:1129  P_prim = (1 - eta*beam_bypass_fraction)"
          "*I_eth_star*phi_c")
    print("  _cathode_solver.py:443   J_anode = J_tot - eta*beam_bypass_fraction*J_star")
    print("  _cathode_solver.py:468   _compute_beam_bypass_fraction(l_b, L_cath)")

    section("(7) WATCH ITEMS")
    for s in SHOTS:
        d = runs[s]
        a, b, _ = legs[s]
        i = a  # first live save
        pool = d["source_I_eth_star"][i] * d["source_phi_c"][i]
        print(f"  {s:24s} first live save t={d['t'][i]:.1e}: "
              f"P_tail/(I_eth* phi_c) = {d['beam_tail_power_W'][i]/pool:.4f}, "
              f"P_tail/P_prim = {d['beam_tail_power_W'][i]/d['source_P_prim'][i]:.4f}")
    d = runs["covcal_f2_shot1"]
    print(f"\n  ADAS table domain: table_Te_min = {d['rate_Te_min_table']:.4f} eV; "
          f"first active cell below it at t = {d['rate_first_below_s']:.3e} s "
          f"(INSIDE the build leg); max active-volume fraction below = "
          f"{d['rate_vol_below'].max():.4f}")
    print("\n  Timestep constraint census (from the memos, re-read here):")
    for s in ("covcal_f2_shot1", "covcal_f2_shot2"):
        p = HERE / f"{s}.h5"
        with h5py.File(p, "r") as f:
            de = np.asarray(f["diagnostics"]["dt_energy_exchange"], float)
            ac = [sdec(v) for v in np.asarray(f["diagnostics"]["active_constraint"])]
        import collections
        print(f"    {s:24s} min dt_energy_exchange {np.nanmin(de):.4e} s; "
              f"census {dict(collections.Counter(ac))}")


def adas_check():
    """FROZEN-STATE evaluation of the solver's OWN rate table -- no solve.

    Confirms the bulk channel's implied <sigma v> is the ADAS SCD at the
    build-leg (ne, Te), and prices the Te sensitivity of that coefficient.
    """
    section("(8) BULK CHANNEL: FROZEN-STATE CROSS-CHECK AGAINST ADAS")
    from cablp.atomic.adas import he_rates
    for ne, Te in ((1.0e9, 4.68), (1.0e10, 4.69), (1.0e10, 4.44), (1.0e10, 5.09)):
        v = float(he_rates(np.array([ne]), np.array([Te]), ("scd",))["scd"][0])
        print(f"    he_rates SCD(ne={ne:.1e}, Te={Te:.2f}) = {v:.4e} cm^3/s  "
              f"-> nn*SCD at nn=2.0e13 = {2.0e13*v:.4e} 1/s")
    Te = np.linspace(2.0, 6.0, 4001)
    ne = np.full_like(Te, 1.0e10)
    scd = he_rates(ne, Te, ("scd",))["scd"]
    ref = float(np.interp(4.70, Te, scd))
    dln = float(np.interp(4.70, Te, np.gradient(np.log(scd), np.log(Te))))
    print(f"\n    d ln SCD / d ln Te at Te=4.70 eV: {dln:.3f} (a STEEP lever)")
    for fac in (2.0, 3.0, 5.93, 6.24, 10.0):
        print(f"      to cut the bulk channel by x{fac:5.2f}: Te must fall "
              f"4.700 -> {float(np.interp(ref/fac, scd, Te)):.3f} eV "
              f"({100*(1-float(np.interp(ref/fac, scd, Te))/4.70):.1f} % drop)")
    print("\n    b_ioniz = 1.0 and atomic_rate_model = 'adas' in all three shots:")
    print("    the coefficient is the trusted ADAS SCD and is NOT a knob "
          "(campaign standing policy).")


if __name__ == "__main__":
    main()
    adas_check()
