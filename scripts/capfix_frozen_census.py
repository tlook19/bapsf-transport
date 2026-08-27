"""capfix: the frozen-arm digests PLUS a census of cap escapes per step.

Runs exactly the arm ``k7cbuild_frozen_bitexact.py`` runs -- same build, same
digest set, same cadence -- so its digest lines diff line-for-line against
``sbreview_frozen_{tw,twion}_postmerge.txt``. On top of that it wraps
``solve_idriven`` and counts, per step:

- ``solves``        every current-driven sheath solve inside the step;
- ``caplim``        solves tagged ``capability_limited``;
- ``escape``        solves whose NET phi_c exceeds the cap while the regime is
                    NOT ``capability_limited`` -- the defect signature. Split
                    into a MATERIAL count (more than 1e-9 relative above the
                    cap: a genuine ladder escape) and a ULP count (at the cap
                    to within 1e-9: the ceiling brentq's own last bits);
- ``max net/cap``   the largest net-phi_c/cap ratio seen in the step.

Run pre-fix to attribute any digest movement to specific escaped solves, and
post-fix to show the material count is zero. Read-only with respect to the
solver: the wrapper only observes.

Usage (from <checkout>/cablp, PYTHONPATH set to that same cablp):
    python scripts/capfix_frozen_census.py --arm tw --steps 400 --out FILE
"""

import argparse
import os

from k7cbuild_frozen_bitexact import ARMS, build, digest  # noqa: E402

from cablp.funcs import _cathode_solver_idriven as idr  # noqa: E402
from cablp.funcs._kernels import PROVENANCE  # noqa: E402
from cablp.solvers._sim1d.physics import cathode as cath  # noqa: E402

# A net phi_c this far above the cap is a real ladder escape rather than the
# ceiling root-find's own last bits (brentq runs at rtol 1e-14).
MATERIAL_REL = 1.0e-9

OUT = []


def P(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    OUT.append(s)


_real_solve = idr.solve_idriven
TALLY = {
    "solves": 0,
    "caplim": 0,
    "escape_material": 0,
    "escape_ulp": 0,
    "max_ratio": 0.0,
    "max_escape_ratio": 0.0,
}


def traced(config, plasma, **kw):
    res = _real_solve(config, plasma, **kw)
    cap = float(kw["phi_c_cap_V"])
    ratio = float(res.phi_c) / cap if cap > 0.0 else 0.0
    TALLY["solves"] += 1
    if res.regime == "capability_limited":
        TALLY["caplim"] += 1
    elif ratio > 1.0 + MATERIAL_REL:
        TALLY["escape_material"] += 1
        TALLY["max_escape_ratio"] = max(TALLY["max_escape_ratio"], ratio)
    elif ratio > 1.0:
        TALLY["escape_ulp"] += 1
    TALLY["max_ratio"] = max(TALLY["max_ratio"], ratio)
    return res


def snapshot():
    return dict(TALLY)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--report-every", type=int, default=100)
    p.add_argument("--arm", choices=sorted(ARMS), default="tw")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    idr.solve_idriven = traced
    cath.solve_idriven = traced

    sim, pinned = build(args.arm)
    P(f"# capfix_frozen_census  kernels: {PROVENANCE}")
    P(
        f"arm={args.arm} cells={sim.geometry.cells} dvm={sim._dvm is not None} "
        f"anom_transport="
        f"{sim._input_dict.get('heating_anomalous_transport', 'local')!r} "
        f"tail_ionization="
        f"{sim._input_dict.get('heating_anomalous_tail_ionization', 'off')!r} "
        f"E_tail="
        f"{sim._input_dict.get('heating_anomalous_tail_energy_eV', 75.0)!r}"
        f"\nlegacy pins applied: "
        f"{pinned if pinned else '(none -- shipped-closure arm)'}"
    )
    P(f"cap = {sim._input_dict.get('cathode_phi_c_cap_V', 1000.0)!r} V")
    P("")

    window = snapshot()
    rows = []
    for step in range(1, args.steps + 1):
        sim.advance_one_step()
        if step % args.report_every == 0 or step == args.steps:
            now = snapshot()
            rows.append(
                (
                    step,
                    now["solves"] - window["solves"],
                    now["caplim"] - window["caplim"],
                    now["escape_material"] - window["escape_material"],
                    now["escape_ulp"] - window["escape_ulp"],
                    now["max_ratio"],
                    now["max_escape_ratio"],
                )
            )
            window = now
            dvm = sim._dvm
            P(
                f"step {step:6d}  t={sim.time:.9e}  updates={dvm.updates:6d}\n"
                f"    y      {digest(sim._y)}\n"
                f"    f_c    {digest(dvm.f_c)}\n"
                f"    f_a    {digest(dvm.f_a)}\n"
                f"    pend   {digest(dvm.pend_L_c, dvm.pend_R_c, dvm.pend_L_a, dvm.pend_R_a)}\n"
                f"    xfer   {digest(dvm.M_transfer, dvm.Ei_transfer, dvm.S_transfer, dvm.Tn_col_eV)}\n"
                f"    debt   {digest(dvm.M_debt, dvm.Ei_debt, dvm.M_applied_cum, dvm.Ei_applied_cum, dvm.M_booked_cum, dvm.Ei_booked_cum)}",
            )
    P("")
    P("== CAP-ESCAPE CENSUS (per reporting window)")
    P(
        f"{'thru step':>10} {'solves':>8} {'caplim':>8} "
        f"{'escape(material)':>17} {'escape(ulp)':>12} "
        f"{'max net/cap':>12} {'max escape':>11}"
    )
    for r in rows:
        P(
            f"{r[0]:>10} {r[1]:>8} {r[2]:>8} {r[3]:>17} {r[4]:>12} "
            f"{r[5]:>12.6f} {r[6]:>11.6f}"
        )
    tot = snapshot()
    P("")
    P(
        f"TOTAL solves={tot['solves']} caplim={tot['caplim']} "
        f"escape_material={tot['escape_material']} "
        f"escape_ulp={tot['escape_ulp']} "
        f"max_net_over_cap={tot['max_ratio']!r} "
        f"max_escape_ratio={tot['max_escape_ratio']!r}"
    )
    inc = (
        100.0 * tot["escape_material"] / tot["solves"] if tot["solves"] else 0.0
    )
    P(f"material escape incidence: {inc:.4f} % of solves")
    if args.out:
        with open(args.out, "w") as fh:
            fh.write("\n".join(OUT) + "\n")
        print(f"# wrote {os.path.abspath(args.out)}", flush=True)


if __name__ == "__main__":
    main()
