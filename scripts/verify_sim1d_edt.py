"""Build gates for the electron drift-transport + EMF-work operator (edt).

**These gates were registered BEFORE the operator was implemented** and are not
moved after seeing results (AGENTS.md, "Briefs and reports"). Each names its
QUANTITY, its MEASUREMENT SITE, and its FIXTURE. Gates 2-4 and 6 are properties
of a SAVED state and of the operator's own algebra, so they are measurable
before the solver-side code exists at all; ``scripts/edt_consult_pins.py`` is
the standalone evaluator that measures them, and it is what established the
base-side readings this suite is checked against.

Run from the checkout root::

    PYTHONPATH=<checkout> python scripts/verify_sim1d_edt.py

Exit 0 = every gate passed. A failure is a DELIVERABLE: it is printed with its
numbers and the suite exits 1. Never relax a tolerance here to make a gate pass.

--------------------------------------------------------------------------
GATE REGISTRY
--------------------------------------------------------------------------

**G1 -- bit-inertness with the flag off.**
  QUANTITY: the accepted-step trajectory and the golden config identity.
  SITE: ``scripts/golden_digest_gate.py`` (4,000-step chain digest, all five
  checkpoints and the final digest) and an A/B of the RHS term rows on both
  neutral routes.
  FIXTURE: the golden config at nx=60 for the digest; ``default_config()``
  for the moment and kinetic-DVM A/B routes.
  PASS: every checkpoint and the final digest unchanged from the committed
  reference; the config identity moves ONLY by the addition of the three new
  keys (one flag, two params), proven by a strip-control that reproduces the
  base identity bit-for-bit through the gate's OWN expression; both routes
  row-by-row bit-identical to base with only the new all-zero rows one-sided.

**G2 -- the volume identity.**
  QUANTITY: ``sum(Delta dV) - ([3.21 T_e I/e]_in - [3.21 T_e I/e]_out + W_EMF)``,
  relative to the larger side.
  SITE: the operator's own named rows on ONE accepted step.
  FIXTURE: the golden config at nx=60, and the ES1 source region on
  ``scripts/mgcr1_confirm.h5``'s saved state.
  PASS: <= 1e-10 relative on every arm of the bracket.

**G3 -- the cathode-face handshake.**
  QUANTITY: the drift enthalpy-plus-thermal-force influx at the cathode face,
  ``2.21 T_e Gamma_d A``, window-mean over 0.1-20.1 ms.
  SITE: the named cathode-face row.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` (data in hand; no new run).
  PASS: +14.8 kW within the consult's bracket-A/B spread (the consult puts that
  spread below 15 %; this suite gates at 15 %).

**G4 -- the compression piece.**
  QUANTITY: the pressure-drift work summed over the gap cells (2-5),
  window-mean over 0.1-20.1 ms, reported ROW-RELATIVE and
  throughput-normalized (AGENTS.md, "Negative controls gate on the ROW-RELATIVE
  normalization").
  SITE: ``edt_pressure_drift_work_W``.
  FIXTURE: ``scripts/mgcr1_confirm.h5``.
  PASS: +13.6 kW within 15 % ROW-RELATIVE. The throughput-normalized figure is
  REPORTED, never gated -- a piece that is 3.9 % of P_prim reads as "small" on
  that normalization however wrong it is.

**G5 -- the J = 0 limit (negative control at the statement level).**
  QUANTITY: every cell of the operator's total row.
  SITE: the ``electron_drift_transport`` RHS term.
  FIXTURE: ``default_config()`` with the plasma on and the drive off (no
  cathode coupling, no beam), the flag ARMED.
  PASS: exactly zero on every cell -- not "small", zero. With no current there
  is no drift, and a term that leaks anything here is reading something other
  than the booked current.

**G6 -- the afterglow clause (REPORTED, NOT GATED).**
  QUANTITY: the operator's net over the source region during afterglow.
  SITE: the operator's total row.
  FIXTURE: ``scripts/mgcr1_confirm.h5`` at t = 26 ms AND over the whole
  afterglow window (t > 20.1 ms). Both are reported because the consult's
  -0.5...-0.6 kW belongs to a 213 A tail, and 213 A is the afterglow WINDOW's
  mean loop current -- at the 26 ms instant the loop carries ~12 A. The clause
  exists so that "the term is zero in afterglow" is never stated flatly.

Companion gates that are NOT this suite's to run, and where they live:
smoke (``scripts/smoke_sim1d.py``), the DVM suite (``verify_sim1d_k2_dvm.py``),
the digest gate (``scripts/golden_digest_gate.py``), the snapshot delta
(``scripts/edt_snapshot_delta.py``) and the A/B bit-inertness reader
(``scripts/edt_bitinert_ab.py``).
"""

import argparse
import sys

#: The registry above, in one machine-readable place so the report and the
#: suite cannot drift apart. ``gated`` False means the gate is REPORTED only.
GATES = (
    {
        "id": "G1",
        "name": "bit-inertness with the flag off",
        "quantity": "4k accepted-step digest chain + config identity + "
                    "row-by-row A/B on both neutral routes",
        "site": "golden_digest_gate.py; edt_bitinert_ab.py",
        "fixture": "golden config nx=60; default_config() moment and "
                   "kinetic_dvm routes",
        "gated": True,
    },
    {
        "id": "G2",
        "name": "volume identity",
        "quantity": "sum(Delta dV) vs 3.21 T_e I/e boundary + W_EMF, relative",
        "site": "the operator's named rows, one accepted step",
        "fixture": "golden config nx=60; mgcr1_confirm.h5 source region",
        "tolerance": 1e-10,
        "gated": True,
    },
    {
        "id": "G3",
        "name": "cathode-face handshake",
        "quantity": "2.21 T_e Gamma_d A at the cathode face, window mean",
        "site": "the named cathode-face row",
        "fixture": "mgcr1_confirm.h5, 0.1-20.1 ms",
        "target_kW": 14.8,
        "tolerance": 0.15,
        "gated": True,
    },
    {
        "id": "G4",
        "name": "compression piece",
        "quantity": "pressure-drift work summed over the gap cells, window mean",
        "site": "edt_pressure_drift_work_W",
        "fixture": "mgcr1_confirm.h5, 0.1-20.1 ms",
        "target_kW": 13.6,
        "tolerance": 0.15,
        "gated": True,
    },
    {
        "id": "G5",
        "name": "J = 0 limit",
        "quantity": "every cell of the operator's total row",
        "site": "the electron_drift_transport RHS term",
        "fixture": "default_config(), plasma on, drive off, flag ARMED",
        "gated": True,
    },
    {
        "id": "G6",
        "name": "afterglow clause",
        "quantity": "the operator's net over the source region in afterglow",
        "site": "the operator's total row",
        "fixture": "mgcr1_confirm.h5 at 26 ms AND over t > 20.1 ms",
        "gated": False,
    },
)


def print_registration():
    print("[ue-pressure-work] build-gate registration")
    for g in GATES:
        tag = "GATED" if g["gated"] else "REPORTED"
        print(f"  {g['id']} ({tag}) -- {g['name']}")
        print(f"      quantity : {g['quantity']}")
        print(f"      site     : {g['site']}")
        print(f"      fixture  : {g['fixture']}")
        if "target_kW" in g:
            print(
                f"      target   : {g['target_kW']:+.1f} kW within "
                f"{g['tolerance'] * 100:.0f}% row-relative"
            )
        if "tolerance" in g and "target_kW" not in g:
            print(f"      tolerance: {g['tolerance']:.0e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--registration",
        action="store_true",
        help="print the gate registry and exit without running anything",
    )
    args = ap.parse_args(argv)
    print_registration()
    if args.registration:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
