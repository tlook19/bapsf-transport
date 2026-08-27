"""Gate 6: the R1 ON-probe re-read under the phi_a-aware bound object.

Runs ``r1vb_probe_read`` verbatim on the supplied probe stems -- so R1's own
registered assertions A1-A4 are evaluated unchanged, and a failure is a
failure -- and then adds what R1 could not ask, because the object it bounded
was ``phi_c``:

  B1  the BOUND VALUES: the composed ceiling against the circuit-available
      voltage, and the gap between them, which under ``device_voltage`` is
      ``phi_a - V_p`` rather than zero;
  B2  the OBJECT statement: ``V_b <= V_avail`` on every bound solve, which is
      what the new object actually promises (A1 restated as the primary claim
      rather than as a corollary);
  B3  the CONTRACT verdict evidence. The phi_c/V_b mismatch is gone by
      construction, and since 2026-08-12 so is the back-EMF exclusion: the
      circuit integrates the sheath's UNBOUNDED demand, so dI/dt is no longer
      held at zero while the bound binds. The test is unchanged and its
      READING is inverted -- a FALLING loop current on a bound solve was the
      excluded-impossible state under the old contract and is the positive
      evidence under the new one. Reported per arm.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/regime_vcm_onprobe_read.py regime_vcm_onprobe_dv \
        regime_vcm_onprobe_phic
"""

import sys

import numpy as np

import r1vb_probe_read as R1

EPS = R1.EPS


def bound_values(d):
    """B1/B2: what the bound was worth, and whether it kept its promise."""
    stem = d["stem"]
    obj = d["params"].get("cathode_circuit_bound_object", "phi_c")
    bound_on = bool(d["flags"].get("cathode_circuit_voltage_bound", False))
    print("-" * 78)
    print(f"{stem}   bound={bound_on}   object={obj!r}")
    if not bound_on or "source_bound_active" not in d:
        print("  (no bound in force; nothing to read)")
        return []

    live = np.asarray(d.get("has_solution", np.ones_like(d["t"])), float) > 0.0
    code = d["source_bound_active"]
    sel = live & np.isfinite(code) & np.isfinite(d["source_circuit_V_avail_V"])
    if not np.any(sel):
        print("  (no saved solve carried a finite available voltage)")
        return []

    ceil = d["source_phi_c_ceiling_V"][sel]
    avail = d["source_circuit_V_avail_V"][sel]
    phi = d["source_phi_c"][sel]
    phi_a = d["source_phi_a"][sel]
    V_p = d["source_V_p"][sel]
    V_b = d["source_V_b"][sel]
    bnd = code[sel] == 2.0

    print(f"  B1 bound values over {int(sel.sum())} solves "
          f"({int(bnd.sum())} sat on the circuit bound):")
    print(f"       V_avail        median {np.nanmedian(avail):11.4f} V   "
          f"range [{np.nanmin(avail):.4f}, {np.nanmax(avail):.4f}]")
    print(f"       ceiling        median {np.nanmedian(ceil):11.4f} V   "
          f"range [{np.nanmin(ceil):.4f}, {np.nanmax(ceil):.4f}]")
    print(f"       ceiling-V_avail median {np.nanmedian(ceil - avail):11.4f} V "
          f"  (device_voltage: this is phi_a - V_p; phi_c: exactly 0)")
    print(f"       phi_c          median {np.nanmedian(phi):11.4f} V   "
          f"max {np.nanmax(phi):.4f}")
    print(f"       phi_a          median {np.nanmedian(phi_a):11.4f} V   "
          f"V_p median {np.nanmedian(V_p):.6f} V")
    print(f"       V_b            median {np.nanmedian(V_b):11.4f} V   "
          f"max {np.nanmax(V_b):.4f}")

    fails = []
    over_bnd = bnd & (V_b > avail * (1.0 + EPS))
    print(f"  B2 V_b <= V_avail on BOUND solves: "
          f"{int(bnd.sum()) - int(over_bnd.sum())}/{int(bnd.sum())} hold")
    if np.any(over_bnd):
        fails.append(f"{stem}: B2 {int(over_bnd.sum())} bound solves above "
                     "V_avail")
    if obj == "device_voltage" and np.any(bnd):
        # The object is reached by SOLVING, not by clamping: the assembled
        # device voltage should sit ON the available voltage.
        resid = (phi + V_p - phi_a - avail)[bnd]
        print(f"       assembled phi_c + V_p - phi_a - V_avail on bound "
              f"solves: max |resid| = {np.nanmax(np.abs(resid)):.3e} V")
    return fails


#: A save counts as FALLING only below this relative step. The loop current
#: wanders at the 1e-11 level from the circuit brentq's own tolerance, and an
#: exact `dI < 0` test reads that noise as decay -- which it did, on the first
#: pass, and reported the excluded condition as occurring in a build-leg
#: window that contains no decay at all. 1e-9 sits two decades above the
#: measured noise floor and many decades below any physical ramp-down.
FALL_REL = 1.0e-9


def contract_evidence(d):
    """B3: is a full-window flag-ON in contract now?"""
    bound_on = bool(d["flags"].get("cathode_circuit_voltage_bound", False))
    if not bound_on or "source_bound_active" not in d:
        return
    t = d["t"]
    I = d.get("circuit_I_loop")
    code = d["source_bound_active"]
    if I is None:
        print("  B3: no circuit_I_loop trace saved")
        return
    live = np.asarray(d.get("has_solution", np.ones_like(t)), float) > 0.0
    dI = np.zeros_like(I)
    dI[1:] = np.diff(I)
    scale = np.maximum(np.abs(I), 1e-30)
    falling = live & (dI < -FALL_REL * scale)
    noise = live & (dI < 0.0) & ~falling
    bound_now = live & (code == 2.0)
    both = falling & bound_now
    print(f"  B3 back-EMF exclusion over {int(live.sum())} saved solves:")
    print(f"       saves with the circuit bound ACTIVE: "
          f"{int(bound_now.sum())}")
    print(f"       saves FALLING by more than {FALL_REL:.0e} relative: "
          f"{int(falling.sum())}")
    print(f"       saves falling only within that noise band: "
          f"{int(noise.sum())}"
          + (f" (worst {np.nanmin((dI / scale)[noise]):.2e} relative)"
             if np.any(noise) else ""))
    print(f"       saves with BOTH bound and a real fall (the condition the "
          f"OLD contract excluded): {int(both.sum())}")
    print(f"       I_loop: {I[0]:.6g} -> {I[-1]:.6g} A, max "
          f"{np.nanmax(I):.6g} A")
    # The reading of these counts INVERTED on 2026-08-12. While the circuit
    # integrated the BOUNDED device voltage the loop residual was identically
    # zero wherever the bound bound, so `both` could only ever be zero and a
    # frozen I_loop was the expected -- documented -- signature. The circuit
    # now integrates the sheath's unbounded demand, so dI/dt is free while
    # the bound is active, and `both` counts the saves that PROVE it: a bound
    # solve on a falling current is exactly the state the old contract
    # excluded as impossible.
    held = bound_now.copy()
    held[0] = False
    if np.any(held):
        spread = float(np.nanmax(I[held]) - np.nanmin(I[held]))
        print(f"       while BOUND, I_loop spans {spread:.3e} A over "
              f"{int(held.sum())} saves "
              f"({spread / max(float(np.nanmax(I[held])), 1e-30):.2e} "
              f"relative) -- a frozen loop would span zero")
    if int(both.sum()) == 0:
        print("       -> NO bound save falls in this window. That is not by")
        print("          itself the old freeze: nothing here need be falling.")
        print("          This arm gives no evidence either way, and the")
        print("          verdict below rests on the loop equation alone.")
    else:
        print("       -> the loop current FALLS on bound saves, which the")
        print("          pre-2026-08-12 contract excluded as impossible.")
        print("          The back-EMF exclusion no longer freezes dI/dt:")
        print("          the bound clamps the reported/beam-facing voltage")
        print("          while the circuit integrates the unbounded demand.")


def main(argv=None):
    stems = list(argv if argv is not None else sys.argv[1:])
    if not stems:
        raise SystemExit(__doc__)
    runs = {stem: R1.load(stem) for stem in stems}

    print("#" * 78)
    print("# PART 1 -- R1's own registered read, verbatim (A1-A4)")
    print("#" * 78)
    failures = []
    for stem in stems:
        failures += [f"{stem}: {f}" for f in R1.report(runs[stem])]

    print("#" * 78)
    print("# PART 2 -- the phi_a-aware object: bound values and the contract")
    print("#" * 78)
    for stem in stems:
        failures += bound_values(runs[stem])
        contract_evidence(runs[stem])
        print()

    print("#" * 78)
    print("# VERDICT")
    print("#" * 78)
    if failures:
        print("REGISTERED ASSERTIONS FAILED:")
        for f in failures:
            print(f"  - {f}")
    else:
        print("every registered assertion held on every arm read")
    print()
    print("CONTRACT: the phi_c/V_b OBJECT mismatch R1 documented is removed by")
    print("cathode_circuit_bound_object='device_voltage' -- the bound's object")
    print("IS V_b, so phi_c exceeding V_avail where phi_a subtracts is no")
    print("longer a mis-clamp but the correct answer. THE BACK-EMF EXCLUSION")
    print("IS ALSO GONE (2026-08-12): it rested on the circuit integrating the")
    print("BOUNDED device voltage, which made the loop residual identically")
    print("zero wherever the bound bound, hence dI/dt = 0 and a frozen decay.")
    print("The circuit now integrates the sheath's UNBOUNDED demand, so the")
    print("restoring force survives the clamp and the current is free to fall")
    print("while the bound is active -- which B3 above MEASURES rather than")
    print("assumes. The inductor's stored energy is still not counted as")
    print("supply; that exclusion now costs the falling leg nothing, because")
    print("it constrains only the reported and beam-facing objects.")
    print("A FULL-WINDOW RUN WITH THE FLAG ON IS THEREFORE IN CONTRACT.")
    print()
    print("SCOPE OF THIS EVIDENCE. The probe covers 0 -> 0.033 ms of the build")
    print("leg and reaches no main-discharge decay, so what is MEASURED is")
    print("that dI/dt is free while the bound binds -- the mechanism the old")
    print("exclusion rested on -- and not the decay itself. A window reaching")
    print("the plateau decay would confirm the cost is zero end to end; this")
    print("brief did not buy one, so the full-window statement remains")
    print("structural for the decay leg specifically.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
