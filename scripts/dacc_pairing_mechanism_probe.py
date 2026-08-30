"""Probe: WHY is the cathode-backscatter pairing refusal unreachable?

The three prose sites (``refuse_cathode_backscatter_double_book``'s
docstring, ``core/config.py``'s ``neutral_kinetic_dvm_cathode_jet`` entry,
and ``gate_cj4``'s docstring) attribute the unreachability to the
``neutral_momentum`` chain. This probe asks the code which statement is
true, by driving ``LAPDSim1D`` at the pair and reading what actually
refuses.

Run from the worktree root with ``PYTHONPATH=<worktree>``.
"""

import os
import sys

import cablp
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.model_families import (
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _provenance():
    pkg = os.path.abspath(cablp.__file__)
    inside = pkg.startswith(ROOT + os.sep)
    print(f"cablp.__file__ = {pkg}")
    print(f"worktree root  = {ROOT}")
    print(f"import inside worktree: {inside}")
    assert inside, "PYTHONPATH TRAP: cablp resolved outside the worktree"


def _build(**over):
    params, flags = default_config()
    params = dict(params)
    flags = dict(flags)
    params["nx"] = 24
    params["neutral_model"] = "kinetic_dvm"
    flags["neutral_two_zone"] = True
    for key, val in over.items():
        if key in flags:
            flags[key] = val
        else:
            params[key] = val
    return LAPDSim1D(input_dict=params, input_flags=flags)


def main():
    _provenance()
    print()

    members = {k: v for (_ns, k, v, _why) in KINETIC_DVM_INCOMPATIBLE_DEFAULTS}
    print("--- Is cathode_jet_surface_debit its OWN resolver member? ---")
    print(
        "cathode_jet_surface_debit in KINETIC_DVM_INCOMPATIBLE_DEFAULTS: "
        f"{'cathode_jet_surface_debit' in members}"
    )
    print(
        "  required value: "
        f"{members.get('cathode_jet_surface_debit', '<absent>')!r}"
    )
    print(
        "cathode_neutral_jet in KINETIC_DVM_INCOMPATIBLE_DEFAULTS: "
        f"{'cathode_neutral_jet' in members}"
    )
    print(
        "neutral_momentum in KINETIC_DVM_INCOMPATIBLE_DEFAULTS: "
        f"{'neutral_momentum' in members}"
    )
    print()

    print("--- A: pair armed, debit left AT ITS CONFIG DEFAULT ---")
    try:
        sim = _build(neutral_kinetic_dvm_cathode_jet=True)
        got = sim._input_dict.get("cathode_jet_surface_debit")
        print(f"constructed OK; resolved cathode_jet_surface_debit = {got!r}")
    except ValueError as exc:
        print(f"REFUSED: {str(exc)[:400]}")
    print()

    print("--- B: pair armed, debit set to True BY THE CALLER ---")
    try:
        sim = _build(
            neutral_kinetic_dvm_cathode_jet=True,
            cathode_jet_surface_debit=True,
        )
        got = sim._input_dict.get("cathode_jet_surface_debit")
        print(f"constructed OK; resolved cathode_jet_surface_debit = {got!r}")
        print(
            "  the resolver reads 'explicitly set' as 'differs from the "
            "config template'. True IS the template value, so a caller "
            "cannot distinguish itself from silence here."
        )
    except ValueError as exc:
        print(f"REFUSED: {str(exc)[:600]}")
    print()

    print("--- B2: the debit's only other value is the one the arm wants ---")
    print(
        "cathode_jet_surface_debit is a bool; the template ships True and "
        "the resolver requires False, so BOTH reachable values leave the "
        "pairing guard looking at False. There is no config that reaches it."
    )
    print()

    print("--- C: what does the config template ship for the debit? ---")
    params, flags = default_config()
    print(
        "default cathode_jet_surface_debit = "
        f"{params['cathode_jet_surface_debit']!r}"
    )
    print(f"default neutral_momentum          = {flags['neutral_momentum']!r}")
    print(
        "default cathode_neutral_jet       = "
        f"{params['cathode_neutral_jet']!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
