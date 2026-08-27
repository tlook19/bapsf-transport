"""Demonstration transcript for the model-preset resolver (2026-08-23h/aj/ak).

Four cases, printed verbatim:

(a) the session-29 entrainment two-delta config -- the g1atrim stance plus
    ``neutral_wall_momentum_partition`` and its cross section -- which used
    to be discovered one guard at a time and now produces ONE collected
    refusal naming the whole ``kinetic_two_moment`` set;
(b) a ``kinetic_dvm`` arm built from ``default_config()`` with NOTHING
    hand-cleared, and the resolved values the six members ended up at;
(b2) the same selection made on top of the g1atrim stance, which the stance
    itself conflicts with (it explicitly arms the anode jet and the mesh
    accommodation) -- one collected refusal, not a three-deep cascade;
(c) an explicitly-conflicted ``kinetic_dvm`` arm.

Run from ``<checkout>/cablp`` with PYTHONPATH pointing at that same
directory. Prints to stdout; exits 0 when every case behaved as described
and 1 otherwise.
"""

import sys
import warnings
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
if str(_SCRIPTS.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS.parent))

from cablp.solvers._sim1d import (  # noqa: E402
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    LAPDSim1D,
    default_config,
)
from stance_config import stance_config  # noqa: E402

# The stance arms are built at a coarse mesh: the resolver runs before any
# geometry is touched, so the refusals below are mesh-independent and this
# only keeps the constructing case cheap.
NX_STANCE = 60
NX_DEFAULT = 40


def banner(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def build(params, flags):
    """Return the exception text, or None when the solver constructed."""
    try:
        LAPDSim1D(input_dict=params, input_flags=flags)
    except ValueError as exc:
        return str(exc)
    return None


def case_a():
    banner("(a) g1atrim + neutral_wall_momentum_partition + sigma_hehe")
    params, flags = stance_config("g1atrim")
    params["nx"] = NX_STANCE
    flags["neutral_wall_momentum_partition"] = True
    params["neutral_wall_partition_sigma_hehe_cm2"] = 1.26e-15
    message = build(params, flags)
    print(message if message is not None else "CONSTRUCTED (unexpected)")
    return message is not None and "kinetic_two_moment" in message


def case_b():
    banner("(b) default_config() + neutral_model='kinetic_dvm', nothing cleared")
    params, flags = default_config()
    params["nx"] = NX_DEFAULT
    params["neutral_model"] = "kinetic_dvm"
    try:
        sim = LAPDSim1D(input_dict=params, input_flags=flags)
    except ValueError as exc:
        print(str(exc))
        return False
    print("CONSTRUCTED. Resolved member values:")
    got_params, got_flags = sim.get_config()
    ok = True
    for space, key, required, _why in KINETIC_DVM_INCOMPATIBLE_DEFAULTS:
        got = (got_flags if space == "flags" else got_params).get(key)
        agree = got == required or (got is None and required is None)
        ok = ok and agree
        print(
            f"  {space}:{key} = {got!r} "
            f"(required {required!r}) {'ok' if agree else 'MISMATCH'}"
        )
    return ok


def case_b2():
    banner("(b2) g1atrim + neutral_model='kinetic_dvm' (stance conflicts)")
    params, flags = stance_config("g1atrim")
    params["nx"] = NX_STANCE
    params["neutral_model"] = "kinetic_dvm"
    message = build(params, flags)
    print(message if message is not None else "CONSTRUCTED (unexpected)")
    return message is not None and "anode_neutral_jet" in message


def case_c():
    banner(
        "(c) default_config() + kinetic_dvm + explicit "
        "flags:neutral_hot_birth_drift=True"
    )
    params, flags = default_config()
    params["nx"] = NX_DEFAULT
    params["neutral_model"] = "kinetic_dvm"
    flags["neutral_hot_birth_drift"] = True
    message = build(params, flags)
    print(message if message is not None else "CONSTRUCTED (unexpected)")
    return message is not None and "neutral_hot_birth_drift" in message


def main():
    warnings.simplefilter("ignore")
    results = [
        ("a", case_a()),
        ("b", case_b()),
        ("b2", case_b2()),
        ("c", case_c()),
    ]
    banner("SUMMARY")
    for label, ok in results:
        print(f"  case {label}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _label, ok in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
