"""B0c measurement harness: DVM neutral-cadence and velocity-grid convergence.

Implements the ratified B0c registration (R1-R16) for the transient
deterministic velocity-grid neutral arm, ``neutral_model = "kinetic_dvm"``.

WHAT IS BEING MEASURED. Between neutral ticks the plasma consumes a
ZERO-ORDER HOLD of the DVM -- republished ``nn``/``nn_a`` and tick-frozen
``M``/``Ei`` transfer rates -- which is a Lie-type splitting whose global
error is ``O(dt_n)``, FIRST order by design. Nothing higher is claimable,
so the registered order band is centred on 1 and the measurement asks
whether the coupling behaves as designed and at which cadence the
truncation error falls under the registered 1 % bar. Particle conservation
is cadence-independent BY CONSTRUCTION (the counted handshake), so the
conservation rows here are a consistency monitor, never the convergence
signal.

STRUCTURE (registration item in brackets).

  --plan            print the full arm plan -- arms x knobs x N_k -- and
                    every registered band, tolerance and observable, WITHOUT
                    solving anything.                              [R1-R7]
  --arm NAME        run exactly ONE arm to its sample time and bank its
                    per-arm npz -- including its observables at EVERY neutral
                    tick -- so a runner can serialize the ladder in a single
                    lane. Evaluates and prints this arm's own per-arm items
                    (R11, R12, R13, R14 inputs).            [R2-R6, R11-R14]
  --table           assemble the committed markdown table and the verdicts
                    from the banked per-arm npz files, evaluating R8-R14 and
                    printing PASS / FAIL / UNDERDETERMINED per registered
                    item with the registered consequence text on any
                    failure. REFUSES any arm banked below the current
                    ``CAPTURE_SCHEMA``.                           [R8-R16]
  --arm-extra K=V   with ``--arm``: layer ONE more config knob on top of that
                    arm's three registered ones, repeatable. The key is
                    resolved against BOTH config templates before anything
                    solves, so a key filed into the wrong namespace fails
                    here and not at solver construction. The result is a
                    CONTROL, not the registered rung: it banks under
                    ``b0c_extra_*`` and ``--table`` refuses it.

SAMPLING [R2, as amended]. The registered sampling is the COMMON ABSOLUTE
t*: --table interpolates every arm's per-tick capture to the one absolute
time ``t_engage + t*``. Reading each arm at its own N_k-th tick instead
leaves a sample-time mismatch across the ladder, which contaminates the very
pair errors R8's order fit is made from; that reading is SUPERSEDED and
reachable only as ``--sampling tick-count``, so the pre-amendment numbers
stay reproducible. Both readings come from the SAME banked arms -- the
sampling is a table-time choice, never a re-run.

R8 ESTIMATOR [R8, as amended]. The registered fit is
``--r8-fit amended``: errors against the converged reference rung
``cad_1.5625e-06``, fitted as ``log(e_k)`` against ``log(h_k - h_ref)``
over the rungs whose NOMINAL cadence is at or below 1.25e-5 s. That
abscissa carries the FINITE-REFERENCE correction -- the reference is a
finite h, so a first-order response gives ``e_k ~ h_k - h_ref`` and not
``~ h_k`` -- and the uncorrected log-e/log-h slope is reported beside it
as a diagnostic. The coarser rungs
are reported but EXCLUDED -- they sit in the ion-debit shortfall /
Ti-collapse regime, a different regime from the one the fit models -- and
the pre-amendment reference is superseded because it was itself
unconverged. ``--r8-fit pre-amendment`` re-forms the superseded
successive-pair fit over the whole ladder. Like the sampling, the
estimator is a table-time choice over the SAME banked arms.

NOTHING RUNS AT IMPORT. Every solve is reached only through ``main``.

EXIT CODES. 0 = every registered item PASS (or REPORTED); 1 = at least one
FAIL, a step-cap hit, or a refusal; 3 = no FAIL but at least one registered
item UNDERDETERMINED (the R8 sampling-floor branch, a missing rung, an
unestablished cadence of record). A null is a deliverable: an
UNDERDETERMINED verdict names what must be run before the item can be read.

FIXTURE [R1]. The arms are the k2_dvm suite's ``make_sim()`` arm exactly --
``default_config()`` plus that suite's kinetic-compatible base,
``neutral_model="kinetic_dvm"``, ``neutral_kinetic_dvm_exchange`` at the
suite's ``EXCHANGE_MODEL``, production geometry, no g1atrim overlay -- with
the registered knob(s) on top. ``arm_config``, ``make_sim``,
``advance_one_step``, ``CADENCE_S``, ``EXCHANGE_MODEL`` and ``ROUNDOFF_REL``
are IMPORTED from that suite rather than restated here, so this harness and
the coupling-integrity gates can never disagree about what the arm is. Each
arm's config diff against the bare fixture is printed and ASSERTED to be
exactly its registered knobs.

MID-PORT CELL [R7]. Resolved at harness-write time from
``compare_sim1d_es1``'s port map: that scorer maps a port to a cell by
``argmin |z_model - z_port|`` over the run's ``z_cm``. The campaign's
mid-port is ES port 29 at ``z = 1045.15 cm``; on this fixture's geometry
(72 cells) that is cell index 39, ``z = 1054.75 cm``. The index is printed
in the table header and re-asserted against the live geometry on every arm,
so a geometry move cannot silently re-point the observable.

ARTIFACTS. ``--arm`` writes one ``.npz`` per arm (gitignored campaign
evidence); ``--table`` writes the registered markdown deliverable
``b0c_convergence_table.md``. The transcripts are the other artifact; the
caller redirects them.

Usage (from <checkout>/cablp, with PYTHONPATH set to that same cablp):
    python scripts/verify_sim1d_b0c_cadence.py --plan
    python scripts/verify_sim1d_b0c_cadence.py --arm base_2.5e-05_16x6
    python scripts/verify_sim1d_b0c_cadence.py --table
"""
import argparse
import ast
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

from cablp.solvers._sim1d import TimestepRejectionError
from cablp.solvers._sim1d.physics.kinetic_dvm import (
    ledger_energy_residual,
    ledger_residual,
)

# The k2_dvm suite owns the fixture. Import it; do not restate it.
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from verify_sim1d_k2_dvm import (  # noqa: E402
    CADENCE_S,
    EXCHANGE_MODEL,
    ROUNDOFF_REL,
    advance_one_step,
    arm_config,
    make_sim,
)

# --------------------------------------------------------------- registration

# [R7] Mid-port observable. Port number and probe z come from the ES overlay
# schema that ``compare_sim1d_es1`` reads (``scripts/data/es1_sim1d_overlay.npz``:
# port = [11 21 29 41 50], z_cm = [470.05 789.55 1045.15 1428.55 1716.10]);
# the cell index is that scorer's own ``argmin |z_model - z|`` rule evaluated
# on this fixture's geometry at harness-write time (2026-08-24).
MID_PORT_NUMBER = 29
MID_PORT_Z_CM = 1045.15
MID_PORT_CELL = 39
MID_PORT_CELL_Z_CM = 1054.75

# [R2] Sample horizon. t* = t_engage + 2.0 ms; 2.0 ms = 40 x 5.0e-5 exactly,
# which makes the tick-count synchronisation exact across the ladder.
# R14 provides for ONE doubling to 4.0 ms on a burn-through miss.
T_STAR_MS_DEFAULT = 2.0
T_STAR_MS_DOUBLED = 4.0
T_STAR_MS_ALLOWED = (T_STAR_MS_DEFAULT, T_STAR_MS_DOUBLED)

# [R2, AMENDED] Sampling. The REGISTERED sampling is the COMMON ABSOLUTE t*:
# every arm captures its observables at EVERY tick and the table interpolates
# them to the one absolute time t_engage + t*. Sampling each arm at its own
# N_k-th tick instead leaves a sample-time mismatch across the ladder, and
# that mismatch -- not discretization -- dominated the pair errors the order
# fit is formed from. The tick-count reading is SUPERSEDED BY THE AMENDMENT
# and survives only under an explicitly named --sampling, so the numbers on
# the record stay reproducible.
#: Per-arm CAPTURE SCHEMA, written into every banked npz's meta and reported
#: in the table. It versions what the per-tick capture CONTAINS, which is a
#: different question from which sampling the table reads (that is
#: ``SAMPLING_REGISTERED``). Growth is additive: every key an older schema
#: wrote is still written, with the same dtype, so a npz banked at an earlier
#: schema stays LOADABLE and only the rows it never carried are unavailable.
#: ``--table`` nonetheless REFUSES an arm below the current schema, because
#: it now reports the hold-debt and ``nu_E`` rows per arm: an arm that cannot
#: supply them would be published as ``n/a``, which puts a missing
#: measurement and a benign one under the same mark.
#:
#:   1  the original common-t capture: per-tick nn, nn_a, nn_midport, the
#:      booked CUMULATIVE Ei/M, Tn, Ti, Te.
#:   2  adds the rows nu_E cannot be reconstructed without -- per-tick plasma
#:      ``n`` and ``Ei``, the tick's FROZEN booked Ei RATE per cell (the
#:      cumulative rows cannot give it: they are integrals over unequal
#:      ticks), and, from the exponential transfer hold, the per-cell hold
#:      debt and nu_E * cadence. So ``nu_E = |booked Ei| / Ei`` and the hold's
#:      own stability parameter are both readable from a banked arm.
CAPTURE_SCHEMA = 2

SAMPLING_REGISTERED = "common-t"
SAMPLING_SUPERSEDED = "tick-count"
SAMPLING_MODES = (SAMPLING_REGISTERED, SAMPLING_SUPERSEDED)
AMENDMENT_LABEL = "per the common-t sampling amendment"

# [R2] Per-arm step cap. Hitting it is a loud FAIL, not a truncated arm.
MAX_STEPS_PER_ARM = 200_000

# [R2] Effective cadence is recorded per arm; if it is more than this far
# from nominal, the EFFECTIVE value is the one R8/R9 use as h_k.
CADENCE_DEV_TOL = 0.01

# [R8] Order bands. The COARSE pair carries the wider band (O(h^2)
# contamination); the FINE pair carries the sampling-floor guard.
ORDER_BAND_COARSE = (0.7, 1.3)
ORDER_BAND_FINE = (0.8, 1.25)
SAMPLING_FLOOR_FACTOR = 10.0

# [R8, AMENDED] The R8 ESTIMATOR is selectable, and the amended one is
# the default. Two defects were identified in the pre-amendment fit and the
# amendment addresses exactly them:
#
#   (i)  the coarse rungs are a DIFFERENT REGIME. At h >= 2.5e-5 the arm runs
#        into the ion-debit shortfall / Ti collapse, so those rungs are not
#        samples of the same smooth truncation-error curve the fit models.
#        They are REPORTED but EXCLUDED from the fit domain.
#   (ii) the pre-amendment reference was UNCONVERGED (the finest rung
#        over-books 5.2 %), which biases every error formed against it.
#        The amended errors are formed against a rung one halving finer
#        still, ``cad_1.5625e-06``.
#
# The amended estimator is therefore a REFERENCE-CORRECTED log-e/log-h fit
# over the fit domain, not the successive-pair ratio. The pre-amendment
# successive-pair path is kept byte-reachable under ``--r8-fit`` exactly as
# the superseded ``--sampling tick-count`` reading is, so the numbers already
# on the record stay reproducible.
R8_FIT_AMENDED = "amended"
R8_FIT_PRE_AMENDMENT = "pre-amendment"
R8_FIT_MODES = (R8_FIT_AMENDED, R8_FIT_PRE_AMENDMENT)
R8_AMENDMENT_LABEL = "per the reference-corrected R8 amendment"

#: [R8, amended] Fit domain: a rung enters the fit iff its NOMINAL cadence is
#: at or below this. Nominal, not effective: the amendment names the RUNGS,
#: and a rung's effective cadence can sit a couple of percent above its
#: nominal (the ticks fire at the first accepted-step boundary >= next_s).
#: The h used IN the fit is still the effective one -- that is unchanged.
R8_FIT_MAX_H = 1.25e-5

#: [R8, amended] The converged reference rung the amended errors are formed
#: against: e_k = ||u_k - u_ref|| at the registered common-t sampling.
R8_REFERENCE_ARM = "cad_1.5625e-06"

#: [R8, amended] Verdict label carried by every rung the amendment excludes.
R8_EXCLUDED_LABEL = "excluded: shortfall regime"

# [R9] Cadence-of-record bar on the first-order-corrected proxy true error.
EHAT_TOL = 0.01

# [R10] Grid criterion bar.
GRID_TOL = 0.01

# [R11] I6 independence band: the suite's ROUNDOFF_REL at (16,6), scaled by
# sqrt(N_bins / 96) so a finer velocity grid's longer summations do not
# false-fail without slackening the base.
I6_REF_BINS = 96

# [R12] Transfer identity tolerance (applied_cum + debt - booked_cum).
R12_TOL = 1.0e-12

# [R13] Debt gates at the cadence-of-record arm; all arms report.
DEBT_TOL = {"Ei": 1.0e-3, "M": 1.0e-3, "ion": 1.0e-6}

# [R14 / NV3] Burn-through non-vacuity on the shipped arm.
BURN_THROUGH_MIN = 0.1

# [R15] Registered consequences, quoted verbatim on the failures they name.
CONSEQUENCE = {
    "a": (
        "R15(a): R8 fails after the underdetermined branch -> the "
        "hold-coupling is not behaving as designed; B1+ is BLOCKED and the "
        "diagnosis is the deliverable."
    ),
    "b": (
        "R15(b): order passes but the shipped cadence fails R9/R13 -> the "
        "cadence of record is the finer passing rung; a config default "
        "change flows through the lifecycle before any quoted DVM number."
    ),
    "c": (
        "R15(c): (16,6) fails R10 -> the named grid is the passing rung; "
        "production and the item-41 retest run there, cost disclosed."
    ),
    "d": (
        "R15(d): all pass -> the config.py PROVISIONAL sentence on "
        "neutral_kinetic_dvm_cadence_s is amended to cite the committed "
        "table; provenance entry class DERIVED."
    ),
}

# [R7] The GATED observable set: (fed-back moments) u (quoted quantities),
# plus Ti as the downstream integrator that catches staircase aliasing.
# ``kind``: "l2" plain relative L2 over cells, "scalar" a single cell,
# "wl2" an nn-WEIGHTED relative L2 (weights taken from the finer arm's own
# published column density).
GATED = (
    ("O1", "nn", "column nn profile", "l2"),
    ("O2", "nn_a", "annulus nn_a profile", "l2"),
    ("O3", "nn_midport", f"mid-port nn (cell {MID_PORT_CELL})", "scalar"),
    ("O4", "Ei_booked_cum", "Ei_booked_cum profile", "l2"),
    ("O5", "M_booked_cum", "M_booked_cum profile", "l2"),
    ("O6", "Tn_col_eV", "column Tn (nn-weighted)", "wl2"),
    ("O7", "Ti", "fluid Ti profile", "l2"),
)
GATED_KEYS = tuple(k for _tag, k, _label, _kind in GATED)
GATED_KIND = {k: kind for _tag, k, _label, kind in GATED}
GATED_TAG = {k: tag for tag, k, _label, _kind in GATED}
GATED_LABEL = {k: label for _tag, k, label, _kind in GATED}

# Reported, never gated.
REPORTED = (("Te", "electron temperature profile", "l2"),)


class ArmSpec:
    """One registered arm: its knobs, its ladders, and its tick count.

    ``extras`` carries EXTRA-KNOB settings layered on top of the three
    registered ones -- the ``--arm-extra`` route. It is empty on every
    registered arm in ``ARMS``, so a spec built from the ladder is exactly
    what it always was; only a spec derived through :meth:`with_extras`
    carries any.
    """

    def __init__(self, name, cadence_s, nvz, nvp, ladders, conditional=False,
                 note="", extras=None):
        self.name = name
        self.cadence_s = float(cadence_s)
        self.nvz = int(nvz)
        self.nvp = int(nvp)
        self.ladders = tuple(ladders)
        self.conditional = bool(conditional)
        self.note = note
        self.extras = dict(extras or {})

    @property
    def knobs(self):
        """The registered knob settings this arm names, as config keys.

        Any ``extras`` are layered on top, so the R1 "config diff is exactly
        the registered knob set" assertion in :func:`run_arm` is made against
        the knobs the arm ACTUALLY names -- an extra knob is registered with
        that assertion, never hidden from it.
        """
        knobs = {
            "neutral_kinetic_dvm_cadence_s": self.cadence_s,
            "neutral_kinetic_dvm_nvz": self.nvz,
            "neutral_kinetic_dvm_nvp": self.nvp,
        }
        knobs.update(self.extras)
        return knobs

    def with_extras(self, extras):
        """Return a COPY of this spec carrying ``extras`` as extra knobs.

        A copy, never a mutation: ``ARMS`` holds one shared spec per
        registered arm and an in-place edit would redefine the ladder for
        everything else in the process.
        """
        return ArmSpec(self.name, self.cadence_s, self.nvz, self.nvp,
                       self.ladders, conditional=self.conditional,
                       note=self.note, extras=extras)

    def n_updates(self, t_star_ms):
        """[R2] N_k = t* / cadence_k, exact by construction of the ladder."""
        target = float(t_star_ms) * 1.0e-3
        n = target / self.cadence_s
        n_int = int(round(n))
        if abs(n - n_int) > 1.0e-9:
            raise ValueError(
                f"arm {self.name}: t* = {t_star_ms} ms is not an integer "
                f"number of {self.cadence_s:g} s ticks (N = {n!r}); the "
                "registered horizon requires exact tick-count sync"
            )
        return n_int


#: The three knobs the REGISTRATION itself names. ``--arm-extra`` refuses to
#: touch any of them: they are what the ladder is made of, so setting one
#: through the extra route would silently redefine which rung the arm is
#: while it kept the rung's name.
REGISTERED_KNOB_KEYS = (
    "neutral_kinetic_dvm_cadence_s",
    "neutral_kinetic_dvm_nvz",
    "neutral_kinetic_dvm_nvp",
)


# [R3] Cadence ladder, coarse -> fine, all at (nvz, nvp) = (16, 6).
# [R5] Grid ladder, coarse -> fine, all at cadence 2.5e-5.
# The (2.5e-5, 16, 6) point is a member of BOTH ladders and is ONE arm.
_BASE_NAME = "base_2.5e-05_16x6"
_CAD_COND_NAME = "cad_3.125e-06"
_CAD_REF_NAME = R8_REFERENCE_ARM
_GRID_COND_NAME = "grid_128x48"
_GRID_COND2_NAME = "grid_256x96"
_CROSS_NAME = "cross_6.25e-06_32x12"

ARMS = {}
for _spec in (
    ArmSpec("cad_5.0e-05", 5.0e-5, 16, 6, ("cadence",),
            note="coarsest cadence rung; R4 stability-escape candidate"),
    ArmSpec(_BASE_NAME, 2.5e-5, 16, 6, ("cadence", "grid"),
            note="SHIPPED cadence; shared cadence/grid ladder member"),
    ArmSpec("cad_1.25e-05", 1.25e-5, 16, 6, ("cadence",), note=""),
    ArmSpec("cad_6.25e-06", 6.25e-6, 16, 6, ("cadence",),
            note="finest unconditional cadence rung; R9 reference"),
    ArmSpec(_CAD_COND_NAME, 3.125e-6, 16, 6, ("cadence",), conditional=True,
            note="CONDITIONAL: R8-underdetermined or R4 stability escape"),
    ArmSpec(_CAD_REF_NAME, 1.5625e-6, 16, 6, ("cadence",),
            note="R8-amendment converged reference rung"),
    ArmSpec("grid_32x12", 2.5e-5, 32, 12, ("grid",), note=""),
    ArmSpec("grid_64x24", 2.5e-5, 64, 24, ("grid",), note=""),
    ArmSpec(_GRID_COND_NAME, 2.5e-5, 128, 48, ("grid",), conditional=True,
            note="CONDITIONAL: only if (32,12)->(64,24) fails R10"),
    ArmSpec(_GRID_COND2_NAME, 2.5e-5, 256, 96, ("grid",), conditional=True,
            note="CONDITIONAL: R10 ladder extension, Tom-ratified "
                 "2026-08-26 -- only if (64,24)->(128,48) fails R10"),
    ArmSpec(_CROSS_NAME, 6.25e-6, 32, 12, ("cross",),
            note="R6 cross arm: REPORTED, NOT GATED"),
):
    ARMS[_spec.name] = _spec

CADENCE_LADDER = ("cad_5.0e-05", _BASE_NAME, "cad_1.25e-05", "cad_6.25e-06")
GRID_LADDER = (_BASE_NAME, "grid_32x12", "grid_64x24")

# [R9] "Shipped 2.5e-5 confirmed iff it is that rung or finer."  [R14/NV3]
# the burn-through statement is made on the shipped arm.
SHIPPED_ARM = _BASE_NAME


# ------------------------------------------------------------------- helpers


def i6_band(nvz, nvp):
    """[R11] The scaled I6 independence band for a (nvz, nvp) velocity grid."""
    return ROUNDOFF_REL * max(1.0, math.sqrt((nvz * nvp) / I6_REF_BINS))


def config_diff(overrides, reference=None):
    """Return {key: (reference_value, arm_value)} for an arm's config diff.

    ``reference`` defaults to the BARE fixture -- ``arm_config()`` with no
    overrides -- which is what R1's "config diff is exactly the registered
    knob(s)" assertion is made against.
    """
    ref_d, ref_fl = arm_config() if reference is None else arm_config(**reference)
    arm_d, arm_fl = arm_config(**overrides)
    diff = {}
    for ref, arm in ((ref_d, arm_d), (ref_fl, arm_fl)):
        for key in sorted(set(ref) | set(arm)):
            a = ref.get(key, "<absent>")
            b = arm.get(key, "<absent>")
            if a != b:
                diff[key] = (a, b)
    return diff


def parse_arm_extras(items):
    """Parse ``--arm-extra key=value`` strings into ``(extras, namespaces)``.

    THE NAMESPACE IS RESOLVED HERE, at argument-parse time. ``input_dict``
    and ``input_flags`` are separate namespaces that share no key, and
    ``arm_config`` files an override into flags iff the flags template owns
    it -- so validating a key against those same two templates is validating
    the routing that will actually happen. A key neither template owns is a
    misfile, and refusing it here turns a run-time construction failure into
    an argument error the caller sees before anything solves.

    Values are read as Python literals where they are ones (``2.5e-5``,
    ``16``, ``True``, ``None``) and kept as bare strings otherwise (``zoh``,
    ``rates``) -- between them that is the set of value kinds the config
    templates hold.
    """
    ref_d, ref_fl = arm_config()
    extras = {}
    namespaces = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(
                f"REFUSED: --arm-extra {item!r} is not of the form key=value"
            )
        key, _, raw = item.partition("=")
        key = key.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        if key in REGISTERED_KNOB_KEYS:
            raise SystemExit(
                f"REFUSED: --arm-extra {key!r} is one of the three REGISTERED "
                f"knobs {list(REGISTERED_KNOB_KEYS)} the ladder is built out "
                "of. Setting one here would redefine WHICH RUNG the arm is "
                "while it kept that rung's name; choose the rung with --arm."
            )
        if key in extras:
            raise SystemExit(
                f"REFUSED: --arm-extra {key!r} given more than once "
                f"({extras[key]!r} then {value!r})"
            )
        if key in ref_fl:
            namespaces[key] = "input_flags"
        elif key in ref_d:
            namespaces[key] = "input_dict"
        else:
            raise SystemExit(
                f"REFUSED: --arm-extra {key!r} is owned by NEITHER config "
                "template -- it is in neither input_dict nor input_flags, so "
                "it would be a silent/inert control. Check which template "
                "owns the key in core/config.py and spell it exactly."
            )
        extras[key] = value
    return extras, namespaces


def rel_error(a, b, kind, weights=None):
    """Relative error of ``a`` against ``b``; ``b`` is the FINER arm [R7]."""
    if kind == "scalar":
        den = abs(float(b))
        if den == 0.0:
            return float("nan")
        return abs(float(a) - float(b)) / den
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if kind == "wl2":
        w = np.asarray(weights, dtype=float)
        den = float(np.sum(w * b * b))
        if den <= 0.0:
            return float("nan")
        return math.sqrt(float(np.sum(w * (a - b) ** 2)) / den)
    den = float(np.linalg.norm(b))
    if den == 0.0:
        return float("nan")
    return float(np.linalg.norm(a - b)) / den


def bracket_index(tick_time, t_target):
    """Return ``(i, w)`` with ``t_target = (1-w)*t[i-1] + w*t[i]`` [R2].

    ``i`` names the tick interval that brackets ``t_target``. It is clamped
    to a real interval at both ends, so a target outside the banked tick
    range is EXTRAPOLATED along the nearest interval rather than silently
    clipped to an endpoint: ``0 <= w <= 1`` is the bracketing test, and the
    table reports it per arm instead of assuming it.
    """
    t = np.asarray(tick_time, dtype=float)
    if t.size < 2:
        raise SystemExit(
            "REFUSED: fewer than two neutral ticks banked, so no interval "
            "exists to interpolate the common-t sample onto"
        )
    i = int(np.searchsorted(t, t_target))
    i = min(max(i, 1), int(t.size) - 1)
    t0 = float(t[i - 1])
    t1 = float(t[i])
    return i, (float(t_target) - t0) / (t1 - t0)


def sampled_tick_row(meta, arrays, key):
    """Return one per-tick capture row read at this arm's sampled time.

    The ONE place the per-tick capture is read, so every row the table
    samples -- observables, ``nu_E * cadence``, the hold debt -- is read at
    the same time by the same rule. Under the registered common-t sampling
    the row is interpolated onto the absolute ``t_engage + t*``; under the
    superseded tick-count sampling it is the arm's own last tick.
    """
    series = np.asarray(arrays[key], dtype=float)
    weight = meta.get("_sample_weight")
    if weight is None:
        return series[-1]
    index, _ = bracket_index(arrays["tick_time"], meta["_t_sample"])
    return (1.0 - weight) * series[index - 1] + weight * series[index]


def sampled_nu_e_cadence(meta, arrays):
    """Return max over cells of ``nu_E * cadence`` at the sampled time.

    ``None`` on an arm banked before ``CAPTURE_SCHEMA`` 2, which did not
    carry the row -- reported as such rather than as a reassuring zero.
    ``--table`` refuses such an arm outright, so this branch is reached only
    by a direct caller.
    """
    key = "tickobs_nu_E_cadence"
    if key not in arrays:
        return None
    row = sampled_tick_row(meta, arrays, key)
    return float(np.max(np.abs(np.asarray(row, dtype=float))))


def hold_debt_rows(meta, arrays):
    """The RR4 hold-debt meter and mid-port ``nu_E * cadence`` for one arm.

    The HOLD DEBT is ``dvm.Ei_hold_debt`` / ``M_hold_debt``, the exponential
    transfer hold's own undelivered remainder in the schema-2 identity
    ``applied_cum + debt + hold_debt == booked_cum``. It is a DIFFERENT
    quantity from the ``Ei_debt`` / ``M_debt`` the R13 table gates: R13's
    debt is what the ion channel could not pay, this is what the hold has
    not yet released. Both are reported; only R13's is gated.

    Five numbers per arm:

      * the two RATIOS ``sum|hold| / sum|booked_cum|`` at the sampled time,
        one per channel, formed from the SAMPLED per-cell rows;
      * the max over ticks of that Ei ratio, formed from the raw tick series
        (its peak is generally not at t*);
      * the max per-tick RESIDUAL ``|d(sum|Ei_hold|)| / sum|Ei_booked_cum|``,
        which is the tick-to-tick movement of the hold rather than its
        level -- a level that is large but static is a lagging hold, a
        residual that is large is a hold still swinging at t*;
      * ``nu_E * cadence`` at the mid-port cell, the companion to the
        max-over-cells value the arms table already carries: the max says
        whether the hold is stable ANYWHERE, the mid-port value says whether
        it is stable at the cell the campaign actually scores.

    The arithmetic is the post-hoc RR5 reading (``b0crr_rr5_holddebt.py``)
    folded in unchanged, so the harness and that script cannot disagree.
    """
    def sampled(key):
        return np.asarray(sampled_tick_row(meta, arrays, key), dtype=float)

    def ratio(hold_key, booked_key):
        hold = np.abs(sampled(hold_key)).sum()
        booked = np.abs(sampled(booked_key)).sum()
        if booked <= 0.0:
            return float("nan")
        return float(hold / booked)

    hold_series = np.abs(
        np.asarray(arrays["tickobs_Ei_hold_debt"], dtype=float)
    ).sum(axis=1)
    booked_series = np.abs(
        np.asarray(arrays["tickobs_Ei_booked_cum"], dtype=float)
    ).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_tick = np.where(booked_series > 0.0,
                            hold_series / booked_series, np.nan)
        residual = np.where(booked_series[1:] > 0.0,
                            np.abs(np.diff(hold_series)) / booked_series[1:],
                            np.nan)
    nu = sampled("tickobs_nu_E_cadence")
    return {
        "Ei_ratio": ratio("tickobs_Ei_hold_debt", "tickobs_Ei_booked_cum"),
        "M_ratio": ratio("tickobs_M_hold_debt", "tickobs_M_booked_cum"),
        "Ei_ratio_max_ticks": float(np.nanmax(per_tick)),
        "residual_max_ticks": float(np.nanmax(residual)),
        "nu_e_cadence_midport": float(nu[MID_PORT_CELL]),
    }


def _nu_e_cell(meta):
    value = meta.get("_nu_e_cadence")
    if value is None:
        return f"n/a (capture schema {meta['capture_schema']} < {CAPTURE_SCHEMA})"
    marker = " **unstable under zoh**" if value > 2.0 else ""
    return f"{value:.4g}{marker}"


def magnitude(u, kind):
    """Scale of an observable row, for the NV2 zero-at-roundoff test."""
    if kind == "scalar":
        return abs(float(u))
    return float(np.max(np.abs(np.asarray(u, dtype=float))))


# ----------------------------------------------------------------- arm runner


def observables(sim):
    """Return the gated + reported rows at the current accepted state."""
    state = sim.state
    derived = sim.derived
    dvm = sim._dvm
    nn = np.asarray(state.nn, dtype=float).copy()
    return {
        "nn": nn,
        "nn_a": np.asarray(state.nn_a, dtype=float).copy(),
        "nn_midport": float(nn[MID_PORT_CELL]),
        "Ei_booked_cum": np.asarray(dvm.Ei_booked_cum, dtype=float).copy(),
        "M_booked_cum": np.asarray(dvm.M_booked_cum, dtype=float).copy(),
        "Tn_col_eV": np.asarray(dvm.Tn_col_eV, dtype=float).copy(),
        "Ti": np.asarray(derived.Ti, dtype=float).copy(),
        "Te": np.asarray(derived.Te, dtype=float).copy(),
        # --- CAPTURE_SCHEMA 2. Everything above is unchanged, in name, order
        # and dtype; everything below is what a banked arm needed and did not
        # have. nu_E = |booked Ei rate| / Ei is not reconstructible from the
        # rows above at any schema: the CUMULATIVE booked rows are integrals
        # over unequal ticks, and neither n nor Ei was captured at all.
        "n": np.asarray(state.n, dtype=float).copy(),
        "Ei": np.asarray(state.Ei, dtype=float).copy(),
        "Ei_booked_rate": np.asarray(dvm.Ei_transfer, dtype=float).copy(),
        "M_booked_rate": np.asarray(dvm.M_transfer, dtype=float).copy(),
        "Ei_hold_debt": np.asarray(dvm.Ei_hold_debt, dtype=float).copy(),
        "M_hold_debt": np.asarray(dvm.M_hold_debt, dtype=float).copy(),
        # The transfer hold's own stability parameter, per cell: the pair
        # collision frequency the tick booked, times the neutral cadence.
        # Above ~2 the superseded zero-order hold is unstable, so this is the
        # number that says whether a cadence rung was resolving the coupling
        # or fighting it (NUMERICS.md, "The DVM transfer hold").
        "nu_E_cadence": (
            np.asarray(dvm.nu_pair, dtype=float) * float(sim._dvm_cadence_s)
        ).copy(),
    }


def transfer_identity(sim):
    """[R12] applied_cum + debt - booked_cum, per channel, relative."""
    ledger = sim.dvm_transfer_ledger()
    return {name: float(ledger[name]["rel"]) for name in ("M", "Ei", "N")}


def debt_ratios(sim):
    """[R13] sum|debt| / sum|booked_cum| per channel."""
    dvm = sim._dvm
    out = {}
    for tag, debt, booked in (
        ("Ei", dvm.Ei_debt, dvm.Ei_booked_cum),
        ("M", dvm.M_debt, dvm.M_booked_cum),
        ("ion", dvm.ion_debt, dvm.ion_booked_cum),
    ):
        num = float(np.sum(np.abs(np.asarray(debt, dtype=float))))
        den = float(np.sum(np.abs(np.asarray(booked, dtype=float))))
        out[tag] = (num / den) if den > 0.0 else (0.0 if num == 0.0
                                                 else float("inf"))
    return out


def floor_census(sim, obs):
    """Floor-republish census: cells sitting exactly at the nn floor."""
    floor = float(sim._floors["nn"])
    nn = np.asarray(obs["nn"], dtype=float)
    nn_a = np.asarray(obs["nn_a"], dtype=float)
    return {
        "nn_floor": floor,
        "nn_floor_cells": int(np.count_nonzero(nn <= floor)),
        "nn_a_floor_cells": int(np.count_nonzero(nn_a <= floor)),
    }


def run_arm(spec, n_updates, verbose=True):
    """Run ONE arm to ``n_updates`` neutral ticks and return its record.

    Returns a dict of everything the table mode needs. Arm death at dt_min is
    a RESULT (R4), recorded as ``status = "dead_dt_min"``, not an exception.
    """
    overrides = dict(spec.knobs)
    diff = config_diff(overrides)
    expected = {k: v for k, v in spec.knobs.items()
                if k in diff and diff[k][1] == v}
    if set(diff) != set(expected):
        raise ValueError(
            f"[R1] arm {spec.name}: config diff against the bare fixture is "
            f"{sorted(diff)}, which is not exactly the registered knob set "
            f"{sorted(expected)}. An unregistered delta makes the arm "
            "uninterpretable; refusing to run it."
        )

    sim = make_sim(**overrides)

    z_cm = np.asarray(sim._geometry.z_cm, dtype=float)
    resolved = int(np.argmin(np.abs(z_cm - MID_PORT_Z_CM)))
    if resolved != MID_PORT_CELL:
        raise ValueError(
            f"[R7] the mid-port map moved: ES port {MID_PORT_NUMBER} at "
            f"z = {MID_PORT_Z_CM} cm now resolves to cell {resolved} "
            f"(z = {z_cm[resolved]:.4g} cm), not the harness-write-time "
            f"cell {MID_PORT_CELL}. The registered observable would silently "
            "change meaning; re-resolve and re-register."
        )
    if (sim._dvm.g.nvz, sim._dvm.g.nvp) != (spec.nvz, spec.nvp):
        raise ValueError(
            f"[R1] arm {spec.name}: built velocity grid is "
            f"({sim._dvm.g.nvz},{sim._dvm.g.nvp}), not the registered "
            f"({spec.nvz},{spec.nvp})"
        )

    # Capture the exact (dt_n, nu_ion) each tick consumed, without touching
    # the solver: wrap the bound update the solver calls.
    dvm_update = sim._dvm.update
    seen = {}

    def _capture(dt, **kwargs):
        seen["dt"] = float(dt)
        seen["nu_ion"] = np.asarray(kwargs["nu_ion"], dtype=float).copy()
        return dvm_update(dt, **kwargs)

    sim._dvm.update = _capture

    tick_time = []
    tick_dt_n = []
    tick_burn = []
    tick_birth_puff = []
    tick_p_dist = []
    tick_p_domain = []
    tick_e_dist = []
    tick_e_domain = []

    tick_obs = []

    t_engage = None
    obs_now = None
    obs_prev = None
    t_now = None
    t_prev = None
    steps = 0
    status = "ok"
    dead_reason = ""
    shortfall_warned = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        while sim._dvm.updates < n_updates and steps < MAX_STEPS_PER_ARM:
            was_engaged = sim._dvm_engaged
            before = sim._dvm.updates
            try:
                advance_one_step(sim)
            except TimestepRejectionError as exc:
                status = "dead_dt_min"
                dead_reason = f"{type(exc).__name__}: {exc}"
                break
            steps += 1
            if t_engage is None and sim._dvm_engaged and not was_engaged:
                t_engage = float(sim._dvm_last_s)
            if sim._dvm.updates > before:
                ledger = sim._dvm.last_ledger
                p_res = ledger_residual(ledger)
                e_res = ledger_energy_residual(ledger)
                tick_time.append(float(sim._time))
                tick_dt_n.append(seen["dt"])
                tick_burn.append(
                    float(np.max(np.abs(seen["nu_ion"]))) * seen["dt"]
                )
                tick_birth_puff.append(float(ledger["birth_puff"]))
                tick_p_dist.append(abs(float(p_res["distribution_rel"])))
                tick_p_domain.append(abs(float(p_res["domain_rel"])))
                tick_e_dist.append(abs(float(e_res["distribution_rel"])))
                tick_e_domain.append(abs(float(e_res["domain_rel"])))
                obs_prev, t_prev = obs_now, t_now
                obs_now, t_now = observables(sim), float(sim._time)
                tick_obs.append(obs_now)
                if verbose and (
                    sim._dvm.updates % max(1, n_updates // 10) == 0
                ):
                    print(
                        f"    tick {sim._dvm.updates}/{n_updates}  "
                        f"t={sim._time:.6g} s  steps={steps}",
                        flush=True,
                    )
        messages = []
        for w in caught:
            text = str(w.message)
            if text not in messages:
                messages.append(text)
        shortfall_warned = any(
            "could not debit the whole ionization" in text
            for text in messages
        )

    hit_cap = steps >= MAX_STEPS_PER_ARM and sim._dvm.updates < n_updates
    if hit_cap:
        status = "max_steps"

    if t_engage is None:
        raise RuntimeError(
            f"arm {spec.name}: the DVM never engaged in {steps} steps; the "
            "registered window does not exist for this arm"
        )
    if obs_now is None:
        raise RuntimeError(
            f"arm {spec.name}: no completed neutral tick in {steps} steps "
            f"(status {status}); nothing to sample"
        )

    n_done = int(sim._dvm.updates)
    eff_cadence = (t_now - t_engage) / n_done if n_done else float("nan")
    ident = transfer_identity(sim)
    debts = debt_ratios(sim)
    floors = floor_census(sim, obs_now)
    band = i6_band(spec.nvz, spec.nvp)

    record = {
        "name": spec.name,
        "status": status,
        "dead_reason": dead_reason,
        "cadence_nominal_s": spec.cadence_s,
        "cadence_effective_s": eff_cadence,
        "nvz": spec.nvz,
        "nvp": spec.nvp,
        "ladders": list(spec.ladders),
        "conditional": spec.conditional,
        # Empty on every registered arm; non-empty marks an --arm-extra
        # control, which --table refuses to read as the registered rung.
        "extras": dict(spec.extras),
        "n_updates_requested": int(n_updates),
        "n_updates_done": n_done,
        "steps": steps,
        "max_steps": MAX_STEPS_PER_ARM,
        "t_engage_s": t_engage,
        "t_star_s": t_now,
        "t_prev_s": t_prev,
        "nz": int(z_cm.size),
        "midport_cell": MID_PORT_CELL,
        "midport_port": MID_PORT_NUMBER,
        "midport_z_cm": float(z_cm[MID_PORT_CELL]),
        "exchange_model": EXCHANGE_MODEL,
        "roundoff_rel": ROUNDOFF_REL,
        "fixture_cadence_s": CADENCE_S,
        "config_diff": {k: [str(v[0]), str(v[1])] for k, v in diff.items()},
        "i6_band": band,
        "i6_particle_distribution_max": (
            max(tick_p_dist) if tick_p_dist else 0.0
        ),
        "i6_particle_domain_max": max(tick_p_domain) if tick_p_domain else 0.0,
        "i6_energy_distribution_max": (
            max(tick_e_dist) if tick_e_dist else 0.0
        ),
        "i6_energy_domain_max": max(tick_e_domain) if tick_e_domain else 0.0,
        "identity_M_rel": ident["M"],
        "identity_Ei_rel": ident["Ei"],
        "identity_N_rel": ident["N"],
        "debt_Ei_ratio": debts["Ei"],
        "debt_M_ratio": debts["M"],
        "debt_ion_ratio": debts["ion"],
        "ion_shortfall_updates": int(sim._dvm.ion_shortfall_updates),
        "ion_shortfall_warned": bool(shortfall_warned),
        "warnings": messages,
        "relax_steps": int(sim._dvm.relax_steps),
        "relax_limited_steps": int(sim._dvm.relax_limited_steps),
        "relax_limited_cells": int(
            np.count_nonzero(sim._dvm.relax_cell_steps)
        ),
        "burn_through_max": max(tick_burn) if tick_burn else 0.0,
        "puff_ticks_active": int(sum(1 for v in tick_birth_puff if v > 0.0)),
        "puff_births_total": float(sum(tick_birth_puff)),
        # neutral_kinetic_dvm_tn_feedback was RETIRED 2026-08-31 (Tom) with
        # its only consumer. The DVM's Tn moment stays an in-process
        # diagnostic that nothing consumes, which is what this row now records.
        "tn_feedback": False,
    }
    record.update(floors)
    arrays = {
        "tick_time": np.asarray(tick_time, dtype=float),
        "tick_dt_n": np.asarray(tick_dt_n, dtype=float),
        "tick_burn": np.asarray(tick_burn, dtype=float),
        "tick_birth_puff": np.asarray(tick_birth_puff, dtype=float),
        "tick_particle_distribution_rel": np.asarray(tick_p_dist, dtype=float),
        "tick_particle_domain_rel": np.asarray(tick_p_domain, dtype=float),
        "tick_energy_distribution_rel": np.asarray(tick_e_dist, dtype=float),
        "tick_energy_domain_rel": np.asarray(tick_e_domain, dtype=float),
        "Ei_debt": np.asarray(sim._dvm.Ei_debt, dtype=float),
        "M_debt": np.asarray(sim._dvm.M_debt, dtype=float),
        "ion_debt": np.asarray(sim._dvm.ion_debt, dtype=float),
        "z_cm": z_cm,
    }
    for key, value in obs_now.items():
        arrays[f"obs_{key}"] = np.asarray(value, dtype=float)
    if obs_prev is not None:
        for key, value in obs_prev.items():
            arrays[f"obsprev_{key}"] = np.asarray(value, dtype=float)
    # Per-tick observable capture: the whole series, so the table can sample
    # every arm at ONE common absolute time instead of at its own N_k-th
    # tick. ``tick_time`` above is the matching time axis.
    for key in obs_now:
        arrays[f"tickobs_{key}"] = np.asarray(
            [o[key] for o in tick_obs], dtype=float
        )
    return record, arrays


def save_arm(path, record, arrays, sanity, t_star_ms):
    meta = dict(record)
    meta["sanity"] = bool(sanity)
    meta["t_star_ms"] = float(t_star_ms)
    meta["capture_schema"] = int(CAPTURE_SCHEMA)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta=np.array(json.dumps(meta)), **arrays)
    return path


def load_arm(path):
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["meta"]))
    arrays = {k: data[k] for k in data.files if k != "meta"}
    meta["_path"] = str(path)
    # A npz written before the marker existed is schema 1 by definition: it
    # carries the original per-tick capture and nothing added since.
    meta.setdefault("capture_schema", 1)
    return meta, arrays


# ------------------------------------------------------------------ plan mode


def plan_lines(t_star_ms):
    out = []
    out.append("B0c arm plan -- registration of record: "
               "the B0c registration, R1-R16, ratified 2026-08-24")
    out.append("")
    out.append("[R1] Fixture: verify_sim1d_k2_dvm.make_sim() exactly "
               f"(exchange={EXCHANGE_MODEL!r}, fixture cadence "
               f"{CADENCE_S:g} s, production geometry, no g1atrim overlay).")
    out.append(f"[R2, AMENDED {AMENDMENT_LABEL}] sampling = "
               f"{SAMPLING_REGISTERED!r}: every arm captures its observables "
               "at EVERY neutral tick and --table interpolates them to the "
               "one absolute time t_engage + t*, so no pair error carries a "
               f"sample-time mismatch. {SAMPLING_SUPERSEDED!r} -- each arm "
               "read at its own N_k-th tick -- is SUPERSEDED and reachable "
               f"only as --sampling {SAMPLING_SUPERSEDED}.")
    out.append(f"[R2] t* = t_engage + {t_star_ms:g} ms; N_k = t*/cadence_k; "
               f"max_steps = {MAX_STEPS_PER_ARM} per arm (hit = loud FAIL); "
               "arms bit-identical until engagement (NV1 asserts equal "
               "t_engage); effective cadence recorded, and used as h_k in R8 "
               f"if it is more than {CADENCE_DEV_TOL:.0%} off nominal.")
    out.append(f"[R7] Mid-port observable: ES port {MID_PORT_NUMBER} at "
               f"z = {MID_PORT_Z_CM} cm -> cell {MID_PORT_CELL} "
               f"(z = {MID_PORT_CELL_Z_CM} cm), resolved at harness-write "
               "time from compare_sim1d_es1's port map and re-asserted "
               "against the live geometry on every arm.")
    out.append("")
    header = (
        f"{'arm':<26} {'ladders':<16} {'cadence [s]':>12} {'nvz':>4} "
        f"{'nvp':>4} {'N_k':>5} {'cond':>5}"
    )
    out.append(header)
    out.append("-" * len(header))
    for name, spec in ARMS.items():
        out.append(
            f"{name:<26} {'+'.join(spec.ladders):<16} "
            f"{spec.cadence_s:>12.6g} {spec.nvz:>4d} {spec.nvp:>4d} "
            f"{spec.n_updates(t_star_ms):>5d} "
            f"{'yes' if spec.conditional else 'no':>5}"
        )
    out.append("")
    out.append("[R3] cadence ladder (coarse -> fine): "
               + " -> ".join(CADENCE_LADDER))
    out.append(f"     conditional 5th rung: {_CAD_COND_NAME} "
               "(R8-underdetermined, or R4 stability escape)")
    out.append("[R5] grid ladder (coarse -> fine): " + " -> ".join(GRID_LADDER))
    out.append(f"     conditional 4th rung: {_GRID_COND_NAME} "
               "(only if (32,12)->(64,24) fails R10)")
    out.append(f"     conditional 5th rung: {_GRID_COND2_NAME} "
               "(only if (64,24)->(128,48) fails R10; R10 ladder extension, "
               "Tom-ratified 2026-08-26)")
    out.append(f"[R6] cross arm (REPORTED, NOT GATED): {_CROSS_NAME}")
    out.append(f"     NB {_BASE_NAME} is ONE arm shared by both ladders; it "
               "is run once and read by both.")
    out.append("")
    out.append("[R1] per-arm config diff against the bare fixture "
               "(asserted to be exactly the registered knobs):")
    for name, spec in ARMS.items():
        diff = config_diff(dict(spec.knobs))
        rendered = ", ".join(
            f"{k}: {v[0]!r} -> {v[1]!r}" for k, v in sorted(diff.items())
        ) or "(none -- this arm IS the bare fixture)"
        out.append(f"  {name:<26} {rendered}")
    out.append("")
    out.append("[R1] per-arm knob diff against the B0c BASE "
               f"({_BASE_NAME}) -- the 'one knob per arm, two on R6' "
               "statement:")
    base = ARMS[_BASE_NAME]
    for name, spec in ARMS.items():
        knobs = []
        if spec.cadence_s != base.cadence_s:
            knobs.append("cadence")
        if (spec.nvz, spec.nvp) != (base.nvz, base.nvp):
            knobs.append("grid")
        out.append(
            f"  {name:<26} {len(knobs)} knob(s): "
            + (", ".join(knobs) or "(base)")
        )
    out.append("")
    out.append("[R7] GATED observables (all at t*, finer-arm denominator):")
    for tag, key, label, kind in GATED:
        out.append(f"  {tag}  {label:<34} [{kind}]  key={key}")
    out.append("      cumulative rows are read from the solver's own "
               "accumulators over [t_engage, t*] -- no post-hoc integration.")
    out.append("  REPORTED, NOT GATED: Te, floor-cell census, "
               "relax-limited count, terminal debts, R6 cross arm.")
    out.append("  NV2: a gated row zero-at-roundoff on the finer arm is "
               "INACTIVE-with-disclosure, never a trivial pass.")
    out.append("")
    out.append(f"[R8, AMENDED {R8_AMENDMENT_LABEL}] the DEFAULT fit is "
               f"--r8-fit {R8_FIT_AMENDED}: errors "
               f"e_k = ||u_k - u_ref|| against the converged reference rung "
               f"{R8_REFERENCE_ARM}, fitted as log(e_k) against "
               "log(h_k - h_ref) by least squares over the rungs with "
               f"NOMINAL cadence <= {R8_FIT_MAX_H:g} s, using each rung's "
               "EFFECTIVE h. The abscissa carries the FINITE-REFERENCE "
               "correction (R9's h_k/(h_k - h_ref)): the reference is a "
               "finite h, so a first-order response gives "
               "e_k ~ h_k - h_ref, not e_k ~ h_k. The uncorrected "
               "log-e/log-h slope is REPORTED beside it, with the value a "
               "first-order coupling would take there. Coarser "
               f"rungs are REPORTED but {R8_EXCLUDED_LABEL}: at "
               "h >= 2.5e-5 s the arm is in the ion-debit shortfall / "
               "Ti-collapse regime, not on the truncation-error curve. "
               "Confirmed ~1 per gated ACTIVE observable iff the errors are "
               f"monotone and the domain slope is in [{ORDER_BAND_COARSE[0]}, "
               f"{ORDER_BAND_COARSE[1]}] (the wider registered band: the "
               "domain fit carries the coarsest fit rung's O(h^2) "
               "contamination). Local per-rung slopes are reported beside it.")
    out.append(f"[R8, SUPERSEDED] --r8-fit {R8_FIT_PRE_AMENDMENT} re-forms "
               "the pre-amendment successive-pair fit over the whole banked "
               "ladder (e_i = ||u(h_i) - u(h_i+1)||, p = log2 ratios; never "
               "distance-to-finest), confirmed iff errors are monotone, the "
               f"FINE pair p is in [{ORDER_BAND_FINE[0]}, "
               f"{ORDER_BAND_FINE[1]}] and the COARSE pair p is in "
               f"[{ORDER_BAND_COARSE[0]}, {ORDER_BAND_COARSE[1]}]. It is kept "
               "only so the pre-amendment numbers stay reproducible.")
    out.append(f"[R8] Sampling-floor guard, both fits: the finest error "
               f"<= {SAMPLING_FLOOR_FACTOR:g} x (dt_sample x |dO/dt|) makes "
               "that slope UNDERDETERMINED -- it carries no order "
               "information; only a clear guard can FAIL out of band.")
    out.append(f"[R9] Ehat_k = ||u_k - u_finest|| x h_k/(h_k - h_finest); "
               f"cadence of record = COARSEST rung with Ehat_k < "
               f"{EHAT_TOL:.0%} on every gated ACTIVE observable AND passing "
               "R13. Shipped 2.5e-5 confirmed iff it is that rung or finer.")
    out.append(f"[R10] named grid = coarsest grid whose next refinement moves "
               f"every gated observable < {GRID_TOL:.0%} at the shipped "
               "cadence; no grid order is fitted; all rungs published.")
    out.append(f"[R11] I6 band per tick per arm: |ledger residual| (both "
               f"forms, particle and energy) <= {ROUNDOFF_REL:g} x "
               f"max(1, sqrt(nvz*nvp/{I6_REF_BINS})). Violation = FAIL "
               "regardless of convergence results.")
    for name, spec in ARMS.items():
        out.append(f"      {name:<26} band = {i6_band(spec.nvz, spec.nvp):g}")
    out.append(f"[R12] applied_cum + debt - booked_cum (M, Ei, ion) "
               f"<= {R12_TOL:g} rel at t*, per arm.")
    out.append("[R13] debt gates at the cadence-of-record arm at t*: "
               f"sum|Ei_debt|/sum|Ei_booked_cum| <= {DEBT_TOL['Ei']:g}, same "
               f"for M; ion <= {DEBT_TOL['ion']:g}; no persistent "
               "ion-shortfall warning. All arms report these.")
    out.append(f"[R14/NV3] burn-through non-vacuity on the shipped arm "
               f"({SHIPPED_ARM}): max nu_ion*dt_n >= {BURN_THROUGH_MIN:g} "
               f"in-window; a miss doubles t* ONCE (to "
               f"{T_STAR_MS_DOUBLED:g} ms) and re-runs; a second miss is a "
               "registered anomaly -> stop. Puff coverage reported.")
    out.append("")
    out.append("[R16] deliverable: b0c_convergence_table.md, assembled by "
               "--table from the banked per-arm npz files.")
    return out


# ----------------------------------------------------------------- table mode


class Verdict:
    def __init__(self, item, state, detail, consequence=None):
        self.item = item
        self.state = state          # PASS / FAIL / UNDERDETERMINED / REPORTED
        self.detail = detail
        self.consequence = consequence


def _pair_errors(arms, key, kind):
    """Successive-pair relative errors along a coarse->fine arm list [R8]."""
    errs = []
    for coarse, fine in zip(arms[:-1], arms[1:]):
        weights = fine["_obs"]["nn"] if kind == "wl2" else None
        errs.append(
            rel_error(coarse["_obs"][key], fine["_obs"][key], kind, weights)
        )
    return errs


def _dodt(arm, key, kind):
    """Relative rate of change of an observable on ``arm``, per second."""
    obs = arm["_obs"]
    prev = arm.get("_obsprev")
    if prev is None or arm.get("_t_prev_sample") is None:
        return None
    dt = float(arm["_t_sample"]) - float(arm["_t_prev_sample"])
    if dt <= 0.0:
        return None
    weights = obs["nn"] if kind == "wl2" else None
    return rel_error(prev[key], obs[key], kind, weights) / dt


def _activity(arms, key, kind):
    """NV2: is this row zero-at-roundoff on the FINEST arm?"""
    scales = [magnitude(a["_obs"][key], kind) for a in arms]
    ref = max(scales) if scales else 0.0
    fine = scales[-1] if scales else 0.0
    if ref == 0.0:
        return False, fine, ref
    return fine > ROUNDOFF_REL * ref, fine, ref


def _ref_errors(fit_arms, ref, key, kind):
    """[R8, amended] Reference-corrected errors e_k = ||u_k - u_ref||.

    ``ref`` is the converged reference rung. Every e_k on one row shares
    ITS denominator, unlike the successive-pair form where each pair
    carries its own -- which is what makes the log-e/log-h slope over the
    row a fit of one error curve rather than of a chain of differences.
    """
    weights = ref["_obs"]["nn"] if kind == "wl2" else None
    return [rel_error(a["_obs"][key], ref["_obs"][key], kind, weights)
            for a in fit_arms]


def _log_slope(hs, errs):
    """[R8, amended] Least-squares slope of log(e) against log(h).

    Returns nan unless EVERY point is finite and strictly positive: a zero
    or non-finite error carries no order information, and dropping it
    silently would fit a domain other than the registered one.
    """
    hs = [float(h) for h in hs]
    es = [float(e) for e in errs]
    if len(es) < 2 or len(hs) != len(es):
        return float("nan")
    if not all(np.isfinite(v) and v > 0.0 for v in es + hs):
        return float("nan")
    return float(np.polyfit(np.log(hs), np.log(es), 1)[0])


def _local_slopes(hs, errs):
    """[R8, amended] Adjacent-rung log-log slopes across the fit domain.

    Reported beside the domain fit so a slope that is drifting rung to rung
    is visible rather than averaged away by the least squares.
    """
    ps = []
    for i in range(len(errs) - 1):
        e0, e1 = float(errs[i]), float(errs[i + 1])
        h0, h1 = float(hs[i]), float(hs[i + 1])
        if (np.isfinite(e0) and np.isfinite(e1) and e0 > 0.0 and e1 > 0.0
                and h0 > 0.0 and h1 > 0.0 and h0 != h1):
            ps.append(math.log(e0 / e1) / math.log(h0 / h1))
        else:
            ps.append(float("nan"))
    return ps


def _log2_vs_h_max_rescale(ladder):
    """[R8] Largest rescaling the EFFECTIVE-h estimator would apply to p.

    The registered successive-pair order is ``p = log2(e_i / e_i+1)``, which
    reads the rungs as halving exactly. They do not: a neutral tick fires at
    the first accepted-step boundary at or past ``next_s``, so an arm's
    EFFECTIVE cadence can sit a little above its nominal. Forming the same
    pair against the effective h -- ``log(e_i / e_i+1) / log(h_i / h_i+1)``,
    which is what ``_local_slopes`` does on the amended path -- multiplies
    every p by ``log(2) / log(h_i / h_i+1)``. Returns the largest
    ``|factor - 1|`` over the ladder, or nan if no pair is usable.
    """
    devs = []
    for coarse, fine in zip(ladder[:-1], ladder[1:]):
        h0, h1 = float(coarse["_h"]), float(fine["_h"])
        if h0 > 0.0 and h1 > 0.0 and h0 != h1:
            devs.append(abs(math.log(2.0) / math.log(h0 / h1) - 1.0))
    return max(devs) if devs else float("nan")


def _r8_pre_amendment(lines, verdicts, arms, ladder, dead, registered):
    """[R8, PRE-AMENDMENT] Successive-pair order fit over the whole ladder.

    SUPERSEDED by ``_r8_amended`` and kept reachable only under
    ``--r8-fit pre-amendment``, exactly as ``--sampling tick-count`` keeps
    the superseded reading: the numbers already on the record stay
    reproducible. Its two identified defects are the ones the amendment
    addresses -- it fits across the coarse shortfall-regime rungs, and it
    measures against an unconverged finest rung.

    Returns the ACTIVE gated keys (the NV2 survivors), which R9 reads.
    """
    lines.append("## R8 order fit -- successive pairs only")
    lines.append("")
    lines.append(
        f"Re-formed from the common-t rows, {AMENDMENT_LABEL}: the pair "
        "errors below compare arms at ONE absolute time, so the sample-time "
        "mismatch that the floor guard exists to catch is zero by "
        "construction and the guard reads clear on every row."
        if registered else
        f"Formed from each arm's own N_k-th tick -- SUPERSEDED "
        f"{AMENDMENT_LABEL}."
    )
    lines.append("")
    if len(ladder) < 4:
        lines.append(
            f"Only {len(ladder)} usable cadence rung(s) banked "
            f"({', '.join(a['name'] for a in ladder) or 'none'}); the "
            "successive-pair fit needs at least 4 rungs to form two order "
            "pairs."
        )
        lines.append("")
        verdicts.append(Verdict(
            "R8 order ~ 1",
            "UNDERDETERMINED",
            f"{len(ladder)} usable rungs banked; run the missing rungs "
            + (f"(including the conditional {_CAD_COND_NAME}) "
               if dead else "")
            + "before the fit can be formed",
        ))
        verdicts.append(Verdict(
            "NV2 inactive rows", "UNDERDETERMINED",
            "the zero-at-roundoff test is made on the finest rung of the "
            "cadence ladder, which is not yet complete",
        ))
        active_keys = list(GATED_KEYS)
    else:
        lines.append(
            "Ladder (coarse -> fine): "
            + " -> ".join(f"`{a['name']}` (h = {a['_h']:.6g} s)"
                          for a in ladder)
        )
        lines.append("")
        rescale = _log2_vs_h_max_rescale(ladder)
        lines.append(
            "ESTIMATOR DISCLOSURE: `p` below is the registered `log2` "
            "ratio, which reads the rungs as halving exactly; the EFFECTIVE "
            "cadences do not, so the effective-h estimator "
            "`log(e_i/e_i+1)/log(h_i/h_i+1)` -- the amended path's form, and "
            "the one R8-bis registers -- would rescale every `p` on this "
            + (f"ladder by up to {rescale:.1%}"
               if np.isfinite(rescale) else "ladder by an unreadable amount")
            + ": reported, not applied."
        )
        lines.append("")
        lines.append(
            "| obs | row | active (NV2) | "
            + " | ".join(f"e{i+1}" for i in range(len(ladder) - 1))
            + " | "
            + " | ".join(f"p{i+1}{i+2}" for i in range(len(ladder) - 2))
            + " | monotone | floor guard | verdict |"
        )
        lines.append("|---" * (6 + (len(ladder) - 1) + (len(ladder) - 2))
                     + "|")
        active_keys = []
        per_obs_state = {}
        floor_tripped = []
        for tag, key, label, kind in GATED:
            active, fine_mag, ref_mag = _activity(ladder, key, kind)
            errs = _pair_errors(ladder, key, kind)
            ps = [
                math.log2(errs[i] / errs[i + 1])
                if (errs[i] > 0 and errs[i + 1] > 0
                    and np.isfinite(errs[i]) and np.isfinite(errs[i + 1]))
                else float("nan")
                for i in range(len(errs) - 1)
            ]
            monotone = all(
                np.isfinite(errs[i]) and np.isfinite(errs[i + 1])
                and errs[i] > errs[i + 1] for i in range(len(errs) - 1)
            )
            dodt = _dodt(ladder[-1], key, kind)
            dt_sample = abs(float(ladder[-1]["_t_sample"])
                            - float(ladder[-2]["_t_sample"]))
            floor = None
            guard = "n/a"
            if dodt is not None:
                floor = SAMPLING_FLOOR_FACTOR * dt_sample * dodt
                tripped = np.isfinite(errs[-1]) and errs[-1] <= floor
                guard = (f"{'TRIPPED' if tripped else 'clear'} "
                         f"(e_fine {errs[-1]:.3g} vs {floor:.3g})")
                if tripped and active:
                    floor_tripped.append(key)
            if not active:
                state = "INACTIVE"
            else:
                active_keys.append(key)
                p_fine = ps[-1] if ps else float("nan")
                p_coarse = ps[0] if ps else float("nan")
                in_fine = (np.isfinite(p_fine)
                           and ORDER_BAND_FINE[0] <= p_fine
                           <= ORDER_BAND_FINE[1])
                in_coarse = (np.isfinite(p_coarse)
                             and ORDER_BAND_COARSE[0] <= p_coarse
                             <= ORDER_BAND_COARSE[1])
                if key in floor_tripped:
                    state = "UNDERDETERMINED"
                elif monotone and in_fine and in_coarse:
                    state = "PASS"
                else:
                    state = "FAIL"
            per_obs_state[key] = state
            active_cell = (
                "yes" if active
                else f"NO (|u|_fine {fine_mag:.3g} vs ladder max "
                     f"{ref_mag:.3g})"
            )
            lines.append(
                f"| {tag} | {label} | {active_cell} | "
                + " | ".join(f"{e:.4g}" for e in errs) + " | "
                + " | ".join(f"{p:.4g}" for p in ps)
                + f" | {'yes' if monotone else 'NO'} | {guard} | {state} |"
            )
        lines.append("")
        if not active_keys:
            r8_state = "UNDERDETERMINED"
            detail = ("every gated row is zero-at-roundoff on the finest arm "
                      "(NV2): there is nothing to fit -- "
                      "inactive-with-disclosure, not a pass")
            verdicts.append(Verdict("R8 order ~ 1", r8_state, detail))
        elif floor_tripped:
            have_cond = (_CAD_COND_NAME in arms
                         and arms[_CAD_COND_NAME]["status"] == "ok")
            if not have_cond:
                r8_state = "UNDERDETERMINED"
                verdicts.append(Verdict(
                    "R8 order ~ 1", r8_state,
                    "sampling-floor guard TRIPPED on "
                    f"{', '.join(sorted(floor_tripped))}: the finest pair "
                    "error is at or below 10x the sample-time mismatch "
                    f"contribution. Registration: add the {_CAD_COND_NAME} "
                    "rung and re-form the pairs; only then can out-of-band "
                    "FAIL.",
                ))
            else:
                r8_state = "UNDERDETERMINED"
                verdicts.append(Verdict(
                    "R8 order ~ 1", r8_state,
                    "sampling-floor guard TRIPPED again on "
                    f"{', '.join(sorted(floor_tripped))} WITH the "
                    f"conditional rung {_CAD_COND_NAME} already banked and "
                    "the pairs re-formed. The registration provides for ONE "
                    "conditional rung, so this is past its branch: a "
                    "registered anomaly for adjudication, not a FAIL and not "
                    "a pass.",
                ))
        else:
            failed = [k for k in active_keys if per_obs_state[k] == "FAIL"]
            r8_state = "PASS" if not failed else "FAIL"
            verdicts.append(Verdict(
                "R8 order ~ 1", r8_state,
                "every gated ACTIVE observable is monotone with both pair "
                "orders in band" if not failed else
                "out of band or non-monotone on "
                + ", ".join(f"{GATED_TAG[k]} ({GATED_LABEL[k]})"
                            for k in failed),
                None if not failed else CONSEQUENCE["a"],
            ))
        inactive = [k for k in GATED_KEYS if k not in active_keys]
        if inactive:
            verdicts.append(Verdict(
                "NV2 inactive rows", "REPORTED",
                "zero-at-roundoff on the finest arm, so INACTIVE with "
                "disclosure (never a trivial pass): "
                + ", ".join(f"{GATED_TAG[k]} ({GATED_LABEL[k]})"
                            for k in inactive),
            ))
        else:
            verdicts.append(Verdict(
                "NV2 inactive rows", "PASS",
                "every gated row carries signal on the finest arm",
            ))

    return active_keys


def _r8_amended(lines, verdicts, arms, ladder, sampling):
    """[R8, AMENDED] Reference-corrected log-e/log-h order fit.

    Errors are formed against the converged reference rung
    ``R8_REFERENCE_ARM``, and the order is the least-squares slope of
    ``log(e_k)`` against ``log(h_k - h_ref)`` over the FIT DOMAIN -- the
    rungs whose NOMINAL cadence is at or below ``R8_FIT_MAX_H``.

    The abscissa carries the FINITE-REFERENCE CORRECTION, the same one R9
    applies as ``h_k/(h_k - h_ref)``. The reference is a finite h, not the
    exact solution, so a first-order response gives
    ``e_k = ||u_k - u_ref|| ~ h_k - h_ref`` and NOT ``~ h_k``. Fitting
    against ``h_k`` therefore reads a first-order coupling as steeper than
    first order, by an amount set only by how close the reference sits to
    the finest fit rung. That uncorrected slope is still REPORTED as a
    diagnostic column, beside the value it would take if the coupling were
    exactly first order, so the size of the bias is visible rather than
    inferred.

    Coarser rungs are reported with their own errors and carry
    ``R8_EXCLUDED_LABEL``: at h >= 2.5e-5 s the arm is in the ion-debit
    shortfall / Ti-collapse regime, which is not a sample of the smooth
    truncation-error curve the fit models.

    Returns the ACTIVE gated keys (the NV2 survivors), which R9 reads.
    """
    lines.append("## R8 order fit -- AMENDED: finite-reference corrected, "
                 f"fit domain nominal h <= {R8_FIT_MAX_H:g} s")
    lines.append("")
    lines.append(
        f"**Fit mode `{R8_FIT_AMENDED}` (the default), "
        f"{R8_AMENDMENT_LABEL}.** Errors are "
        f"`e_k = ||u_k - u_ref||` against the converged reference rung "
        f"`{R8_REFERENCE_ARM}`, read at the `{sampling}` sampling, and the "
        "gated order is the least-squares slope of `log e_k` against "
        "**`log(h_k - h_ref)`** over the fit domain, on each rung's "
        "EFFECTIVE h. That abscissa carries the FINITE-REFERENCE "
        "CORRECTION -- the same one R9 applies as `h_k/(h_k - h_ref)` -- "
        "because the reference is a finite h and not the exact solution, so "
        "a first-order response gives `e_k ~ h_k - h_ref` rather than "
        "`e_k ~ h_k`. This addresses two identified defects of the "
        "pre-amendment fit: the coarse rungs are a DIFFERENT REGIME "
        "(ion-debit shortfall / Ti collapse at h >= 2.5e-5 s), and the "
        "pre-amendment reference was UNCONVERGED (its finest rung "
        "over-books 5.2 %)."
    )
    lines.append("")

    ref = arms.get(R8_REFERENCE_ARM)
    if ref is not None and ref["status"] != "ok":
        ref = None
    fit_arms = [a for a in ladder
                if a["name"] != R8_REFERENCE_ARM
                and float(a["cadence_nominal_s"]) <= R8_FIT_MAX_H * (1.0 + 1e-9)]
    fit_names = {a["name"] for a in fit_arms}
    excluded_arms = [a for a in ladder
                     if a["name"] != R8_REFERENCE_ARM
                     and a["name"] not in fit_names]

    if ref is None or len(fit_arms) < 2:
        missing = []
        if ref is None:
            missing.append(f"the reference rung `{R8_REFERENCE_ARM}`")
        if len(fit_arms) < 2:
            missing.append(
                "at least 2 fit-domain rungs (banked: "
                + (", ".join(f"`{a['name']}`" for a in fit_arms) or "none")
                + ")"
            )
        lines.append("The amended fit cannot be formed: missing "
                     + " and ".join(missing) + ".")
        lines.append("")
        verdicts.append(Verdict(
            "R8 order ~ 1 (amended fit)", "UNDERDETERMINED",
            "the amended fit needs " + " and ".join(missing)
            + "; run the missing arm(s) before the fit can be formed",
        ))
        verdicts.append(Verdict(
            "NV2 inactive rows", "UNDERDETERMINED",
            "the zero-at-roundoff test is made on the amended fit's "
            "reference rung, which is not banked",
        ))
        return list(GATED_KEYS)

    hs = [float(a["_h"]) for a in fit_arms]
    h_ref = float(ref["_h"])
    lines.append(
        "Fit domain (coarse -> fine): "
        + " -> ".join(f"`{a['name']}` (h = {h:.6g} s)"
                      for a, h in zip(fit_arms, hs))
        + f"; reference `{ref['name']}` (h = {h_ref:.6g} s)."
    )
    lines.append("")
    if excluded_arms:
        lines.append(
            "EXCLUDED from the fit and reported below: "
            + ", ".join(f"`{a['name']}` (h = {float(a['_h']):.6g} s)"
                        for a in excluded_arms)
            + f" -- {R8_EXCLUDED_LABEL}."
        )
        lines.append("")

    inverted = [a for a, h in zip(fit_arms, hs) if h <= h_ref]
    if inverted:
        raise SystemExit(
            "REFUSED: the amended fit's abscissa is h_k - h_ref, so every "
            "fit-domain rung must be strictly COARSER than the reference "
            f"`{ref['name']}` (h = {h_ref:.6g} s), and "
            + ", ".join(f"`{a['name']}` (h = {float(a['_h']):.6g} s)"
                        for a in inverted)
            + " is not. The EFFECTIVE cadences have inverted the ladder, and "
            "a non-positive abscissa is not a fit domain."
        )
    hb = [h - h_ref for h in hs]
    p_ideal_unc = _log_slope(hs, hb)
    p_ideal_unc_local = _local_slopes(hs, hb)
    lines.append(
        "**The uncorrected slope is REPORTED, not gated.** Fitting the same "
        "errors against `log h_k` instead of `log(h_k - h_ref)` leaves the "
        "finite-reference bias in, and on THIS domain and this reference a "
        "perfectly first-order coupling would read "
        f"p_uncorrected = {p_ideal_unc:.4g} (local "
        + ", ".join(f"{p:.4g}" for p in p_ideal_unc_local)
        + ") rather than 1. That ideal is printed beside the uncorrected "
        "column below so the size of the bias is visible on the page; the "
        "corrected column is the one the registered band is read against."
    )
    lines.append("")

    lines.append(f"### Reference-corrected errors vs `{ref['name']}`")
    lines.append("")
    lines.append("| arm | h_k [s] | h_k - h_ref [s] (fit abscissa) | "
                 "in fit domain | "
                 + " | ".join(f"{GATED_TAG[k]} e_k" for k in GATED_KEYS)
                 + " |")
    lines.append("|---" * (4 + len(GATED_KEYS)) + "|")
    for a in [x for x in ladder if x["name"] != R8_REFERENCE_ARM]:
        errs = [_ref_errors([a], ref, k, GATED_KIND[k])[0] for k in GATED_KEYS]
        in_fit = a["name"] in fit_names
        h_a = float(a["_h"])
        lines.append(
            f"| `{a['name']}` | {h_a:.6g} | {h_a - h_ref:.6g} | "
            + ("yes" if in_fit else f"NO -- {R8_EXCLUDED_LABEL}") + " | "
            + " | ".join(f"{e:.4g}" for e in errs) + " |"
        )
    lines.append("")

    lines.append("### Order over the fit domain")
    lines.append("")
    lines.append(
        f"The band read here is the WIDER registered pair band "
        f"[{ORDER_BAND_COARSE[0]}, {ORDER_BAND_COARSE[1]}]: a domain fit "
        "carries the coarsest fit rung's O(h^2) contamination, which is what "
        f"that band exists for. Containment in the tighter "
        f"[{ORDER_BAND_FINE[0]}, {ORDER_BAND_FINE[1]}] is reported per row "
        "beside it and gates nothing. `fit p` and the local slopes are on "
        "the CORRECTED abscissa `h_k - h_ref`; `uncorrected p` is the same "
        "errors against `h_k` and is a diagnostic only."
    )
    lines.append("")
    n_local = len(hs) - 1
    lines.append(
        "| obs | row | active (NV2) | "
        + " | ".join(f"e({h:.4g})" for h in hs) + " | "
        + " | ".join(f"local p{i + 1}{i + 2}" for i in range(n_local))
        + " | fit p | in fine band | uncorrected p (finite-reference "
        f"biased; ideal-first-order p = {p_ideal_unc:.4g}) | monotone | "
        "floor guard | verdict |"
    )
    lines.append("|---" * (9 + len(hs) + n_local) + "|")

    active_keys = []
    per_obs_state = {}
    per_obs_p = {}
    floor_tripped = []
    dt_sample = abs(float(fit_arms[-1]["_t_sample"]) - float(ref["_t_sample"]))
    for tag, key, label, kind in GATED:
        active, fine_mag, ref_mag = _activity(ladder, key, kind)
        errs = _ref_errors(fit_arms, ref, key, kind)
        loc = _local_slopes(hb, errs)
        p = _log_slope(hb, errs)
        p_unc = _log_slope(hs, errs)
        per_obs_p[key] = p
        monotone = all(
            np.isfinite(errs[i]) and np.isfinite(errs[i + 1])
            and errs[i] > errs[i + 1] for i in range(len(errs) - 1)
        )
        dodt = _dodt(ref, key, kind)
        guard = "n/a"
        if dodt is not None:
            floor = SAMPLING_FLOOR_FACTOR * dt_sample * dodt
            tripped = np.isfinite(errs[-1]) and errs[-1] <= floor
            guard = (f"{'TRIPPED' if tripped else 'clear'} "
                     f"(e_fine {errs[-1]:.3g} vs {floor:.3g})")
            if tripped and active:
                floor_tripped.append(key)
        in_band = (np.isfinite(p)
                   and ORDER_BAND_COARSE[0] <= p <= ORDER_BAND_COARSE[1])
        in_fine = (np.isfinite(p)
                   and ORDER_BAND_FINE[0] <= p <= ORDER_BAND_FINE[1])
        if not active:
            state = "INACTIVE"
        else:
            active_keys.append(key)
            if key in floor_tripped:
                state = "UNDERDETERMINED"
            elif monotone and in_band:
                state = "PASS"
            else:
                state = "FAIL"
        per_obs_state[key] = state
        active_cell = (
            "yes" if active
            else f"NO (|u|_ref {fine_mag:.3g} vs ladder max {ref_mag:.3g})"
        )
        lines.append(
            f"| {tag} | {label} | {active_cell} | "
            + " | ".join(f"{e:.4g}" for e in errs) + " | "
            + " | ".join(f"{v:.4g}" for v in loc)
            + f" | {p:.4g} | {'yes' if in_fine else 'no'} | {p_unc:.4g} | "
            f"{'yes' if monotone else 'NO'} | {guard} | {state} |"
        )
    lines.append("")

    domain = ", ".join(f"`{a['name']}`" for a in fit_arms)
    stem = (
        f"amended fit ({R8_AMENDMENT_LABEL}): domain {domain} "
        f"(nominal h <= {R8_FIT_MAX_H:g} s), reference `{ref['name']}` "
        f"(h = {h_ref:.6g} s), finite-reference-corrected log-e against "
        f"log(h_k - h_ref) on effective h, band [{ORDER_BAND_COARSE[0]}, "
        f"{ORDER_BAND_COARSE[1]}]; rungs coarser than the domain are "
        f"{R8_EXCLUDED_LABEL}"
    )
    if not active_keys:
        verdicts.append(Verdict(
            "R8 order ~ 1 (amended fit)", "UNDERDETERMINED",
            stem + " -- every gated row is zero-at-roundoff on the reference "
            "rung (NV2), so there is nothing to fit: "
            "inactive-with-disclosure, not a pass",
        ))
    elif floor_tripped:
        verdicts.append(Verdict(
            "R8 order ~ 1 (amended fit)", "UNDERDETERMINED",
            stem + " -- sampling-floor guard TRIPPED on "
            f"{', '.join(sorted(floor_tripped))}: the finest fit-domain "
            f"error is at or below {SAMPLING_FLOOR_FACTOR:g}x the "
            "sample-time mismatch against the reference, so its slope "
            "carries no order information",
        ))
    else:
        failed = [k for k in active_keys if per_obs_state[k] == "FAIL"]
        verdicts.append(Verdict(
            "R8 order ~ 1 (amended fit)",
            "PASS" if not failed else "FAIL",
            stem + " -- "
            + ("every gated ACTIVE observable is monotone with its domain "
               "slope in band"
               if not failed else
               "out of band or non-monotone on "
               + ", ".join(f"{GATED_TAG[k]} ({GATED_LABEL[k]}, p = "
                           f"{per_obs_p[k]:.4g})" for k in failed)),
            None if not failed else CONSEQUENCE["a"],
        ))

    inactive = [k for k in GATED_KEYS if k not in active_keys]
    if inactive:
        verdicts.append(Verdict(
            "NV2 inactive rows", "REPORTED",
            "zero-at-roundoff on the reference rung, so INACTIVE with "
            "disclosure (never a trivial pass): "
            + ", ".join(f"{GATED_TAG[k]} ({GATED_LABEL[k]})"
                        for k in inactive),
        ))
    else:
        verdicts.append(Verdict(
            "NV2 inactive rows", "PASS",
            "every gated row carries signal on the reference rung",
        ))
    return active_keys


def evaluate(arm_records, out_path=None, sampling=SAMPLING_REGISTERED,
             r8_fit=R8_FIT_AMENDED):
    """Assemble the R16 table and evaluate R8-R14. Returns (lines, verdicts).

    ``sampling`` selects how each arm's observable rows are read: the
    registered common absolute t* (per-tick capture interpolated to
    ``t_engage + t*``) or the superseded per-arm N_k-th tick.

    ``r8_fit`` selects the R8 ESTIMATOR those rows are fed to: the amended
    reference-corrected fit over the rungs at or below ``R8_FIT_MAX_H``
    (the default), or the superseded successive-pair fit over the whole
    ladder. Both read the same banked arms -- the estimator, like the
    sampling, is a table-time choice and never a re-run.
    """
    if sampling not in SAMPLING_MODES:
        raise SystemExit(
            f"REFUSED: unknown sampling {sampling!r}; the registered mode is "
            f"{SAMPLING_REGISTERED!r} and {SAMPLING_SUPERSEDED!r} is the "
            "superseded one"
        )
    if r8_fit not in R8_FIT_MODES:
        raise SystemExit(
            f"REFUSED: unknown R8 fit mode {r8_fit!r}; the registered mode is "
            f"{R8_FIT_AMENDED!r} and {R8_FIT_PRE_AMENDMENT!r} is the "
            "superseded one"
        )
    lines = []
    verdicts = []

    arms = {}
    for meta, arrays in arm_records:
        name = meta["name"]
        if name not in ARMS:
            raise SystemExit(
                f"REFUSED: {meta['_path']} names arm {name!r}, which is not "
                "a registered B0c arm"
            )
        if meta.get("sanity"):
            raise SystemExit(
                "REFUSED: "
                f"{meta['_path']} is SANITY data (--updates override, "
                f"{meta['n_updates_done']} ticks against the registered "
                f"{ARMS[name].n_updates(meta['t_star_ms'])}). Sanity runs "
                "are deliberately un-bankable: they do not reach t* and "
                "cannot stand in for a registered arm. Re-run the arm "
                "without --updates."
            )
        schema = int(meta["capture_schema"])
        if schema < CAPTURE_SCHEMA:
            raise SystemExit(
                f"REFUSED: {meta['_path']} was banked at CAPTURE SCHEMA "
                f"{schema}; --table requires capture schema {CAPTURE_SCHEMA}. "
                f"Schema {schema} does not carry the rows the table reports "
                "per arm -- the per-cell hold debt and `nu_E * cadence` -- so "
                "reading it here would print `n/a` for a measurement that is "
                "simply ABSENT, which is how a missing number gets mistaken "
                "for a benign one. Re-run "
                f"`--arm {name}` on this harness to bank it at schema "
                f"{CAPTURE_SCHEMA}."
            )
        extras = meta.get("extras") or {}
        if extras:
            raise SystemExit(
                f"REFUSED: {meta['_path']} was run with the EXTRA KNOB(S) "
                + ", ".join(f"{k}={v!r}" for k, v in sorted(extras.items()))
                + f" on top of arm {name!r}'s registered knobs (the "
                "--arm-extra route). It is a control, not that registered "
                "arm: folding it into the ladder would put a differently "
                "configured run in the rung's place. Read it with the "
                "instrument that asked for it."
            )
        if name in arms:
            raise SystemExit(f"REFUSED: arm {name} banked twice")
        keys = [k for k in list(GATED_KEYS) + ["Te"]]
        if sampling == SAMPLING_REGISTERED:
            missing = [k for k in keys if f"tickobs_{k}" not in arrays]
            if missing:
                raise SystemExit(
                    f"REFUSED: {meta['_path']} carries no per-tick capture "
                    f"for {', '.join(missing)}, so it cannot be sampled at "
                    "the common absolute t* the amendment registers. It was "
                    "banked by a pre-amendment harness; re-run "
                    f"--arm {name}. (--sampling {SAMPLING_SUPERSEDED} reads "
                    "the superseded per-arm tick-count rows it does carry.)"
                )
            t_target = (float(meta["t_engage_s"])
                        + float(meta["t_star_ms"]) * 1.0e-3)
            index, weight = bracket_index(arrays["tick_time"], t_target)
            meta["_t_sample"] = t_target
            meta["_t_prev_sample"] = float(arrays["tick_time"][index - 1])
            meta["_sample_weight"] = weight
            meta["_obs"] = {
                k: (1.0 - weight) * arrays[f"tickobs_{k}"][index - 1]
                + weight * arrays[f"tickobs_{k}"][index]
                for k in keys
            }
            meta["_obsprev"] = {k: arrays[f"tickobs_{k}"][index - 1]
                                for k in keys}
        else:
            meta["_t_sample"] = float(meta["t_star_s"])
            meta["_t_prev_sample"] = (
                None if meta.get("t_prev_s") is None
                else float(meta["t_prev_s"])
            )
            meta["_sample_weight"] = None
            meta["_obs"] = {k: arrays[f"obs_{k}"] for k in keys
                            if f"obs_{k}" in arrays}
            if f"obsprev_{GATED_KEYS[0]}" in arrays:
                meta["_obsprev"] = {k: arrays[f"obsprev_{k}"] for k in keys
                                    if f"obsprev_{k}" in arrays}
        meta["_arrays"] = arrays
        meta["_nu_e_cadence"] = sampled_nu_e_cadence(meta, arrays)
        arms[name] = meta

    if not arms:
        raise SystemExit("REFUSED: no banked arms to assemble")

    t_star_set = {a["t_star_ms"] for a in arms.values()}
    if len(t_star_set) != 1:
        raise SystemExit(
            f"REFUSED: banked arms disagree on t* ({sorted(t_star_set)} ms); "
            "the R2 horizon must be common to the ladder"
        )
    t_star_ms = t_star_set.pop()

    # ---------------------------------------------------------- header [R16]
    registered = sampling == SAMPLING_REGISTERED
    lines.append("# B0c convergence table -- DVM cadence + velocity grid")
    lines.append("")
    lines.append(
        "Registration of record: the B0c registration, R1-R16, ratified "
        "2026-08-24. Harness: `scripts/verify_sim1d_b0c_cadence.py`."
    )
    lines.append("")
    if registered:
        lines.append(
            f"**Sampled at the COMMON ABSOLUTE t*, {AMENDMENT_LABEL}.** Every "
            "arm's observables are captured at every neutral tick and "
            "interpolated to the one absolute time `t_engage + t*`, so no "
            "pair error below carries a sample-time mismatch. R8 and R9 are "
            "re-formed from these rows."
        )
    else:
        lines.append(
            f"**Sampled at each arm's own N_k-th tick. This reading is "
            f"SUPERSEDED {AMENDMENT_LABEL}** -- it is reproduced here only "
            "so the pre-amendment numbers stay on the record. The registered "
            f"sampling is `--sampling {SAMPLING_REGISTERED}`."
        )
    lines.append("")
    lines.append("| header field | value |")
    lines.append("|---|---|")
    lines.append(f"| fixture [R1] | `verify_sim1d_k2_dvm.make_sim()` exactly; "
                 f"exchange = `{EXCHANGE_MODEL}`; no g1atrim overlay |")
    lines.append(f"| horizon [R2] | t* = t_engage + {t_star_ms:g} ms |")
    lines.append(f"| sampling [R2, amended] | `{sampling}` -- "
                 + ("REGISTERED " + AMENDMENT_LABEL if registered else
                    "SUPERSEDED " + AMENDMENT_LABEL) + " |")
    lines.append(
        f"| R8 fit mode [R8, amended] | `{r8_fit}` -- "
        + (f"REGISTERED {R8_AMENDMENT_LABEL}: finite-reference-corrected "
           f"fit of log e_k against log(h_k - h_ref) over the rungs with "
           f"nominal h <= {R8_FIT_MAX_H:g} s, against the reference rung "
           f"`{R8_REFERENCE_ARM}`; the uncorrected log-e/log-h slope is "
           f"reported as a diagnostic; coarser rungs reported but "
           f"{R8_EXCLUDED_LABEL}"
           if r8_fit == R8_FIT_AMENDED else
           f"SUPERSEDED {R8_AMENDMENT_LABEL}: successive-pair fit over the "
           "whole banked ladder, reproduced only so the pre-amendment "
           "numbers stay on the record")
        + " |"
    )
    lines.append(f"| mid-port cell [R7] | ES port {MID_PORT_NUMBER}, "
                 f"z_probe = {MID_PORT_Z_CM} cm -> cell {MID_PORT_CELL} "
                 f"(z = {MID_PORT_CELL_Z_CM} cm), from "
                 "`compare_sim1d_es1`'s port map |")
    lines.append(f"| suite roundoff | ROUNDOFF_REL = {ROUNDOFF_REL:g} |")
    schemas = sorted({int(a["capture_schema"]) for a in arms.values()})
    lines.append(
        f"| capture schema | banked arms carry {schemas}; this harness banks "
        f"{CAPTURE_SCHEMA}. Schema 2 adds the per-tick plasma `n` and `Ei`, "
        "the tick's frozen booked Ei RATE per cell, the per-cell hold debt "
        "and `nu_E * cadence`, so `nu_E` is reconstructible from a banked "
        "arm. An arm below schema 2 is REFUSED by `--table` rather than "
        "reported as `n/a`: the rows are absent, not benign |"
    )
    tn_fb = {bool(a["tn_feedback"]) for a in arms.values()}
    lines.append(
        f"| Tn feedback | none ({sorted(tn_fb)}) -- O6 (column Tn) is "
        "DIAGNOSTIC-ONLY: the `neutral_kinetic_dvm_tn_feedback` switch and "
        "its consumer were retired (Tom, 2026-08-31), so Tn is never fed back |"
    )
    lines.append("")

    # --------------------------------------------------------- arms table
    lines.append("## Arms")
    lines.append("")
    lines.append(
        "| arm | ladders | cadence nominal [s] | cadence effective [s] | "
        "dev | h_k used [s] | nvz | nvp | N_k done/req | steps | "
        "t_engage [s] | last tick [s] | sample t [s] | interp w | "
        "max_cell(nu_E*cadence) at t* | capture schema | status |"
    )
    lines.append("|---" * 17 + "|")
    for name in sorted(arms, key=lambda n: (-arms[n]["cadence_nominal_s"],
                                            arms[n]["nvz"])):
        a = arms[name]
        nom = float(a["cadence_nominal_s"])
        eff = float(a["cadence_effective_s"])
        dev = abs(eff - nom) / nom if nom else float("nan")
        a["_h"] = eff if dev > CADENCE_DEV_TOL else nom
        w = a["_sample_weight"]
        if w is None:
            w_cell = "n/a (tick-count)"
        elif 0.0 <= w <= 1.0:
            w_cell = f"{w:.4f} (bracketed)"
        else:
            w_cell = f"{w:.4f} **EXTRAPOLATED**"
        lines.append(
            f"| `{name}` | {'+'.join(a['ladders'])} | {nom:.6g} | "
            f"{eff:.6g} | {dev:.2%} | {a['_h']:.6g} | {a['nvz']} | "
            f"{a['nvp']} | {a['n_updates_done']}/{a['n_updates_requested']} | "
            f"{a['steps']} | {a['t_engage_s']:.10g} | {a['t_star_s']:.10g} | "
            f"{a['_t_sample']:.10g} | {w_cell} | {_nu_e_cell(a)} | "
            f"{a['capture_schema']} | {a['status']} |"
        )
    lines.append("")
    if registered:
        extrapolated = [n for n, a in sorted(arms.items())
                        if not (0.0 <= a["_sample_weight"] <= 1.0)]
        lines.append(
            "The interpolation weight is reported rather than assumed: a "
            "weight outside [0, 1] means the arm's tick series does not "
            "bracket t* and the row is an extrapolation along the nearest "
            "tick interval."
        )
        lines.append("")
        verdicts.append(Verdict(
            "R2 common-t bracketing", "PASS" if not extrapolated else
            "REPORTED",
            "every arm's tick series brackets the common t*, so every "
            "sampled row is an interpolation"
            if not extrapolated else
            "t* is NOT bracketed on " + ", ".join(extrapolated)
            + "; those rows are extrapolated along the nearest tick "
            "interval and are disclosed as such rather than clipped",
        ))

    # ------------------------------------------------------------- NV1 [R2]
    engages = {name: float(a["t_engage_s"]) for name, a in arms.items()}
    unique = sorted(set(engages.values()))
    if len(unique) == 1:
        verdicts.append(Verdict(
            "NV1 [R2] equal t_engage",
            "PASS",
            f"all {len(engages)} arms engage at t_engage = {unique[0]:.12g} s "
            "(bit-identical pre-engagement, as designed)",
        ))
    else:
        verdicts.append(Verdict(
            "NV1 [R2] equal t_engage",
            "FAIL",
            "arms disagree on t_engage: "
            + ", ".join(f"{n}={v:.12g}" for n, v in sorted(engages.items()))
            + " -- the arms are NOT bit-identical before engagement, so the "
            "ladder is not a cadence ladder",
            CONSEQUENCE["a"],
        ))

    # ------------------------------------------------------------ R11, R12
    lines.append("## R11 I6 independence band (per tick, per arm)")
    lines.append("")
    lines.append(
        "| arm | band | max particle distribution_rel | max particle "
        "domain_rel | max energy distribution_rel | max energy domain_rel | "
        "verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    r11_fail = []
    for name in sorted(arms):
        a = arms[name]
        band = float(a["i6_band"])
        vals = [
            float(a["i6_particle_distribution_max"]),
            float(a["i6_particle_domain_max"]),
            float(a["i6_energy_distribution_max"]),
            float(a["i6_energy_domain_max"]),
        ]
        ok = all(v <= band for v in vals)
        if not ok:
            r11_fail.append(name)
        lines.append(
            f"| `{name}` | {band:.3g} | " + " | ".join(f"{v:.3g}" for v in vals)
            + f" | {'PASS' if ok else 'FAIL'} |"
        )
    lines.append("")
    verdicts.append(Verdict(
        "R11 I6 independence band",
        "PASS" if not r11_fail else "FAIL",
        "every tick of every arm within its scaled band"
        if not r11_fail else
        f"band violated on {', '.join(r11_fail)} -- a ledger residual above "
        "the band is a FAIL regardless of the convergence results",
        None if not r11_fail else CONSEQUENCE["a"],
    ))

    lines.append("## R12 transfer identity at t* "
                 "(applied_cum + debt - booked_cum)")
    lines.append("")
    lines.append("| arm | M rel | Ei rel | ion (N) rel | verdict |")
    lines.append("|---|---|---|---|---|")
    r12_fail = []
    for name in sorted(arms):
        a = arms[name]
        vals = [float(a["identity_M_rel"]), float(a["identity_Ei_rel"]),
                float(a["identity_N_rel"])]
        ok = all(v <= R12_TOL for v in vals)
        if not ok:
            r12_fail.append(name)
        lines.append(
            f"| `{name}` | " + " | ".join(f"{v:.3g}" for v in vals)
            + f" | {'PASS' if ok else 'FAIL'} |"
        )
    lines.append("")
    verdicts.append(Verdict(
        "R12 transfer identity",
        "PASS" if not r12_fail else "FAIL",
        f"all channels <= {R12_TOL:g} rel" if not r12_fail else
        f"identity broken on {', '.join(r12_fail)}",
        None if not r12_fail else CONSEQUENCE["a"],
    ))

    # ---------------------------------------------------------------- R13
    lines.append("## R13 debt gates (all arms report; gated at the "
                 "cadence-of-record arm)")
    lines.append("")
    lines.append(
        "| arm | sum\\|Ei_debt\\|/sum\\|Ei_booked\\| | "
        "sum\\|M_debt\\|/sum\\|M_booked\\| | ion ratio | "
        "ion_shortfall_updates | shortfall warning | verdict |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    def r13_ok(a):
        return (
            float(a["debt_Ei_ratio"]) <= DEBT_TOL["Ei"]
            and float(a["debt_M_ratio"]) <= DEBT_TOL["M"]
            and float(a["debt_ion_ratio"]) <= DEBT_TOL["ion"]
            and not bool(a["ion_shortfall_warned"])
        )

    for name in sorted(arms):
        a = arms[name]
        lines.append(
            f"| `{name}` | {float(a['debt_Ei_ratio']):.3g} | "
            f"{float(a['debt_M_ratio']):.3g} | "
            f"{float(a['debt_ion_ratio']):.3g} | "
            f"{a['ion_shortfall_updates']} | "
            f"{'yes' if a['ion_shortfall_warned'] else 'no'} | "
            f"{'PASS' if r13_ok(a) else 'FAIL'} |"
        )
    lines.append("")

    # ------------------------------- hold debt + mid-port nu_E (reported)
    lines.append("## Hold debt and mid-port nu_E (reported, not gated)")
    lines.append("")
    lines.append(
        "The HOLD DEBT is the exponential transfer hold's own undelivered "
        "remainder, `Ei_hold_debt` / `M_hold_debt` in the schema-2 identity "
        "`applied_cum + debt + hold_debt == booked_cum`. It is a DIFFERENT "
        "quantity from the `Ei_debt` / `M_debt` gated by R13 above: R13's "
        "debt is what the ion channel could not pay, this is what the hold "
        "has not yet released. The ratios are read at the same sampled time "
        "as every other row; the max-over-ticks and residual columns are "
        "formed from the raw tick series, whose peak is generally not at t*. "
        "The residual is the tick-to-tick MOVEMENT of the hold rather than "
        "its level -- a large but static level is a lagging hold, a large "
        "residual is a hold still swinging at t*."
    )
    lines.append("")
    lines.append(
        "`nu_E * cadence` is carried here at the MID-PORT cell "
        f"({MID_PORT_CELL}), the companion to the max-over-cells value in "
        "the arms table above: the max says whether the hold is stable "
        "ANYWHERE, the mid-port value says whether it is stable at the cell "
        "the campaign scores."
    )
    lines.append("")
    lines.append(
        "| arm | h_k nominal [s] | sum\\|Ei_hold\\|/sum\\|Ei_booked\\| at t* "
        "| sum\\|M_hold\\|/sum\\|M_booked\\| at t* | max over ticks (Ei) | "
        "max per-tick resid \\|d(Ei_hold)\\|/\\|Ei_booked\\| | "
        f"nu_E*cadence mid-port (cell {MID_PORT_CELL}) at t* |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for name in sorted(arms):
        a = arms[name]
        hd = hold_debt_rows(a, a["_arrays"])
        lines.append(
            f"| `{name}` | {float(a['cadence_nominal_s']):.6g} | "
            f"{hd['Ei_ratio']:.6g} | {hd['M_ratio']:.6g} | "
            f"{hd['Ei_ratio_max_ticks']:.6g} | "
            f"{hd['residual_max_ticks']:.6g} | "
            f"{hd['nu_e_cadence_midport']:.6g} |"
        )
    lines.append("")

    # ------------------------------------------- reported-not-gated block
    lines.append("## Reported, not gated")
    lines.append("")
    lines.append(
        "| arm | Te rel vs next-finer | nn floor cells | nn_a floor cells | "
        "relax_steps | relax_limited_steps | limited cells | "
        "sum Ei_debt | sum M_debt | sum ion_debt | max nu_ion*dt_n | "
        "puff ticks active | puff births |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    ordered_all = sorted(arms, key=lambda n: -arms[n]["cadence_nominal_s"])
    for i, name in enumerate(ordered_all):
        a = arms[name]
        te = "n/a"
        if i + 1 < len(ordered_all):
            nxt = arms[ordered_all[i + 1]]
            if nxt["nvz"] == a["nvz"] and nxt["nvp"] == a["nvp"]:
                te = f"{rel_error(a['_obs']['Te'], nxt['_obs']['Te'], 'l2'):.3g}"
        arr = a["_arrays"]
        lines.append(
            f"| `{name}` | {te} | {a['nn_floor_cells']} | "
            f"{a['nn_a_floor_cells']} | {a['relax_steps']} | "
            f"{a['relax_limited_steps']} | {a['relax_limited_cells']} | "
            f"{float(np.sum(np.abs(arr['Ei_debt']))):.3g} | "
            f"{float(np.sum(np.abs(arr['M_debt']))):.3g} | "
            f"{float(np.sum(np.abs(arr['ion_debt']))):.3g} | "
            f"{float(a['burn_through_max']):.3g} | "
            f"{a['puff_ticks_active']} | "
            f"{float(a['puff_births_total']):.3g} |"
        )
    lines.append("")
    lines.append(
        "The floor-cell census carries a cross-arm EQUALITY expectation "
        "(assumptions ledger: floor-republish nonsmoothness "
        "PLAUSIBLE-UNVERIFIED); the relax-limited count sits next to every "
        "order number for the same reason (relax limiter inactive in-window "
        "PLAUSIBLE-UNVERIFIED)."
    )
    lines.append("")

    # ------------------------------------------------- R4 stability escape
    dead = [n for n in CADENCE_LADDER if n in arms
            and arms[n]["status"] == "dead_dt_min"]
    cap = [n for n in arms if arms[n]["status"] == "max_steps"]
    if cap:
        verdicts.append(Verdict(
            "R2 step cap",
            "FAIL",
            f"arms {', '.join(sorted(cap))} hit the {MAX_STEPS_PER_ARM}-step "
            "cap without reaching t*; a truncated arm is a loud FAIL, not a "
            "shorter window",
            CONSEQUENCE["a"],
        ))
    r4_note = ""
    if dead:
        r4_note = (
            f"R4 stability escape ENGAGED: {', '.join(dead)} died at dt_min "
            "(the frozen-drain path). That is a RESULT -- the stability "
            "boundary of the ladder -- and the conditional rung "
            f"{_CAD_COND_NAME} is required so two order pairs survive."
        )
        verdicts.append(Verdict(
            "R4 stability boundary", "REPORTED", r4_note,
        ))

    # ------------------------------------------------------- cadence ladder
    ladder_names = [n for n in CADENCE_LADDER
                    if n in arms and arms[n]["status"] == "ok"]
    for extra in (_CAD_COND_NAME, _CAD_REF_NAME):
        if extra in arms and arms[extra]["status"] == "ok":
            ladder_names.append(extra)
    ladder = [arms[n] for n in ladder_names]
    ladder.sort(key=lambda a: -float(a["_h"]))

    if r8_fit == R8_FIT_AMENDED:
        active_keys = _r8_amended(lines, verdicts, arms, ladder, sampling)
    else:
        active_keys = _r8_pre_amendment(lines, verdicts, arms, ladder, dead,
                                        registered)

    # ---------------------------------------------------------------- R9
    lines.append("## R9 corrected proxy true error "
                 "Ehat_k = ||u_k - u_finest|| x h_k/(h_k - h_finest)")
    lines.append("")
    lines.append(
        f"Re-formed from the common-t rows, {AMENDMENT_LABEL}."
        if registered else
        f"Formed from each arm's own N_k-th tick -- SUPERSEDED "
        f"{AMENDMENT_LABEL}."
    )
    lines.append("")
    lines.append(
        "The Ehat columns are FRACTIONS, not percent."
    )
    lines.append("")
    cadence_of_record = None
    if len(ladder) >= 2:
        finest = ladder[-1]
        h_f = float(finest["_h"])
        lines.append(
            f"Reference (finest banked rung): `{finest['name']}`, "
            f"h = {h_f:.6g} s. It serves R9 only; its own Ehat is undefined."
        )
        lines.append("")
        keys = active_keys or list(GATED_KEYS)
        lines.append("| arm | h_k [s] | correction h_k/(h_k - h_finest) | "
                     + " | ".join(f"{GATED_TAG[k]} Ehat" for k in keys)
                     + " | all < 1% | R13 | candidate |")
        lines.append("|---" * (6 + len(keys)) + "|")
        for a in ladder[:-1]:
            h_k = float(a["_h"])
            corr = h_k / (h_k - h_f)
            ehats = []
            for k in keys:
                kind = GATED_KIND[k]
                w = finest["_obs"]["nn"] if kind == "wl2" else None
                ehats.append(
                    rel_error(a["_obs"][k], finest["_obs"][k], kind, w) * corr
                )
            ok_e = all(np.isfinite(v) and v < EHAT_TOL for v in ehats)
            ok_d = r13_ok(a)
            a["_ehat_ok"] = ok_e
            a["_r13_ok"] = ok_d
            lines.append(
                f"| `{a['name']}` | {h_k:.6g} | {corr:.4g} | "
                + " | ".join(f"{v:.3g}" for v in ehats)
                + f" | {'yes' if ok_e else 'NO'} | "
                f"{'PASS' if ok_d else 'FAIL'} | "
                f"{'yes' if (ok_e and ok_d) else 'no'} |"
            )
        lines.append("")
        candidates = [a for a in ladder[:-1]
                      if a.get("_ehat_ok") and a.get("_r13_ok")]
        if candidates:
            cadence_of_record = max(candidates, key=lambda a: float(a["_h"]))
    if cadence_of_record is None:
        verdicts.append(Verdict(
            "R9 cadence of record", "UNDERDETERMINED"
            if len(ladder) < 2 else "FAIL",
            "no rung coarser than the reference meets Ehat < "
            f"{EHAT_TOL:.0%} on every gated active observable AND passes "
            "R13" if len(ladder) >= 2 else
            "fewer than two usable rungs banked",
            CONSEQUENCE["b"] if len(ladder) >= 2 else None,
        ))
        shipped_ok = None
    else:
        h_rec = float(cadence_of_record["_h"])
        shipped_ok = CADENCE_S <= h_rec
        verdicts.append(Verdict(
            "R9 cadence of record", "PASS",
            f"cadence of record = `{cadence_of_record['name']}` "
            f"(h = {h_rec:.6g} s) -- the coarsest rung under "
            f"Ehat < {EHAT_TOL:.0%} on every gated active observable that "
            "also passes R13",
        ))
        verdicts.append(Verdict(
            f"R9 shipped cadence ({CADENCE_S:g} s)",
            "PASS" if shipped_ok else "FAIL",
            f"shipped {CADENCE_S:g} s is "
            + ("that rung or finer -- CONFIRMED" if shipped_ok else
               f"COARSER than the cadence of record {h_rec:.6g} s -- NOT "
               "confirmed"),
            None if shipped_ok else CONSEQUENCE["b"],
        ))

    # --------------------------------------------------------------- R13
    if cadence_of_record is not None:
        rec_ok = r13_ok(cadence_of_record)
        verdicts.append(Verdict(
            "R13 debt gates at the cadence-of-record arm",
            "PASS" if rec_ok else "FAIL",
            f"arm `{cadence_of_record['name']}`: "
            f"Ei {float(cadence_of_record['debt_Ei_ratio']):.3g} "
            f"(<= {DEBT_TOL['Ei']:g}), "
            f"M {float(cadence_of_record['debt_M_ratio']):.3g} "
            f"(<= {DEBT_TOL['M']:g}), "
            f"ion {float(cadence_of_record['debt_ion_ratio']):.3g} "
            f"(<= {DEBT_TOL['ion']:g}), shortfall warning "
            f"{'RAISED' if cadence_of_record['ion_shortfall_warned'] else 'none'}",
            None if rec_ok else CONSEQUENCE["b"],
        ))
    else:
        verdicts.append(Verdict(
            "R13 debt gates at the cadence-of-record arm", "UNDERDETERMINED",
            "no cadence of record established, so its debt gates cannot be "
            "read; per-arm ratios are in the R13 table above",
        ))

    # --------------------------------------------------------------- R10
    lines.append("## R10 velocity-grid criterion (at the shipped cadence)")
    lines.append("")
    # Every banked, usable "grid"-kind arm is a rung -- the unconditional
    # GRID_LADDER and any conditional extension of it alike -- so a rung
    # added to the registration joins the pair chain without a second
    # special case here. Registration order for determinism, then sorted by
    # bin count so the chain is coarse -> fine whatever order they banked in.
    grid = [arms[n] for n in ARMS
            if n in arms and arms[n]["status"] == "ok"
            and "grid" in ARMS[n].ladders]
    grid.sort(key=lambda a: a["nvz"] * a["nvp"])
    named_grid = None
    if len(grid) < 2:
        lines.append(f"Only {len(grid)} usable grid rung(s) banked; the "
                     "criterion needs at least one refinement pair.")
        lines.append("")
        verdicts.append(Verdict(
            "R10 named velocity grid", "UNDERDETERMINED",
            f"{len(grid)} usable grid rung(s) banked",
        ))
    else:
        keys = list(GATED_KEYS)
        lines.append("| grid | next refinement | "
                     + " | ".join(f"{GATED_TAG[k]} rel" for k in keys)
                     + f" | all < {GRID_TOL:.0%} |")
        lines.append("|---" * (3 + len(keys)) + "|")
        for coarse, fine in zip(grid[:-1], grid[1:]):
            rels = []
            for k in keys:
                kind = GATED_KIND[k]
                w = fine["_obs"]["nn"] if kind == "wl2" else None
                rels.append(rel_error(coarse["_obs"][k], fine["_obs"][k],
                                      kind, w))
            ok = all(np.isfinite(v) and v < GRID_TOL for v in rels)
            if ok and named_grid is None:
                named_grid = coarse
            lines.append(
                f"| ({coarse['nvz']},{coarse['nvp']}) | "
                f"({fine['nvz']},{fine['nvp']}) | "
                + " | ".join(f"{v:.3g}" for v in rels)
                + f" | {'yes' if ok else 'NO'} |"
            )
        lines.append("")
        lines.append("No grid order is fitted -- the registration declares no "
                     "model for it -- and every rung is published above.")
        lines.append("")
        if named_grid is None:
            # Name the rung that is actually actionable, which depends on
            # how far down the conditional chain the banked set already is.
            if _GRID_COND_NAME in arms and _GRID_COND2_NAME in arms:
                next_rung = (
                    f"both conditional rungs ({_GRID_COND_NAME}, "
                    f"{_GRID_COND2_NAME}) are banked -- past the conditional "
                    "branch, adjudicate"
                )
            elif _GRID_COND2_NAME in arms:
                # Out-of-order banking: the chain's second conditional rung
                # without its first. Unreachable by the registered protocol,
                # but the text must not claim both are banked.
                next_rung = (
                    f"{_GRID_COND2_NAME} is banked but {_GRID_COND_NAME} is "
                    "not, so the conditional chain has a gap and no "
                    "refinement pair spans it -- run "
                    f"{_GRID_COND_NAME}"
                )
            elif _GRID_COND_NAME in arms:
                next_rung = (
                    f"{_GRID_COND_NAME} is already banked, so the actionable "
                    f"next rung is {_GRID_COND2_NAME} -- the R10 ladder "
                    "extension, Tom-ratified 2026-08-26 -- run it"
                )
            else:
                next_rung = (
                    "the registration provides the conditional "
                    f"{_GRID_COND_NAME} rung -- run it"
                )
            verdicts.append(Verdict(
                "R10 named velocity grid", "UNDERDETERMINED",
                "no banked grid rung has its next refinement inside "
                f"{GRID_TOL:.0%} on every gated observable; " + next_rung,
                CONSEQUENCE["c"],
            ))
        else:
            base_named = (named_grid["nvz"], named_grid["nvp"]) == (16, 6)
            verdicts.append(Verdict(
                "R10 named velocity grid", "PASS",
                f"named grid = ({named_grid['nvz']},{named_grid['nvp']}) -- "
                "the coarsest whose next refinement moves every gated "
                f"observable < {GRID_TOL:.0%}"
                + ("" if base_named else
                   "; (16,6) did NOT clear its own refinement"),
                None if base_named else CONSEQUENCE["c"],
            ))

    # ---------------------------------------------------------- R14 / NV3
    lines.append("## R14 / NV3 burn-through non-vacuity")
    lines.append("")
    if SHIPPED_ARM in arms:
        a = arms[SHIPPED_ARM]
        burn = float(a["burn_through_max"])
        ok = burn >= BURN_THROUGH_MIN
        lines.append(
            f"Shipped arm `{SHIPPED_ARM}`: max nu_ion*dt_n = {burn:.4g} "
            f"(bar {BURN_THROUGH_MIN:g}); puff active on "
            f"{a['puff_ticks_active']}/{a['n_updates_done']} ticks, "
            f"{float(a['puff_births_total']):.4g} particles born."
        )
        lines.append("")
        if a["puff_ticks_active"] == 0:
            lines.append(
                "Puff INACTIVE in-window: puff adequacy is therefore claimed "
                "by timescale separation only, and that is disclosed here "
                "rather than assumed."
            )
            lines.append("")
        if ok:
            verdicts.append(Verdict(
                "R14 / NV3 burn-through", "PASS",
                f"max nu_ion*dt_n = {burn:.4g} >= {BURN_THROUGH_MIN:g} "
                "in-window on the shipped arm -- the window is the marginal "
                "regime the registration asks for",
            ))
        elif t_star_ms == T_STAR_MS_DEFAULT:
            verdicts.append(Verdict(
                "R14 / NV3 burn-through", "UNDERDETERMINED",
                f"max nu_ion*dt_n = {burn:.4g} < {BURN_THROUGH_MIN:g} at "
                f"t* = {t_star_ms:g} ms. Registration: double t* ONCE to "
                f"{T_STAR_MS_DOUBLED:g} ms and re-run the ladder "
                f"(--t-star-ms {T_STAR_MS_DOUBLED:g}).",
            ))
        else:
            verdicts.append(Verdict(
                "R14 / NV3 burn-through", "FAIL",
                f"max nu_ion*dt_n = {burn:.4g} < {BURN_THROUGH_MIN:g} at the "
                f"ALREADY-DOUBLED t* = {t_star_ms:g} ms: a second miss is a "
                "registered anomaly -- stop.",
                CONSEQUENCE["a"],
            ))
    else:
        lines.append(f"Shipped arm `{SHIPPED_ARM}` not banked.")
        lines.append("")
        verdicts.append(Verdict(
            "R14 / NV3 burn-through", "UNDERDETERMINED",
            f"shipped arm `{SHIPPED_ARM}` not banked",
        ))

    # ------------------------------------------------------------ R6 cross
    lines.append("## R6 cross arm (reported, not gated)")
    lines.append("")
    if _CROSS_NAME in arms:
        c = arms[_CROSS_NAME]
        keys = list(GATED_KEYS)
        partners = [n for n in ("cad_6.25e-06", "grid_32x12") if n in arms]
        lines.append("| comparand | "
                     + " | ".join(f"{GATED_TAG[k]} rel" for k in keys) + " |")
        lines.append("|---" * (1 + len(keys)) + "|")
        for pname in partners:
            p = arms[pname]
            rels = []
            for k in keys:
                kind = GATED_KIND[k]
                w = c["_obs"]["nn"] if kind == "wl2" else None
                rels.append(rel_error(p["_obs"][k], c["_obs"][k], kind, w))
            lines.append(f"| `{pname}` vs `{_CROSS_NAME}` | "
                         + " | ".join(f"{v:.3g}" for v in rels) + " |")
    else:
        lines.append(f"`{_CROSS_NAME}` not banked.")
    lines.append("")

    # ------------------------------------------------------------- verdicts
    lines.append("## Verdicts")
    lines.append("")
    lines.append("| registered item | verdict | detail |")
    lines.append("|---|---|---|")
    for v in verdicts:
        lines.append(f"| {v.item} | **{v.state}** | {v.detail} |")
    lines.append("")
    consequences = [v for v in verdicts if v.consequence]
    if consequences:
        lines.append("### Registered consequences")
        lines.append("")
        for v in consequences:
            lines.append(f"- **{v.item}** -> {v.consequence}")
        lines.append("")
    elif all(v.state in ("PASS", "REPORTED") for v in verdicts):
        lines.append("### Registered consequence")
        lines.append("")
        lines.append(f"- {CONSEQUENCE['d']}")
        lines.append("")
    if r4_note:
        lines.append(f"> {r4_note}")
        lines.append("")

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(lines) + "\n")
    return lines, verdicts


# ------------------------------------------------------------------- driver


def extras_slug(extras):
    """A filesystem-safe tag naming an --arm-extra control's knobs."""
    parts = []
    for key, value in sorted(extras.items()):
        short = key.removeprefix("neutral_kinetic_dvm_")
        text = f"{short}-{value}"
        parts.append("".join(c if c.isalnum() or c in "-." else "_"
                             for c in text))
    return "_".join(parts)


def default_arm_path(spec, updates_override, out_dir):
    # An --arm-extra control is NOT the registered arm, so it is never
    # written under the ``b0c_arm_`` stem --table globs: a control must not
    # be swept into the ladder by a bare --table.
    stem = (f"b0c_arm_{spec.name}" if not spec.extras
            else f"b0c_extra_{spec.name}__{extras_slug(spec.extras)}")
    if updates_override is None:
        return Path(out_dir) / f"{stem}.npz"
    if spec.extras:
        return Path(out_dir) / f"{stem}_u{updates_override}.npz"
    return Path(out_dir) / f"b0c_sanity_{spec.name}_u{updates_override}.npz"


def main(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "B0c cadence + velocity-grid convergence harness "
            "(the B0c registration, R1-R16, ratified 2026-08-24)."
        )
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true",
                      help="print the full arm plan; solves nothing")
    mode.add_argument("--arm", metavar="NAME",
                      help="run exactly one registered arm and bank its npz")
    mode.add_argument("--table", action="store_true",
                      help="assemble the R16 table and verdicts from the "
                           "banked per-arm npz files")
    p.add_argument("--t-star-ms", type=float, default=T_STAR_MS_DEFAULT,
                   help=f"R2/R14 sample horizon in ms; "
                        f"{T_STAR_MS_DEFAULT:g} registered, "
                        f"{T_STAR_MS_DOUBLED:g} is the one R14 doubling")
    p.add_argument("--updates", type=int, default=None,
                   help="SANITY ONLY: stop after this many neutral ticks "
                        "instead of the registered N_k. The resulting npz is "
                        "marked sanity and --table REFUSES it.")
    p.add_argument("--arm-extra", metavar="KEY=VALUE", action="append",
                   default=None,
                   help="--arm: add ONE more config knob on top of the arm's "
                        "three registered ones, repeatable. The key must be "
                        "owned by input_dict or input_flags (checked here, "
                        "before anything solves) and must not be one of the "
                        "registered knobs. The result is a CONTROL, not the "
                        "registered rung: it banks under b0c_extra_* and "
                        "--table REFUSES it.")
    p.add_argument("--out", default=None,
                   help="output path (npz for --arm, markdown for --table)")
    p.add_argument("--out-dir", default=str(_SCRIPTS),
                   help="directory for the default artifact names")
    p.add_argument("--npz", nargs="*", default=None,
                   help="--table: explicit per-arm npz paths (default: "
                        "b0c_arm_*.npz under --out-dir)")
    p.add_argument("--quiet", action="store_true",
                   help="--arm: suppress per-tick progress")
    p.add_argument("--sampling", choices=SAMPLING_MODES,
                   default=SAMPLING_REGISTERED,
                   help=f"--table: how each arm's rows are read. "
                        f"{SAMPLING_REGISTERED!r} (default) is the REGISTERED "
                        f"sampling {AMENDMENT_LABEL} -- per-tick capture "
                        f"interpolated to the common absolute t*. "
                        f"{SAMPLING_SUPERSEDED!r} reads each arm at its own "
                        f"N_k-th tick and is SUPERSEDED {AMENDMENT_LABEL}; it "
                        "is kept only so the pre-amendment numbers stay "
                        "reproducible")
    p.add_argument("--r8-fit", choices=R8_FIT_MODES, default=R8_FIT_AMENDED,
                   help=f"--table: which R8 ESTIMATOR the sampled rows are "
                        f"fed to. {R8_FIT_AMENDED!r} (default) is the "
                        f"REGISTERED fit {R8_AMENDMENT_LABEL} -- "
                        f"errors against {R8_REFERENCE_ARM!r}, fitted "
                        "finite-reference-corrected as log(e_k) against "
                        "log(h_k - h_ref) over the rungs with nominal "
                        f"h <= {R8_FIT_MAX_H:g} s (uncorrected slope "
                        "reported as a diagnostic), coarser rungs reported "
                        f"but {R8_EXCLUDED_LABEL}. "
                        f"{R8_FIT_PRE_AMENDMENT!r} is the successive-pair fit "
                        "over the whole ladder and is SUPERSEDED; it is kept "
                        "only so the pre-amendment numbers stay reproducible")
    args = p.parse_args(argv)

    if args.t_star_ms not in T_STAR_MS_ALLOWED:
        p.error(
            f"--t-star-ms {args.t_star_ms} is not registered; R2 fixes t* at "
            f"{T_STAR_MS_DEFAULT:g} ms and R14 allows exactly one doubling "
            f"to {T_STAR_MS_DOUBLED:g} ms"
        )
    if args.updates is not None and not args.arm:
        p.error("--updates is a sanity override for --arm only")
    if args.arm_extra and not args.arm:
        p.error(
            "--arm-extra adds a knob to the arm --arm runs; it changes "
            "nothing about --plan or --table, so it must not be passed here "
            "as a silent no-op"
        )
    if args.sampling != SAMPLING_REGISTERED and not args.table:
        p.error(
            "--sampling selects how --table READS the banked arms; it "
            "changes nothing about what --arm runs (every arm banks its "
            "per-tick capture either way), so it must not be passed here "
            "as a silent no-op"
        )
    if args.r8_fit != R8_FIT_AMENDED and not args.table:
        p.error(
            "--r8-fit selects which R8 estimator --table forms from the "
            "banked arms; it changes nothing about what --arm runs, so it "
            "must not be passed here as a silent no-op"
        )

    if args.plan:
        for line in plan_lines(args.t_star_ms):
            print(line)
        return 0

    if args.arm:
        if args.arm not in ARMS:
            p.error(
                f"unknown arm {args.arm!r}; registered arms are "
                + ", ".join(ARMS)
            )
        spec = ARMS[args.arm]
        namespaces = {}
        if args.arm_extra:
            extras, namespaces = parse_arm_extras(args.arm_extra)
            spec = spec.with_extras(extras)
        n_registered = spec.n_updates(args.t_star_ms)
        n_updates = n_registered if args.updates is None else int(args.updates)
        sanity = args.updates is not None
        print(f"B0c arm {spec.name}  cadence {spec.cadence_s:g} s  "
              f"grid ({spec.nvz},{spec.nvp})  ladders "
              f"{'+'.join(spec.ladders)}")
        print(f"  registered N_k = {n_registered} ticks to "
              f"t* = t_engage + {args.t_star_ms:g} ms")
        if sanity:
            print(f"  *** SANITY OVERRIDE: running {n_updates} ticks. The "
                  "npz will be marked sanity and --table will REFUSE it. ***")
        if spec.extras:
            print("  *** EXTRA KNOB(S): this is a CONTROL, not the "
                  f"registered arm {spec.name}. --table will REFUSE the "
                  "npz. ***")
            for key, value in sorted(spec.extras.items()):
                print(f"        {key} = {value!r}  ({type(value).__name__}) "
                      f"-> {namespaces[key]}")
        diff = config_diff(dict(spec.knobs))
        print("  [R1] config diff vs the bare k2_dvm fixture:")
        for key, (a, b) in sorted(diff.items()):
            print(f"        {key}: {a!r} -> {b!r}")
        if not diff:
            print("        (none -- this arm IS the bare fixture)")
        inert = [k for k in spec.extras if k not in diff]
        if inert:
            p.error(
                "--arm-extra " + ", ".join(sorted(inert)) + " already "
                "hold(s) that value in the fixture, so the knob moves "
                "nothing and the control would be identical to the "
                "registered arm under a different name. A silent no-op "
                "control is worse than no control; pass a value that "
                "differs, or drop the option."
            )
        record, arrays = run_arm(spec, n_updates, verbose=not args.quiet)
        out = Path(args.out) if args.out else default_arm_path(
            spec, args.updates, args.out_dir
        )
        save_arm(out, record, arrays, sanity, args.t_star_ms)

        band = float(record["i6_band"])
        i6_vals = {
            "particle distribution_rel": record["i6_particle_distribution_max"],
            "particle domain_rel": record["i6_particle_domain_max"],
            "energy distribution_rel": record["i6_energy_distribution_max"],
            "energy domain_rel": record["i6_energy_domain_max"],
        }
        i6_ok = all(v <= band for v in i6_vals.values())
        ident = {k: record[f"identity_{k}_rel"] for k in ("M", "Ei", "N")}
        r12_ok = all(v <= R12_TOL for v in ident.values())
        debt = {k: record[f"debt_{k}_ratio"] for k in ("Ei", "M", "ion")}
        r13_pass = (
            debt["Ei"] <= DEBT_TOL["Ei"] and debt["M"] <= DEBT_TOL["M"]
            and debt["ion"] <= DEBT_TOL["ion"]
            and not record["ion_shortfall_warned"]
        )
        eff = float(record["cadence_effective_s"])
        dev = abs(eff - spec.cadence_s) / spec.cadence_s

        print("")
        print(f"  status            {record['status']}"
              + (f"  ({record['dead_reason']})"
                 if record["dead_reason"] else ""))
        print(f"  t_engage          {record['t_engage_s']:.12g} s")
        print(f"  t*                {record['t_star_s']:.12g} s "
              f"({record['n_updates_done']} ticks, {record['steps']} steps)")
        print(f"  cadence effective {eff:.6g} s "
              f"(nominal {spec.cadence_s:g}, dev {dev:.2%}"
              + ("; EFFECTIVE value is the h_k R8/R9 will use)"
                 if dev > CADENCE_DEV_TOL else ")"))
        print(f"  [R11] band {band:.3g}: "
              + ", ".join(f"{k} {v:.3g}" for k, v in i6_vals.items())
              + f"  -> {'PASS' if i6_ok else 'FAIL'}")
        print(f"  [R12] identity rel: "
              + ", ".join(f"{k} {v:.3g}" for k, v in ident.items())
              + f" (tol {R12_TOL:g})  -> {'PASS' if r12_ok else 'FAIL'}")
        print(f"  [R13] debt ratios: "
              + ", ".join(f"{k} {v:.3g}" for k, v in debt.items())
              + f", shortfall updates {record['ion_shortfall_updates']}, "
              f"warning {'RAISED' if record['ion_shortfall_warned'] else 'none'}"
              f"  -> {'PASS' if r13_pass else 'FAIL'}")
        print(f"  [R14] max nu_ion*dt_n {float(record['burn_through_max']):.4g} "
              f"(bar {BURN_THROUGH_MIN:g}); puff active on "
              f"{record['puff_ticks_active']}/{record['n_updates_done']} ticks")
        print(f"  floor cells nn {record['nn_floor_cells']}, "
              f"nn_a {record['nn_a_floor_cells']} "
              f"(floor {record['nn_floor']:.6g}); relax_steps "
              f"{record['relax_steps']}, limited steps "
              f"{record['relax_limited_steps']}, limited cells "
              f"{record['relax_limited_cells']}")
        if record["warnings"]:
            print(f"  warnings raised during the arm "
                  f"({len(record['warnings'])} distinct):")
            for text in record["warnings"]:
                print(f"        {text.splitlines()[0][:160]}")
        print(f"  wrote {out}")
        if record["status"] == "max_steps":
            print("  FAIL: the arm hit the registered step cap without "
                  "reaching t* [R2].")
            return 1
        if record["status"] == "dead_dt_min":
            print("  R4: the arm died at dt_min. That is a RESULT (the "
                  "stability boundary), not a harness failure; the "
                  f"conditional rung {_CAD_COND_NAME} is now required.")
        if not (i6_ok and r12_ok):
            print("  FAIL: R11 and/or R12 violated -- a FAIL regardless of "
                  "the convergence results [R11].")
            return 1
        return 0

    # --table
    if args.npz:
        paths = [Path(x) for x in args.npz]
    else:
        paths = sorted(Path(args.out_dir).glob("b0c_arm_*.npz"))
    if not paths:
        raise SystemExit(
            f"REFUSED: no per-arm npz found (looked for b0c_arm_*.npz under "
            f"{args.out_dir}). Run --arm for each registered arm first."
        )
    records = [load_arm(path) for path in paths]
    out = Path(args.out) if args.out else (
        Path(args.out_dir) / "b0c_convergence_table.md"
    )
    lines, verdicts = evaluate(records, out_path=out, sampling=args.sampling,
                               r8_fit=args.r8_fit)
    for line in lines:
        print(line)
    print(f"\nwrote {out}")
    states = [v.state for v in verdicts]
    if "FAIL" in states:
        return 1
    if "UNDERDETERMINED" in states:
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
