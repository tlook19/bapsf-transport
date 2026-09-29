"""Assertion smoke suite for LAPDSim1D: the harness of the ``smoke`` package.

Run the whole suite (the gate) with no arguments; it exits 0 on success and
dies at the first failing assert, exactly as the single linear script it
replaces did.

    python scripts/gates/smoke_sim1d.py              # full suite, the gate
    python scripts/gates/smoke_sim1d.py --list       # case names, in order
    python scripts/gates/smoke_sim1d.py --only cathode-boundary-beam-terms
    python scripts/gates/smoke_sim1d.py --trace      # log each case name as it starts

This module holds the case registry, the ``@_case`` decorator, the runner and
command line, the historical-stance pins, the fixtures, and the helpers that
cases in more than one module share. The cases themselves live in one module
per subsystem beside it; ``smoke/__init__.py`` imports them and puts the
registry in run order.

There is no pytest dependency and no discovery: ``_CASES`` is an ordered list
built by the ``@_case`` decorator at import time. Registration is per module,
so the package re-sorts ``_CASES`` into the order ``_CASE_ORDER`` (in
``smoke/__init__.py``) states, and the full suite runs it in that order,
which is the order the blocks had inside the old ``main()``.

SHARED STATE, AND WHAT ``--only`` CAN AND CANNOT DO
---------------------------------------------------
The old ``main()`` was one scope, so every block could see everything the
blocks above it had built. Two mechanisms replace that, and both are explicit:

* **Fixtures** -- ``_base_config``, ``_base_sim``, ``_resolved_config``,
  ``_resolved_geometry``, ``_cathode_flags``, ``_resolved_cathode_flags``.
  These are the objects ``main()`` built ONCE and dozens of blocks reused.
  Each is a plain function whose value is cached for the process, so in a
  full-suite run every case gets the same object -- mutations included --
  that the single shared scope used to hand out. A case that asks for one
  binds it in the first lines of its body.

* **The case context** -- values that one case computes and a later case
  consumes travel through a ``ctx`` dict the runner threads. A case declares
  what it hands on in ``@_case(..., provides=(...))`` and what it consumes as
  its parameters; the runner injects the parameters and harvests the declared
  names from the case's ``return locals()``. The registry is therefore the
  dependency graph, and ``--only`` refuses a case whose inputs are missing
  with a message naming the case that produces them.

Consequences of running a subset, stated rather than hidden:

* A case that mutates a fixture (most of them mutate ``params``/``flags``)
  changes what every LATER case sees. Under ``--only`` a case gets the fixture
  as built, without its predecessors' mutations, so a case can pass in the
  full suite and fail alone, or the reverse. ``--only`` is a debugging aid;
  the FULL SUITE is the gate.
* Nothing is reordered and nothing is duplicated: the cases hold the original
  statements, in the original order. The one deliberate exception is ONE extra
  ``default_config()`` call at the top. The shipped-defaults case still calls
  it for its read-only assertions on the SHIPPED values, which have to be read
  before anything is pinned; ``_base_config`` then calls it again to build the
  pair it pins the historical stance onto. That is a dict build, and no
  assertion, construction or run happens twice.

DEPRECATION-WARNING SUPPRESSION (and its limits)
------------------------------------------------
Two helpers pin a deliberately legacy stance: ``_pin_pre_r2a_neutral_stance``
(the pre-R2a 6-field cold-neutral layout) and ``_pin_operator_algebra_stance``
(the historical all-cells operator algebra). Every key they touch is listed in
``_HISTORICAL_PIN_KEYS``. Cases registered with ``historical_stance=True`` run
inside ``_historical_pin_warnings()``, which ignores DeprecationWarnings whose
message names one of those keys -- so the suite's stderr is not buried under
warnings about pins it makes on purpose.

The suppression is deliberately narrow in both directions:

* only those keys are muted, so a case on the historical stance that
  deprecates any OTHER key still prints its warning;
* only cases that BUILD ON a pinned config dict are muted -- either straight
  from a fixture, or from one a predecessor derived and handed on -- so a case
  that constructs a deprecated configuration of its own still prints its
  warning (how the registry splits is ``_CASE_CENSUS``, asserted against
  the live ``_CASES`` at import so no count can go stale here);
* ``production-construction-warning-free`` is not muted, and it asserts
  against a fresh ``warnings.catch_warnings(record=True)`` with
  ``simplefilter("always")``, which overrides any outer filter -- a production
  default that starts warning is still caught there.
"""

import argparse
import ast
import contextlib
import inspect
from pathlib import Path
import sys
import warnings

import numpy as np

from cablp.constants import m_He_cgs
from cablp.solvers._sim1d import LAPDSim1D, default_config
from cablp.solvers._sim1d.core.geometry import anode_flanking_cells
from cablp.solvers._sim1d.core.state import ConservativeState1D


# scripts/ sibling imports: the seven purpose subdirectories on sys.path.
# Installed at MODULE scope, once, so that every case importing a scripts/
# sibling can do so whether it runs in the full suite or alone under --only.
# Per-case copies of this block only reach the cases that run after them, so
# a case whose own body carried none raised ModuleNotFoundError under --only
# while passing in the suite.
for _sub in ("atomic", "gates", "kinetic", "run", "score", "stance",
             "verify"):
    _dir = str(Path(__file__).resolve().parents[2] / _sub)
    if _dir not in sys.path:
        sys.path.insert(0, _dir)


def _cov_blank_rhs_term(cells, n_row):
    """Return a fresh 5-row RHS bundle carrying ``n_row`` and zeros elsewhere.

    ``LAPDSim1D._zero_rhs_state()`` hands back the solver's ONE shared
    all-zero bundle with read-only rows, so a case that needs a term with
    content of its own builds that term itself.
    """
    return ConservativeState1D(
        n=np.asarray(n_row, dtype=float),
        nn=np.zeros(cells, dtype=float),
        M=np.zeros(cells, dtype=float),
        Ee=np.zeros(cells, dtype=float),
        Ei=np.zeros(cells, dtype=float),
    )


# R2a fold-in (2026-08-20): the neutral closure family and the cathode neutral
# jet became config DEFAULTS. Many blocks below were written against the older
# cold-neutral stance -- they hand-pack cold state vectors or check a refusal
# that a second default-on conflict would now pre-empt. This scoped helper puts
# one (params, flags) pair back on that stance so those blocks run as written;
# the closure family and the jet keep their own blocks, which build their own
# configs and exercise the shipped defaults. The two-zone split is
# unconditional, so the cold layout is the 6-field (n, nn, M, Ee, Ei, nn_a).
#
# The three jet keys travel with neutral_momentum: the jet is M_n momentum
# physics and requires the flag, the surface debit requires the jet, and the
# total_reflected convention requires the jet too.
def _pin_pre_r2a_neutral_stance(params, flags):
    """Pin the pre-R2a 6-field cold-neutral stance in place; return the pair."""
    flags["neutral_momentum"] = False
    flags["neutral_energy"] = False
    flags["neutral_hot_internal_wall"] = False
    params["cathode_neutral_jet"] = False
    params["cathode_jet_surface_debit"] = False
    params["cathode_jet_energy_convention"] = "legacy"
    return params, flags


# The long-standing operator algebra isolates a simple fluid stance;
# dedicated R1/R2/R3/R4 cases exercise the live defaults. This helper
# holds the pins that used to sit inline at the top of main(): the cathode
# and implicit heat substep off, the scheduled phase machine, and the pre-R2a
# cold-neutral layout the hand-packed (n, nn, M, Ee, Ei, nn_a) state vectors
# below assume. The puff is the shipped square valve pulse on the orifice row.
def _pin_operator_algebra_stance(params, flags):
    """Pin the historical operator-algebra stance in place; return the pair."""
    params["phase_transition_mode"] = "scheduled"
    flags["cathode_coupling"] = False
    flags["implicit_heat_conduction"] = False
    return _pin_pre_r2a_neutral_stance(params, flags)


# The two helpers above are the ONLY places the smoke pins a deliberately
# legacy stance, and every key they touch is listed here. The runner mutes the
# DeprecationWarnings these keys raise -- and only these keys, and only for the
# cases that declare the historical stance (see ``_historical_pin_warnings``
# and the module docstring). A case that deprecates any OTHER key, or that
# constructs a deprecated config without asking for the historical fixtures,
# still prints its warning.
_HISTORICAL_PIN_KEYS = (
    # _pin_pre_r2a_neutral_stance
    "neutral_momentum",
    "neutral_energy",
    "neutral_hot_internal_wall",
    "cathode_neutral_jet",
    "cathode_jet_surface_debit",
    "cathode_jet_energy_convention",
    # _pin_operator_algebra_stance
    "phase_transition_mode",
    "cathode_coupling",
    "implicit_heat_conduction",
)


# ----------------------------------------------------------------------
# Fixtures. main() built each of these ONCE and dozens of blocks reused the
# result, mutations included. A fixture is a plain function whose value is
# cached, so every case that asks gets the SAME object the single shared
# scope used to hand out -- and a case run alone under --only builds it on
# demand instead of inheriting nothing.
# ----------------------------------------------------------------------
_FIXTURE_CACHE = {}


def _fixture(fn):
    """Cache a fixture's value for the lifetime of the process."""
    def wrapper():
        if fn.__name__ not in _FIXTURE_CACHE:
            _FIXTURE_CACHE[fn.__name__] = fn()
        return _FIXTURE_CACHE[fn.__name__]
    wrapper.__name__ = fn.__name__
    wrapper.__doc__ = fn.__doc__
    wrapper.__wrapped__ = fn
    return wrapper


@_fixture
def _base_config():
    """(params, flags) on the historical operator-algebra stance."""
    params, flags = default_config()
    return _pin_operator_algebra_stance(params, flags)


@_fixture
def _base_sim():
    """(sim, snapshot) for the base stance; snapshot carries geom/state/derived."""
    params, flags = _base_config()
    sim = LAPDSim1D(params, flags)
    snapshot = sim.get_initial_snapshot()
    return sim, snapshot


@_fixture
def _resolved_config():
    """(resolved_params, resolved_flags): resolved typed-segment geometry."""
    resolved_params, resolved_flags = default_config()
    # This fixture and everything derived from it (twin_*, m5_*, rgap_*, ...)
    # exercises geometry, the cathode solve and the beam on the 5-field
    # cold-neutral layout, hand-packing (n, nn, M, Ee, Ei) state vectors, and
    # several of the derived cases arm the jet themselves to check its
    # refusals.
    return _pin_pre_r2a_neutral_stance(resolved_params, resolved_flags)


@_fixture
def _resolved_geometry():
    """The resolved typed-segment geometry."""
    resolved_params, resolved_flags = _resolved_config()
    return LAPDSim1D(
        resolved_params, resolved_flags
    ).get_initial_snapshot().geometry


@_fixture
def _cathode_flags():
    """The base flags with cathode_coupling armed."""
    _, flags = _base_config()
    cathode_flags = dict(flags)
    cathode_flags["cathode_coupling"] = True
    return cathode_flags


@_fixture
def _resolved_cathode_flags():
    """The resolved flags with cathode_coupling armed, prebreakdown off."""
    _, resolved_flags = _resolved_config()
    resolved_cathode_flags = dict(resolved_flags)
    resolved_cathode_flags["cathode_coupling"] = True
    return resolved_cathode_flags


# The production defaults carry the full cathode/beam stack (csda +
# quasilinear, the surface power balance, the ads/des surface state, presheath
# sample smoothing) and the R2/R3 fluid repairs. The cathode-MECHANISM unit
# tests below isolate a single mechanism against a simple stance; this scoped
# helper returns it. The surface power balance and the ads/des coverage are
# unconditional, so the stance HOLDS them fixed instead: an emitting-layer
# heat capacity large enough that no step's temperature increment survives
# the addition (T_s stays at cathode_Ts_base_K to the bit), and a zero
# cleaning cross section with a clean floor chosen so the fully covered
# effective work function is phi_wf exactly. The ion-neutral drag is
# irrelevant to cathode emission/coverage, so the INERT-on-production params
# stay inert. Dedicated production cathode tests do NOT use this and exercise
# the real defaults.
def _tracking_electrode_sample(sim):
    """Make ``sim``'s electrode sample follow its accepted state.

    The sample EMA is re-seeded from the accepted state after every accepted
    step instead of relaxing toward it at the presheath transit time, so the
    sheath solve reads the state the step started from. For probes whose
    window is shorter than that transit time and that need the cathode to
    respond to the plasma inside it. Returns ``sim``.
    """
    sim._update_sample_smoothing = lambda dt: sim._init_sample_smoothing()
    return sim


def _cathode_unit_config():
    """Return (params, flags) on the simple cathode/fluid stance for the
    cathode-mechanism unit tests (moment closure stays on)."""
    p, f = default_config()
    p.update({
        "beam_anomalous_model": "none",
        # The surface temperature these unit tests run at, held there by the
        # heat capacity below.
        "cathode_Ts_base_K": 1998.15,
        "cathode_heat_capacity_J_per_K": 1.0e30,
        "cathode_conduction_W_per_K": 0.0,
        "cathode_phiwf_clean_eV": 2.5,
        "cathode_cleaning_sigma_cm2": 0.0,
        "cathode_cleaning_E_th_eV": None,
        "phi_wf": 3.0,
        # simple 1st-order integration so run-based tests match the analytic
        # backward-Euler forms they check
        "operator_splitting": "lie",
        "implicit_heat_scheme": "backward_euler",
        "heat_picard_iterations": 0,
    })
    f.update({
    })
    # These unit tests are about the cathode and the fluid; keep the simple
    # cold-neutral stance the helper's name promises.
    return _pin_pre_r2a_neutral_stance(p, f)


# The sheath state that escaped the phi_c ceiling before the returned-root fix
# (2026-08-09), frozen from the solve it was captured at: step 39 of the f = 1.0
# tail-walk arm, the solve whose beam energy the EII table-edge guard then
# refused (scripts/capfix_escape_case.txt). At this state, and at THIS imposed
# current, the bracket ladder's first grid point sat below the cap in NET phi_c
# (psi_minus > 0), the ladder doubled once, and the J-root came back FAR above
# the 1000 V cap, tagged virtual_cathode. It is kept as literals
# rather than re-derived: reproducing it needs a 39-step production march, and
# post-fix that march no longer visits this state at all.
_CAPFIX_ESCAPE_CONFIG = dict(
    A_c=706.8583470577034,
    mu=4,
    ion_mass_g=m_He_cgs,
    T_s=1910.0000073162657,
    phi_wf=2.8689998037499964,
    C_R=12.96,
    R_comp=0.0072244,
    R_comp_partition=1.0,
    R_mesh_ohm=0.0,
    eta=0.358,
    L_cath=50.0,
    R_cath=15.0,
    emission_Ts_K=tuple(
        np.float64(v)
        for v in (
            1909.7820600303419, 1908.0402707241847, 1904.566206201057,
            1899.3787650723516, 1892.5059746195277, 1883.9846150874373,
            1873.859733444452, 1862.1840584532556, 1849.0173309615827,
            1834.425564867055,
        )
    ),
    emission_area_cm2=tuple(
        np.float64(v)
        for v in (
            7.0685834705770345, 21.205750411731103, 35.34291735288517,
            49.480084294039244, 63.61725123519331, 77.75441817634739,
            91.89158511750145, 106.02875205865551, 120.16591899980959,
            134.30308594096365,
        )
    ),
    emission_plasma_frac=(1.0,) * 10,
)
_CAPFIX_ESCAPE_PLASMA = dict(
    T_e=2.895378507817385,
    n_e=824479922.2030256,
    n_n=19744589075162.41,
    sigma_b=0.0,
)
_CAPFIX_ESCAPE_KWARGS = dict(
    anode_current_A=0.0424709088850732,
    anode_T_e=2.9855005007754283,
    schottky=True,
    phi_c_cap_V=1000.0,
    alpha_sheath=0.8738291673131621,
    alpha_sheath_anode=None,
)
# The imposed current at capture, and the currents that bracket the escape
# window measured there: below ~5.4698 A the sheath carries the current under
# the ceiling; above it, every current must come back capability_limited AT the
# ceiling. The threshold is a property of this frozen state and of the ion
# current the sheath balance carries, so it moves with the sound speed and the
# sheath lift; what is gated is the ceiling binding the RETURNED ROOT on both
# sides of it, never the threshold's own value.
_CAPFIX_ESCAPE_I_A = 5.5674329614887945


# ----------------------------------------------------------------------
# The case registry. ``_CASES`` is ordered by registration, and registration
# order IS the order the blocks had inside the old linear main().
# ----------------------------------------------------------------------
class _Case:
    """One named, individually-runnable block of the suite."""

    def __init__(self, name, fn, needs, provides, historical_stance):
        self.name = name
        self.fn = fn
        self.needs = needs
        self.provides = provides
        self.historical_stance = historical_stance


_CASES = []
_CASE_BY_NAME = {}
_PRODUCER = {}


def _case(name, provides=(), historical_stance=False):
    """Register a case. Parameters are consumed from the context; ``provides``
    names what the case hands on to later cases."""
    def decorate(fn):
        needs = tuple(inspect.signature(fn).parameters)
        entry = _Case(name, fn, needs, tuple(provides), historical_stance)
        if name in _CASE_BY_NAME:
            raise ValueError(f"duplicate smoke case name: {name}")
        _CASES.append(entry)
        _CASE_BY_NAME[name] = entry
        for key in entry.provides:
            _PRODUCER.setdefault(key, name)
        return fn
    return decorate


@contextlib.contextmanager
def _historical_pin_warnings():
    """Mute the DeprecationWarnings raised by the two historical-pin helpers.

    Scoped to the keys ``_HISTORICAL_PIN_KEYS`` lists and to cases that
    declare ``historical_stance=True``; everything else warns as usual.
    """
    with warnings.catch_warnings():
        for key in _HISTORICAL_PIN_KEYS:
            warnings.filterwarnings(
                "ignore", message=f"{key}=", category=DeprecationWarning
            )
        yield


def _run_case(entry, ctx, trace=False):
    """Run one case against the shared context and harvest what it provides."""
    if trace:
        print(f"case: {entry.name}", file=sys.stderr, flush=True)
    missing = [k for k in entry.needs if k not in ctx]
    if missing:
        where = ", ".join(
            f"{k} (from {_PRODUCER.get(k, 'an earlier case')})" for k in missing
        )
        raise SystemExit(
            f"case {entry.name!r} needs values it did not receive: {where}. "
            "Run the full suite, or add the producing case to --only."
        )
    kwargs = {k: ctx[k] for k in entry.needs}
    if entry.historical_stance:
        with _historical_pin_warnings():
            result = entry.fn(**kwargs)
    else:
        result = entry.fn(**kwargs)
    for key in entry.provides:
        if result is not None and key in result:
            ctx[key] = result[key]


def _anode_sink_config():
    """A small split-stance configuration with a live cathode and anode.

    ``default_config()`` ships ``implicit_heat_conduction`` on, so the anode
    electron-sheath row is in the implicit substep here -- which is what
    these cases are about. The scheduled phases put the run straight into
    the discharge so the anode actually collects.
    """
    params, flags = default_config()
    params.update({
        "nx": 16,
        "nx_gap": 2,
        "ne0": 5.0e11,
        "nn0": 2.0e13,
        "Te0": 3.0,
        "Ti0": 1.0,
        "phase_transition_mode": "scheduled",
        "tau_prebreakdown": 0.0,
        "tau_breakdown": 0.0,
        "tau_discharge": 1.0e-3,
        "initial_neutral_state": "fill",
    })
    flags.update({
        "cathode_coupling": True,
    })
    return params, flags


def _anode_sink_sim(steps=60):
    """A stepped sim on ``_anode_sink_config`` plus its anode cell pair."""
    params, flags = _anode_sink_config()
    sim = LAPDSim1D(params, flags)
    for _ in range(steps):
        sim.advance_one_step()
    pair = anode_flanking_cells(sim.geometry)[0]
    return sim, [int(pair[0]), int(pair[1])]


# ----------------------------------------------------------------------
# Registry census, asserted when the package is imported.
#
# These counts used to sit in the module docstring as prose, where nothing
# checked them and they drifted silently -- the sentence claimed 114 while
# the registry held 116. They live here instead, and every import of the
# ``smoke`` package re-derives them from ``_CASES`` (``smoke/__init__.py``
# calls the check once every case module has registered) and fails loudly on
# a mismatch, so
# adding or removing a case cannot leave a stale number behind.
# ----------------------------------------------------------------------
_CASE_CENSUS = {"total": 189, "historical_stance": 67}


def _assert_case_census():
    """Fail at import if ``_CASE_CENSUS`` no longer describes ``_CASES``."""
    live = {
        "total": len(_CASES),
        "historical_stance": sum(1 for e in _CASES if e.historical_stance),
    }
    if live != _CASE_CENSUS:
        raise AssertionError(
            f"smoke case census is stale: the registry holds {live}, "
            f"_CASE_CENSUS records {_CASE_CENSUS}. Update _CASE_CENSUS in "
            "the same commit that adds or removes a case."
        )


# ----------------------------------------------------------------------
# Case-body reachability, asserted at import.
#
# A case that ends in a ``return`` (the registry's way of handing values to a
# later case) is one careless insertion away from burying the clauses that
# follow it. Nothing catches that on its own: the suite still exits 0, the
# case still "passes", and the buried assertions simply stop running -- the
# worst failure a gate can have, because it is indistinguishable from a green
# one. It happened, to eleven lines of a construction-refusal loop.
#
# So the shape is checked rather than trusted, syntactically: nothing after a
# ``return`` or ``raise`` at the TOP LEVEL of a case body can execute, whatever
# the values, and that is decidable from the parse tree alone. Only the
# function's own top level is inspected -- a return inside an ``if`` or a
# ``for`` says nothing about what follows the block, and the several cases
# that raise inside a ``try``/``else`` are untouched.
# ----------------------------------------------------------------------
def _unreachable_case_statements(source, case_function_names):
    """Return one ``(function, kind, terminator_line, dead_line)`` per offender.

    Takes the SOURCE rather than reading a file, so the check can be pointed at
    any revision of this module -- which is how its own negative control runs.
    """
    findings = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in case_function_names:
            continue
        for index, statement in enumerate(node.body):
            if not isinstance(statement, (ast.Return, ast.Raise)):
                continue
            if index + 1 < len(node.body):
                findings.append(
                    (
                        node.name,
                        "return" if isinstance(statement, ast.Return) else "raise",
                        statement.lineno,
                        node.body[index + 1].lineno,
                    )
                )
            break
    return findings


def _assert_case_bodies_reachable():
    """Fail at import if a registered case buries statements after a return.

    Each case module is parsed on its own, against the names of the cases
    it registered.
    """
    by_file = {}
    for entry in _CASES:
        by_file.setdefault(inspect.getsourcefile(entry.fn), set()).add(
            entry.fn.__name__
        )
    findings = []
    for source_file, names in sorted(by_file.items()):
        findings.extend(
            _unreachable_case_statements(
                Path(source_file).read_text(encoding="utf-8"), names
            )
        )
    if findings:
        where = "; ".join(
            f"{name}: line {dead} onwards is unreachable past the {kind} "
            f"at line {terminator}"
            for name, kind, terminator, dead in findings
        )
        raise AssertionError(
            "smoke case body has unreachable statements -- those assertions "
            f"are silently not running: {where}. Move the terminating "
            "statement to the END of the case body."
        )


def main(argv=None):
    """Run the suite. No arguments = the full gate, in registration order."""
    parser = argparse.ArgumentParser(
        description="LAPDSim1D assertion smoke suite (case registry)."
    )
    parser.add_argument(
        "--list", action="store_true",
        help="print the case names, in run order, and exit",
    )
    parser.add_argument(
        "--only", action="append", default=[], metavar="NAME[,NAME...]",
        help="run only these cases (repeatable, comma-separated); they run in "
             "registration order, never in the order given",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="log each case name to stderr as it starts",
    )
    args = parser.parse_args(argv)

    if args.list:
        for entry in _CASES:
            print(entry.name)
        return 0

    requested = [n for chunk in args.only for n in chunk.split(",") if n]
    if requested:
        unknown = [n for n in requested if n not in _CASE_BY_NAME]
        if unknown:
            raise SystemExit(
                "unknown case name(s): %s (see --list)" % ", ".join(unknown)
            )
        wanted = set(requested)
        selected = [e for e in _CASES if e.name in wanted]
    else:
        selected = list(_CASES)

    ctx = {}
    for entry in selected:
        _run_case(entry, ctx, trace=args.trace)
    return 0
