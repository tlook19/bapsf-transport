"""Shared primitives of the cathode/anode/bank circuit solves.

The two sheath solves -- the current-driven one
(:mod:`cablp.cathode.circuit_idriven`) and the prescribed-measured one
(:mod:`cablp.cathode.circuit_prescribed`) -- are assembled from the pieces
defined here: the static device record, the plasma sample, the result record,
the space-charge emission ceiling, the beam mean free path and gap bypass, the
beam excitation channel, and the electrode power expressions.

Units
-----
- Geometry        : CGS (cm)
- Mass            : CGS (g)
- Temperature     : T_e in eV, T_s in K
- Current         : Amperes [A]
- Resistance      : Ohms [Ω]
- Potential       : Volts [V]
- Power           : Watts [W]

Scaled (dimensionless) quantities:
  psi = phi / T_e        (potential scaled by electron temperature in V)
  J   = I * R_p / T_e   (current scaled by plasma resistance and T_e)
  delta = KB_SI * T_s / (E_SI * T_e)   (temperature ratio T_s / T_e in same units)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from cablp.cathode.kernels import COMPILED_KERNELS as _COMPILED_KERNELS
from cablp.atomic.coefficients import b_11s_21p
from cablp.constants import E_21p as _E_21p_eV, Ry_eV as _Ry_eV, atm_cross_cgs as _atm_cross_cgs

# ---------------------------------------------------------------------------
# Universal physical constants
# ---------------------------------------------------------------------------

E_SI: float = 1.602176634e-19  # Electron charge [C] = [J/eV]
KB_SI: float = 1.380649e-23  # Boltzmann constant [J/K]
ME_CGS: float = 9.1093837015e-28  # Electron mass [g]
MP_CGS: float = 1.67262192369e-24  # Proton mass [g]
PEMR: float = MP_CGS / ME_CGS  # Proton-to-electron mass ratio ≈ 1836.15
ERG_PER_EV: float = E_SI * 1.0e7  # eV → erg conversion


def sheath_lift_lambda(ion_mass_g: float) -> float:
    """Return the sheath lift ``Lambda`` [dimensionless] for the ion mass.

    ``Lambda = ln(sqrt(m_i / (2 pi m_e)))`` is the floating-potential
    parameter in units of ``T_e``: the barrier a Maxwellian electron
    population must climb for its one-sided random flux to be throttled to
    the ion flux at a surface drawing no net current. ``ion_mass_g`` is the
    ion mass [g] and must be positive; helium gives 3.529.

    THE ONE SPEC. :class:`DeviceConfig` stores this as its ``Lambda`` and
    every sheath current rides that value, so a consumer outside the circuit
    that needs the same barrier calls this instead of restating the
    expression.
    """
    return math.log(math.sqrt(ion_mass_g / (2.0 * math.pi * ME_CGS)))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Static device configuration; does not change between RK steps.

    Parameters
    ----------
    A_c     : Cathode area [cm²]
    mu      : Ion mass / proton mass [dimensionless]; sets the electron-to-ion mass ratio of the space-charge emission ceiling only
    ion_mass_g : Ion mass [g]; sets the sound speed the ion collection currents ride and the sheath lift ``Lambda``
    T_s     : Cathode surface temperature [K]
    phi_wf  : Work function [eV]; default 3.0 (LaB6)
    C_R     : Richardson constant [A cm⁻² K⁻²]; default 29 (LaB6)
    R_comp  : Compliance resistor [Ω]; default 0.004
    R_comp_partition : Bank-side share x of ``R_comp`` [dimensionless]; default 1.0
    R_mesh_ohm : Anode-mesh series resistance [Ω]; default 0.0
    eta     : Anode-mesh solid fraction (opacity) [dimensionless]; the share of the anode face its wires occupy, so 1 - eta transmits. It sets the anode's ion collection area (``I_i_a = 2*eta*I_i``, both mesh faces) and the share of the gap-surviving thermionic beam the mesh intercepts (``eta*beam_bypass_fraction``). Valid range [0, 1]; eta = 0 is admissible for the anode's neutral and heat throttles but makes the discharge circuit singular and raises. Default 0.358
    L_cath  : Cathode-to-anode distance [cm]; default 50
    R_cath  : Cathode radius [cm]; default 18
    emission_Ts_K, emission_area_cm2, emission_plasma_frac : Annular emission profile; empty tuples (the default) are the uniform disc
    """

    A_c: float
    mu: float
    ion_mass_g: float
    T_s: float
    phi_wf: float = 3.0
    C_R: float = 29.0
    R_comp: float = 0.004
    # Voltage-probe partition of R_comp plus a separate anode-mesh R.
    # R_external = R_comp_partition*R_comp is bank-side of the probe (in
    # V_dis = V_bank - I*R_external); R_internal = (1-x)*R_comp and R_mesh_ohm
    # are probe->plasma (invisible to the V_dis formula), so the circuit
    # integrates the device voltage V_b + I*((1-x)*R_comp + R_mesh_ohm). They
    # lower the current, which raises V_dis. At the defaults (x=1, R_mesh=0)
    # the internal series drop is identically zero.
    R_comp_partition: float = 1.0
    R_mesh_ohm: float = 0.0
    eta: float = 0.358
    L_cath: float = 50.0
    R_cath: float = 18.0
    # Annular emission profile (empty tuples = the uniform disc). A real
    # cathode's surface temperature falls off radially, so its emission
    # ceiling is a soft ramp, not a razor wall: as the sheath pulls harder the
    # virtual-cathode clamp releases progressively cooler annuli. Each annulus
    # carries its own Richardson emission (T_k over area_k), its own share of
    # the ion current (its overlap with the plasma footprint), and its own
    # space-charge clamp; all share one equipotential psi_c_plus. The
    # current-driven solve's annular Schottky branch and its compiled root
    # (``solve_psi_annular_schottky``) read these.
    emission_Ts_K: tuple = ()
    emission_area_cm2: tuple = ()
    emission_plasma_frac: tuple = ()

    # Derived constants computed once at construction
    # (stored as slots; frozen prevents reassignment)
    Lambda: float = field(init=False)
    I_eth: float = field(init=False)

    def __post_init__(self) -> None:
        # Lambda = sheath floating-potential parameter, from the module's
        # one spec (`sheath_lift_lambda`) so an outside consumer of the same
        # barrier reads the same expression rather than a copy of it.
        lam = sheath_lift_lambda(self.ion_mass_g)
        object.__setattr__(self, "Lambda", lam)

        # I_eth = thermionic emission current [A] (static; depends only on T_s)
        if self.emission_Ts_K:
            i_eth = sum(
                area
                * self.C_R
                * T_k**2
                * math.exp(-E_SI * self.phi_wf / (KB_SI * T_k))
                for T_k, area in zip(self.emission_Ts_K, self.emission_area_cm2)
            )
        else:
            i_eth = (
                self.A_c
                * self.C_R
                * self.T_s**2
                * math.exp(-E_SI * self.phi_wf / (KB_SI * self.T_s))
            )
        object.__setattr__(self, "I_eth", i_eth)


@dataclass(slots=True)
class PlasmaState:
    """Plasma state provided by the Runge-Kutta solver at each step.

    Parameters
    ----------
    T_e    : Electron temperature [eV]
    n_e    : Electron density [cm⁻³]
    n_n    : Neutral density [cm⁻³]; used for beam MFP (default 0)
    sigma_b: Beam ionization cross-section [cm²]; used for beam MFP (default 0)
    """

    T_e: float
    n_e: float
    n_n: float = 0.0
    sigma_b: float = 0.0


@dataclass(slots=True)
class SolverResult:
    """All quantities returned by a cathode sheath solve.

    Potentials [V]
    --------------
    phi_c_plus  : Classical (positive) part of cathode sheath drop
    phi_c_minus : Inverted (virtual-cathode) part of cathode sheath drop
    phi_c       : Total cathode sheath drop  = phi_c_plus - phi_c_minus
    phi_a       : Anode sheath potential (may be negative above plasma potential)
    V_p         : Plasma ohmic voltage drop
    V_b         : Bias voltage across anode and cathode

    Resistance [Ω]
    --------------
    R_p         : Parallel plasma resistance

    Currents [A]
    ------------
    I_i         : Ion saturation current
    I_e         : Electron saturation current
    I_eth       : Total thermionic emission current (config constant)
    I_eth_star  : Allowed thermionic current: the effective emission clamped by
                  the virtual-cathode (space-charge) limit. With Schottky
                  barrier lowering the effective emission is enhanced, in
                  scaled units J_eff = J_eth*exp(dphi/(delta*T_e)),
                  where dphi is the Schottky work-function lowering [eV] set by
                  the sheath surface field and delta*T_e is the emitter surface
                  temperature in eV
    I_tot       : Net circuit current

    Power [W]
    ---------
    P_wall      : Total power demanded from supply  = I_tot * (V_b + I_tot * R_comp)
    P_load      : Power delivered to plasma load    = I_tot * V_b
    P_comp      : Compliance resistor dissipation   = I_tot² * R_comp
    P_prim      : Primary-electron power into plasma, priced at the launched
                  electron current I_eth_star times phi_c, netted by the gap
                  survival
    P_ohmic     : Plasma ohmic heating              = I_tot * V_p
    P_loss      : Sheath power loss = P_cathode_e + P_cathode_i_pl + P_anode_e + P_anode_i_pl

    Metadata
    --------
    regime      : 'classical', 'virtual_cathode', 'capability_limited' or
                  'prescribed'
    """

    # Potentials [V]
    phi_c_plus: float
    phi_c_minus: float
    phi_c: float
    phi_a: float
    V_p: float
    V_b: float
    # Resistance [Ω]
    R_p: float
    # Currents [A]
    I_i: float
    I_i_a: float
    I_e: float
    I_eth: float
    I_eth_star: float
    I_tot: float
    # Power [W]
    P_wall: float
    P_load: float
    P_comp: float
    P_prim: float
    P_ohmic: float
    P_cathode_e: float
    P_cathode_i: float
    P_cathode_i_pl: float
    P_anode_e: float
    P_anode_i: float
    P_anode_i_pl: float
    P_net: float
    P_net2: float
    P_loss: float
    # Regime
    regime: Literal["classical", "virtual_cathode"] = "classical"
    long_mfp: bool = False
    beam_bypass_fraction: float = 0.0
    # Beam mean free path [cm]; 0.0 if beam parameters not provided
    l_b: float = 0.0
    # The electron temperature [eV] the ANODE block of this solve ran on: the
    # caller's ``anode_T_e`` sample where one was given, and the cathode's own
    # ``T_e`` otherwise. Every anode quantity below -- ``phi_a`` and the four
    # ``P_anode_*`` members -- is referenced to it, so a consumer that wants
    # the anode rows at some other temperature has the scaling temperature
    # here rather than having to guess which sample the solve used.
    T_e_anode: float = 0.0
    # R3.2 (A16) one-control-surface split. Each electrode
    # power splits into a PLASMA-THERMAL part (Te/2 per ion, 2Te per electron --
    # sourced from the plasma thermal store) and a SHEATH-FALL phi part (sourced
    # from the sheath field / circuit, deposited on the electrode, never through
    # the plasma thermal store). ``*_thermal + *_phi == P_*_e/_i`` by construction.
    # These are the values the repaired fluid boundary reads so fluid == circuit.
    #
    # DEPRECATED UNCLOSED SCALARS, kept computed only so the R1-R4 golden stays
    # bit-exact and an old file's numbers can still be reproduced here. They are
    # NO LONGER EXPORTED to the HDF5. Each name and what to read instead:
    #   P_cathode_i_pl -> P_cathode_i_thermal. P_cathode_i_pl is built from
    #       I_i_a, the ANODE ion current, under a cathode name.
    #   P_anode_i_pl   -> P_anode_i_thermal.
    #   P_loss         -> P_plasma_thermal_loss. P_loss sums phi-INCLUSIVE
    #       electron powers with thermal-only ion powers, and takes the
    #       cathode ion term from the anode current.
    #   P_net          -> P_into_plasma for the power heating the plasma, or
    #       P_load_residual for the load-power closure check. P_net subtracts
    #       full electrode powers from load field work, booking each sheath
    #       fall twice.
    #   P_net2         -> P_into_plasma, which is what it was reaching for.
    #   P_comp         -> nothing; it is I_tot**2 * R_comp and was never read.
    # The closed audit that replaces them is below.
    P_cathode_e_thermal: float = 0.0
    P_cathode_e_phi: float = 0.0
    P_cathode_i_thermal: float = 0.0
    P_cathode_i_phi: float = 0.0
    P_anode_e_thermal: float = 0.0
    P_anode_e_phi: float = 0.0
    # The QL tail's sheath-fall moment at the anode, ``I_tail_a * phi_a``
    # [W], the partner of ``P_anode_e_phi`` for the current the wires take out
    # of the walked tail rather than out of the thermal return. It is power
    # the CIRCUIT pays and the mesh receives, never routed through the plasma
    # thermal store, exactly like the primary's bypass convention. Identically
    # 0.0 at a non-positive ``phi_a`` (an attracting anode charges no fall)
    # and whenever no tail current is handed in. ``I_tail_a`` is the LAGGED
    # net culled current: the deposition is solved after the circuit within a
    # step, so this solve reads the previous accepted step's cull.
    P_tail_phi: float = 0.0
    P_anode_i_thermal: float = 0.0
    P_anode_i_phi: float = 0.0
    # Closed surface-resolved audit [W], replacing P_net/P_net2:
    # net power heating the plasma vs power onto each electrode surface.
    P_plasma_thermal_loss: float = 0.0   # total plasma-thermal loss to electrodes
    P_into_plasma: float = 0.0           # P_prim + P_ohmic - plasma-thermal loss
    P_cathode_surface: float = 0.0       # plasma power onto the cathode (thermal+phi)
    P_anode_surface: float = 0.0         # plasma power onto the anode (thermal+phi)
    # Measurement-plane bookkeeping aliases. The Poulos names
    # I_tot / V_b are the MODEL LOAD quantities and are kept as-is; these alias to
    # the three-plane convention so a future effective-load change diverges
    # predictably without renaming. All divergences are identically zero until
    # the two declared-but-unpopulated circuit terms defined below land: a
    # hidden series impedance between the bank terminals and the plasma load
    # (``V_series``) and a stray branch carrying current around the
    # load (``I_parallel``). Both are pinned at 0.0 today, so this is
    # scaffolding for those terms, not dead code:
    #   V_dis = V_b + V_series     (measured terminal/discharge voltage)
    #   I_bank = I_plasma + I_parallel   (measured bank/terminal current, I_dis)
    #   I_plasma == I_tot now      (current conducted through the plasma load)
    # so today V_dis == V_b, I_bank == I_plasma == I_tot, and every V*I product
    # coincides: P_load = V_b*I_tot = V_dis*I_bank. The closure is referenced to
    # P_load (power across the load), NOT the terminal product.
    V_series: float = 0.0      # hidden series-impedance drop [V]; 0 today
    I_parallel: float = 0.0    # stray/parallel branch current [A]; 0 today
    V_dis: float = 0.0         # measured terminal voltage [V] = V_b + V_series
    I_plasma: float = 0.0      # plasma-conducted current [A] = I_tot now
    I_bank: float = 0.0        # measured bank current [A] = I_plasma + I_parallel
    # Load-power closure diagnostic. The circuit does I_tot*V_b
    # of NET field work across the load; by the potential ladder V_b = phi_c + V_p
    # - phi_a and Kirchhoff (the same net loop current threads each region), that
    # is the per-region field work, and each region decomposes per species with
    # the current DIRECTIONS respected -- at the cathode the emission and ion
    # collection deliver field energy while the RETURNING plasma-electron current
    # recovers it: I_tot = I_eth_star + I_i - I_e_ret. The sheath phi work is drawn
    # from the CIRCUIT (it circulates: the cathode sheath accelerates carriers, the
    # anode recovers), NOT from the plasma thermal store -- which is exactly why
    # R3.2 routes phi to the electrode book and 2Te/Te/2 to the plasma book. The
    # beam plasma deposition P_prim and the bulk-current ohmic are a SEPARATE
    # plasma-heating book, not the circuit field work.
    I_e_ret: float = 0.0           # returning plasma-electron current to cathode [A]
    P_load_ledger: float = 0.0     # per-region net-current field work [W]
    P_load_residual: float = 0.0   # P_load - P_load_ledger [W] (~0 by closure)
    I_cathode_kirchhoff_residual: float = 0.0  # (I_eth_star+I_i-I_e_ret) - I_tot [A]
    # Ceiling census. ``phi_c_ceiling_V`` is the ceiling the sheath root was
    # solved against, the atomic-data cap ``phi_c_cap_V``.
    # ``circuit_V_avail_V`` is NaN on every solve: no circuit-available
    # voltage bound is formed, and the field is carried because the saved
    # cathode diagnostics export it. ``bound_active`` says whether the solve
    # ended up sitting on the ceiling: 0 = no (the returned phi_c is a free
    # root), 1 = the data cap. It is derived from the regime tag, so nothing
    # is recomputed to produce it.
    phi_c_ceiling_V: float = float("nan")
    circuit_V_avail_V: float = float("nan")
    bound_active: float = float("nan")
    # Anode balance census. ``anode_floor_fired`` is 1.0 when a non-probe
    # solve under ``anode_tail_booking="lagged_current"`` floored the anode
    # sheath's electron current at 1e-300 A (a balance with no floating
    # solution, kept for that booking), 0.0 otherwise.
    # ``anode_direct_collected_fraction`` is the share of the whole fast term
    # (the primary's direct interception, its net return interception and the
    # tail) the anode booked under ``"emission_fraction"`` (1 where the beam
    # clears the anode sheath, 0 where it cannot, between where the sheath is
    # pinned at the beam energy), the same share of each part; NaN under
    # ``"lagged_current"``. ``anode_fast_branch`` is the branch that set it
    # (``ANODE_FAST_BRANCH_BOOKED`` 0, ``_PINNED`` 1, ``_NONE`` 2); NaN under
    # ``"lagged_current"``.
    anode_floor_fired: float = 0.0
    anode_direct_collected_fraction: float = float("nan")
    anode_fast_branch: float = float("nan")


def beam_launched_current_A(result):
    """Return the electron current [A] the cathode LAUNCHES into the gap.

    ``I_eth_star``, the space-charge-released thermionic current, which is the
    whole of the population the cathode fall accelerates across the gap. The
    ONE definition every launched-flux reader takes -- the Beer-Lambert
    beam-array assembly, the CSDA march's ``Gamma0``, and the electron
    drift-transport operator's beam current -- so a build cannot end up with
    two launched fluxes.
    """
    return result.I_eth_star


#: How the anode sheath balance books the fast electrons the mesh collects
#: directly. ``"lagged_current"`` subtracts ``eta * beta * J_star`` for the
#: primary and the previous accepted step's absolute tail-walker current.
#: ``"emission_fraction"`` subtracts ``eta * beta * (1 - w_gap) * J_star`` for
#: the primary, net of the walker flux born upstream of the anode plane,
#: ``c_ret * J_star`` for the primary's net interception on its returns, and
#: ``c_tail * J_star`` for the tail, both coefficients per emitted electron
#: from the previous deposition; the three together only where the beam of
#: the solve that applies them clears the anode sheath.
ANODE_TAIL_BOOKINGS = ("lagged_current", "emission_fraction")


def resolve_anode_tail_booking(
    booking,
    tail_anode_current_A,
    tail_anode_coefficient,
    anode_gap_walker_fraction,
    primary_return_coefficient=0.0,
):
    """Validate one solve's anode fast-electron booking; return the booking.

    ``"lagged_current"`` reads ``tail_anode_current_A`` only, and refuses a
    non-zero coefficient or gap fraction. ``"emission_fraction"`` reads the
    per-emission coefficients ``tail_anode_coefficient`` (``c_tail``) and
    ``primary_return_coefficient`` (``c_ret``, both ``>= 0``) and the
    gap-born walker fraction ``anode_gap_walker_fraction`` (``w_gap``, in
    ``[0, 1]``), and refuses a non-zero absolute tail current. Raises
    ``ValueError`` naming the accepted values otherwise.
    """
    booking = str(booking)
    if booking not in ANODE_TAIL_BOOKINGS:
        raise ValueError(
            "anode_tail_booking must be 'lagged_current' or "
            f"'emission_fraction' (got {booking!r})"
        )
    c_tail = float(tail_anode_coefficient)
    w_gap = float(anode_gap_walker_fraction)
    c_ret = float(primary_return_coefficient)
    if booking == "lagged_current":
        if c_tail != 0.0 or w_gap != 0.0 or c_ret != 0.0:
            raise ValueError(
                "anode_tail_booking='lagged_current' books the tail as the "
                "lagged absolute current tail_anode_current_A; a tail "
                f"coefficient ({c_tail!r}), return coefficient ({c_ret!r}) or "
                f"gap walker fraction ({w_gap!r}) belongs to "
                "'emission_fraction'"
            )
        return booking
    if float(tail_anode_current_A) != 0.0:
        raise ValueError(
            "anode_tail_booking='emission_fraction' books the tail as a "
            "coefficient on this solve's emission; an absolute "
            f"tail_anode_current_A ({tail_anode_current_A!r} A) belongs to "
            "'lagged_current'"
        )
    if not (math.isfinite(c_tail) and c_tail >= 0.0):
        raise ValueError(
            f"the tail coefficient must be finite and >= 0 (got {c_tail!r})"
        )
    if not (math.isfinite(c_ret) and c_ret >= 0.0):
        raise ValueError(
            f"the return coefficient must be finite and >= 0 (got {c_ret!r})"
        )
    if not (math.isfinite(w_gap) and 0.0 <= w_gap <= 1.0):
        raise ValueError(
            f"the gap walker fraction must be in [0, 1] (got {w_gap!r})"
        )
    return booking


#: The branch codes ``emission_fraction_anode_balance`` returns for the fast
#: term: booked whole, pinned (the collected share set by the balance), and
#: none of it booked.
ANODE_FAST_BRANCH_BOOKED = 0.0
ANODE_FAST_BRANCH_PINNED = 1.0
ANODE_FAST_BRANCH_NONE = 2.0


def emission_fraction_anode_balance(
    I_rest_A, I_fast_A, I_e_sat_A, T_e_anode, E_beam_eV, probe, describe
):
    """The anode sheath under ``anode_tail_booking="emission_fraction"``.

    ``I_rest_A`` is the electron current the anode sheath must pass with no
    directly collected fast electron taken out (``I_i,a + I_tot``),
    ``I_fast_A`` the whole fast term the anode books directly (the primary's
    direct interception, its net return interception and the tail,
    ``(eta * beta * (1 - w_gap) + c_ret + c_tail) I_star``) and
    ``E_beam_eV`` the launch energy ``e phi_c`` of the solve that applies it,
    and ``phi_a(I) = T_e,a ln(I_e,sat / I)``. A fast electron of this solve's
    beam carries at most ``e phi_c``, so the whole term is booked only where
    it clears the sheath it books, by three branches and no root-find: with
    the term, accepted if ``phi_a < E_beam``; else without it, accepted if
    ``phi_a >= E_beam``; else ``phi_a`` pinned at ``E_beam`` with the
    balance setting the collected share of the term in ``[0, 1)``.
    Returns ``(phi_a, collected_fraction, branch)`` with ``branch`` one of
    ``ANODE_FAST_BRANCH_BOOKED``, ``ANODE_FAST_BRANCH_PINNED`` and
    ``ANODE_FAST_BRANCH_NONE``.

    A balance with no floating solution (``I_rest_A`` below 1e-300 A on the
    without-term branch) raises ``ValueError`` quoting ``describe()``,
    except under ``probe``, which keeps the floored value
    ``T_e,a ln(I_e,sat / 1e-300)``.
    """
    def phi(current):
        if current > 0.0:
            return T_e_anode * math.log(I_e_sat_A / current)
        return math.inf

    with_term = float(I_rest_A) - float(I_fast_A)
    phi_with = phi(with_term)
    if phi_with < E_beam_eV:
        return phi_with, 1.0, ANODE_FAST_BRANCH_BOOKED
    phi_without = phi(I_rest_A)
    if phi_without >= E_beam_eV:
        if 1e-300 > I_rest_A:
            if not probe:
                raise ValueError(
                    "the anode sheath balance is infeasible: the electron "
                    "current it must pass with no directly collected fast "
                    f"electron taken out, {I_rest_A!r} A, is not positive "
                    f"({describe()})"
                )
            return (
                T_e_anode * math.log(I_e_sat_A / 1e-300), 0.0,
                ANODE_FAST_BRANCH_NONE,
            )
        return phi_without, 0.0, ANODE_FAST_BRANCH_NONE
    I_pinned = I_e_sat_A * math.exp(-E_beam_eV / T_e_anode)
    return (
        float(E_beam_eV),
        (float(I_rest_A) - I_pinned) / float(I_fast_A),
        ANODE_FAST_BRANCH_PINNED,
    )


@dataclass(slots=True)
class BeamResult:
    """Beam quantities of a solved cathode sheath, per cell.

    Built by :func:`cablp.cathode.circuit_idriven.assemble_beam_arrays` for
    both the current-driven and the prescribed-measured solve. Arrays have
    shape (cells,); non-source cells are zero.

    Fields
    ------
    result          : SolverResult for the primary cathode (index 0)
    result_twin     : SolverResult for the twin cathode (index -1); None from
                      the single-cathode assembly
    v_beam          : beam electron velocity [cm/s]
    n_beam          : beam electron density [cm⁻³]
    beam_cross      : EII cross section at beam energy [cm²]
    beam_atten_cross: attenuation cross section fed back to the next sheath
                      solve as ``beam_cross_prev`` [cm²]; assembled equal to
                      ``beam_cross``, and the Sim1D cathode solve overwrites
                      its launch cell with the effective cross section
                      inverted from the CSDA gap transmission
    n_beam_ion      : n_beam * beam_cross * v_beam  [s⁻¹]
    A_ion_beam      : n_beam_ion * nn  [cm⁻³ s⁻¹]
    l_b             : beam mean free path per cathode cell [cm]; 0 elsewhere
    p_beam          : neutral ionization probability = l_b * beam_cross * nn [dimensionless]
    l_b_profile     : per-cell MFP for the primary beam [cm], at the launch
                      cell's ``beam_atten_cross``; assembled as zeros and
                      filled by the Sim1D cathode solve once it has written
                      the attenuation cross section it feeds back
    l_b_profile_twin: per-cell MFP for the twin beam [cm]; zeros if no twin or beam_cross[-1]==0
    x0_next         : the solved ``phi_c_plus`` [V], exported as a cathode
                      diagnostic; the current-driven solve takes no warm start
    x0_twin_next    : the twin's counterpart, or None
    """

    result: SolverResult
    result_twin: SolverResult | None
    v_beam: np.ndarray
    n_beam: np.ndarray
    beam_cross: np.ndarray
    beam_atten_cross: np.ndarray
    n_beam_ion: np.ndarray
    A_ion_beam: np.ndarray
    l_b: np.ndarray
    p_beam: np.ndarray
    l_b_profile: np.ndarray
    l_b_profile_twin: np.ndarray
    x0_next: float
    x0_twin_next: float | None


# ---------------------------------------------------------------------------
# Sheath, emission and beam primitives
# ---------------------------------------------------------------------------


def exp_clamped(x: float) -> float:
    """``math.exp`` with the argument capped at 700 (just under overflow).

    Bit-exact for any argument below the cap; above it, returns a finite
    ~1e304 so an extreme root-bracket probe yields a huge finite residual the
    solver can reject, instead of an OverflowError that kills the whole run.
    """
    return math.exp(min(x, 700.0))


def c_log_ei(T_e: float, n_e: float) -> float:
    """Electron-ion Coulomb logarithm (NRL 2019, eqs. 2-3/2-4)."""
    if T_e > 10.0:
        return 24.0 - math.log(math.sqrt(n_e) / T_e)
    return 23.0 - math.log(math.sqrt(n_e) * T_e**-1.5)


def annular_emission_state(
    psi_c_plus: float,
    J_i: float,
    mu: float,
    J_eth_k: tuple,
    delta_k: tuple,
    ion_frac_k: tuple,
) -> tuple[float, float, bool]:
    """Return ``(J_star_total, psi_minus_eff, any_clamped)`` for the annuli.

    Each annulus sees its own space-charge limit from its share of the ion
    current (``j_eth_crit`` is linear in ``J_i``); an annulus outside the
    plasma footprint (``ion_frac 0``) is fully choked. All annuli share the
    equipotential ``psi_c_plus``; the loop equation's effective
    virtual-cathode drop is the emission-weighted mean of the local barriers,
    which reduces exactly to the uniform expression for a single annulus.
    """
    J_star_total = 0.0
    weighted_pm = 0.0
    any_clamped = False
    for J_eth_a, delta_a, frac_a in zip(J_eth_k, delta_k, ion_frac_k):
        if J_eth_a <= 0.0:
            continue
        J_crit_a = j_eth_crit(psi_c_plus, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
        if J_eth_a <= J_crit_a:
            J_star_total += J_eth_a
            continue
        any_clamped = True
        if J_crit_a <= 0.0:
            continue  # fully choked annulus emits nothing
        pm_a = delta_a * math.log(J_eth_a / J_crit_a)
        J_star_total += J_crit_a
        weighted_pm += J_crit_a * pm_a
    psi_minus_eff = weighted_pm / J_star_total if J_star_total > 0.0 else 0.0
    return J_star_total, psi_minus_eff, any_clamped


def compute_l_b(phi_c: float, T_e: float, n_e: float, n_n: float, sigma_b: float) -> float:
    """Beam mean free path [cm] for primary electrons accelerated through phi_c [V]."""
    if phi_c <= 0.0:
        return 0.0
    v_beam = math.sqrt(2.0 * phi_c * ERG_PER_EV / ME_CGS)
    tau_ei = 3.44e5 * T_e**1.5 / n_e / c_log_ei(T_e, n_e)
    l_bi = v_beam * tau_ei
    if sigma_b > 0.0 and n_n > 0.0:
        l_bn = 1.0 / (sigma_b * n_n)
        return 1.0 / (1.0 / l_bi + 1.0 / l_bn)
    return l_bi


def compute_beam_bypass_fraction(l_b: float, L_cath: float) -> float:
    """Fraction of anode-directed thermionic beam that survives to the anode."""
    if l_b <= 0.0 or L_cath <= 0.0:
        return 0.0
    return math.exp(-L_cath / l_b)


_ATM_CROSS_CGS = float(_atm_cross_cgs)
_RY_EV = float(_Ry_eV)
_E21P_EV = float(_E_21p_eV)


def _he_2p_excitation_cross_cm2(eps: float) -> float:
    """Float port of ``He_EIE_cross_DA(eps, b_11s_21p)`` [cm^2].

    The mpmath original costs ~20 us per scalar call; this is the same
    dipole-allowed formula in plain floats (agreement asserted to 1e-12 in
    the smoke test).
    """
    a = b_11s_21p
    factor1 = _ATM_CROSS_CGS * _RY_EV / (eps * _E21P_EV)
    factor2 = a[0] * math.log(eps) + sum(
        a[i] * eps ** (1 - i) for i in range(1, 5)
    )
    factor3 = (eps + 1.0) / (eps + a[5])
    return factor1 * factor2 * factor3


def j_eth_crit(psi: float, J_i: float, mu: float) -> float:
    """Scaled critical thermionic current J_eth_crit(psi_c_plus).

    J_eth_crit = J_i * sqrt(mu * PEMR) * (exp(-psi) + sqrt(1 + 2*psi) - 2)
                 / sqrt(2 * psi)

    Numerically safe near psi = 0 via a Taylor expansion:
        numerator  ~ (1/3) * psi^3   as psi → 0
        denominator ~ sqrt(2) * psi^(1/2)
        → J_eth_crit ~ J_i * sqrt(mu * PEMR) / (3 * sqrt(2)) * psi^(5/2)
    """
    prefactor = J_i * math.sqrt(mu * PEMR)
    if psi <= 0.0:
        return 0.0
    if psi < 1e-3:
        # Taylor expansion to avoid catastrophic cancellation.
        # Numerator = exp(-psi) + sqrt(1+2*psi) - 2
        #           = psi^3/3 - 7*psi^4/12 + 31*psi^5/60 - ...
        # Denominator = sqrt(2*psi)
        # → J_eth_crit / (J_i*sqrt(mu*pemr)) = psi^(5/2)*(1/3 - 7*psi/12 + ...) / sqrt(2)
        numer = psi**3 * (1.0 / 3.0 - 7.0 * psi / 12.0)
        return prefactor * numer / math.sqrt(2.0 * psi)
    return (
        prefactor
        * (math.exp(-psi) + math.sqrt(1.0 + 2.0 * psi) - 2.0)
        / math.sqrt(2.0 * psi)
    )


# The pure-Python kernels stay reachable under their own names so the compiled
# path can be compared against them (equivalence sweeps, microbenchmarks)
# inside a process that has opted in.
j_eth_crit_pure = j_eth_crit
c_log_ei_pure = c_log_ei
compute_l_b_pure = compute_l_b

# Compiled-kernel selection. Rebinding the module globals HERE, right after
# the definitions and before anything imports the names, is what makes the hot
# path free: callers -- including `circuit_idriven`, which from-imports these
# -- resolve one plain function object with no per-call branch. On the default
# pure path this block is a single `is None` test at import and the names are
# the untouched originals. Each name is rebound in exactly ONE place, the
# module that defines it; `circuit_idriven` owns its own three.
if _COMPILED_KERNELS is not None:
    _COMPILED_KERNELS.check_constants(PEMR, ERG_PER_EV, ME_CGS)
    j_eth_crit = _COMPILED_KERNELS.j_eth_crit
    c_log_ei = _COMPILED_KERNELS.c_log_ei
    compute_l_b = _COMPILED_KERNELS.compute_l_b


def P_ion(phi: float, T_e: float, I_i: float, pl: bool = False) -> float:
    """
    Ion power delivered to an electrode [W].

    For a Bohm-sheath ion current I_i, ions arrive with kinetic energy T_e/2
    plus the sheath acceleration energy phi (in eV, numerically equal to V).
    When pl=True the sheath drop is excluded (plasma-side boundary condition).

    Parameters
    ----------
    phi : float
        Electrode sheath potential [V].
    T_e : float
        Electron temperature [eV].
    I_i : float
        Ion saturation current to the electrode [A].
    pl : bool
        If True, return only the thermal contribution I_i * T_e / 2 (no sheath
        acceleration term); used for the plasma-side power balance.

    Returns
    -------
    float
        Ion power to the electrode [W].
    """
    if pl:
        return I_i * T_e / 2
    else:
        return I_i * (T_e / 2 + phi)
