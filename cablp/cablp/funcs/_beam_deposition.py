"""Solver-agnostic beam deposition along a ray (BEAM_DEPOSITION_PLAN B1).

Deterministic CSDA (continuous-slowing-down) integration of a monoenergetic
primary-electron beam through the column. **Pure function of the beam and the
column** — ``(E0, Gamma0, nn, ne, Te, ray)`` — with no solver state, so both
the voltage-driven and current-driven cathode formulations can consume it at
the same call sites (plan §2).

Energy channels, per unit path length [eV/cm]:

- ionization: ``nn * sigma_ion(E) * (I_ion + <W_sec>(E))`` — the potential
  I_ion is banked as ionization cost (matching the solver's separate
  ``beam_ionization_cost`` term); the mean secondary energy <W_sec> is banked
  as plasma heating (the ejected electron thermalizes locally). <W_sec> uses
  the Opal-Peterson-Beaty differential shape ``dsigma/dW ~ 1/(W^2 + Ebar^2)``
  with the He shape parameter Ebar = 15.8 eV (Opal, Peterson & Beaty, At.
  Data 4, 209 (1971)) over W in [0, (E - I_ion)/2].
- excitation: ``nn * sigma_exc(E) * E_rad(E)`` — the summed Ralchenko singlet
  manifold (``_cross.He_beam_excitation_channel``); banked as radiation.
- Coulomb drag on the plasma, per ``coulomb_model``; banked as plasma heating.
- optional anomalous (beam-plasma instability) drag, per ``anomalous_model``;
  banked as plasma heating (Langmuir turbulence Landau-damps on the bulk
  near where it is driven).

Coulomb closures (both parameter-free):

- ``"fast_electron"`` (default): the physical CSDA stopping power on plasma
  electrons, ``dE/dx = 2 pi e^4 n_e lnLambda / E`` — the collision rate of a
  primary falls as 1/v^3, so at main-discharge conditions (150 eV, n_e =
  5e12) the energy e-fold is ~35 m.
- ``"legacy_tau_ei"``: ``dE/dx = E / (v(E) tau_ei(Te, ne))`` with the
  *thermal* NRL collision time — the historical `_cathode_solver._compute_l_b`
  form (~1.1 m at the same conditions). Provided for continuity experiments;
  its "Coulomb" label overestimates the classical drag ~30x (THESIS_NOTES
  item 12).

Anomalous closure:

- ``"quasilinear"``: the beam-plasma (bump-on-tail) instability drives
  Langmuir waves at v_phase ~ v_b with cold-beam growth rate
  ``~(n_b/n_e)^(1/3) omega_pe`` (growth lengths ~mm-cm here, i.e. saturated
  essentially at the source); quasilinear diffusion then flattens the beam
  over the relaxation length

      l_QL = (n_e / n_b) * (v_b / omega_pe) * ln(n_e / n_b)

  (Vedenov-style estimate; ~0.05-0.1 m at production parameters). Modeled as
  mean-energy drag ``dE/dx = E / l_QL`` with the flux preserved (the module
  carries one mean energy, not a distribution — the plateau spread is a
  documented limitation). **Stated validity domain:** weak beam
  (``n_b << n_e``) with growth beating wave damping — the main-discharge
  column. The breakdown phase (n_b ~ n_e) is outside quasilinear theory and
  deliberately stays collisional. Requires ``beam_area_cm2`` to form
  ``n_b = Gamma0 / (A v_b)``. Order-of-magnitude closure: density-gradient
  detuning and saturation physics can lengthen it substantially; results
  using it must be presented per closure, like the drag story
  (THESIS_NOTES §3).

The primary is followed until it exits the domain (transmitted) or its
energy crosses ``E_stop`` (default: the lowest inelastic threshold, He 2^1S
at 20.6158 eV), where the sub-threshold residual is banked as plasma heating
in the crossing cell. Per-ray energy conservation
``Gamma0*E0 = heating + radiated + ionization cost + transmitted`` holds to
accumulated roundoff by construction (the energy decrement and the channel
banks are the same floating-point sums).

``plasma_heating_erg_s`` lumps four physically distinct deliveries, so the
result also carries them separately as DIAGNOSTIC arrays (``heating_coulomb``
/ ``heating_anomalous`` / ``heating_secondary`` / ``heating_terminal``). These
are bookkeeping only -- they re-add products the energy decrement already
formed, and the lumped bank the RHS consumes is untouched.

Non-local product transport (``product_transport``, WP-D)
---------------------------------------------------------

The two EVENT-PRODUCT channels above -- the mean secondary energy <W_sec> per
ionization, and the primary's terminal sub-threshold residual -- are banked in
their birth cell under ``product_transport="local"`` (the default, and the
historical behaviour). That is perfect local confinement, and at breakdown it
is the wrong limit: both products are BELOW every He inelastic threshold, so
their only loss channel is Coulomb slowing on the bulk, and at n_e ~ 1e10 that
costs ~1 eV per machine pass. They are near-collisionless along B exactly when
the model assumes they stop where they were born.

``product_transport="nonlocal"`` gives each product its own mini-CSDA walk
along B from its birth cell, depositing the same
``coulomb_stopping_eV_per_cm`` drag the primary feels (same ``coulomb_model``)
until it either thermalizes -- energy down to the local Maxwellian mean
``1.5*Te`` (floored at the module's own lnLambda temperature clamp, 0.1 eV),
below which the fast test-particle stopping formula has no meaning and the
electron is indistinguishable from the bulk -- or leaves an end of the domain,
where its remaining energy is booked to the END LEDGER
(``end_loss_low_erg_s`` / ``end_loss_high_erg_s``) and LEAVES the system.

Products and their directions:

- terminal residual: one walk from the stop cell in the PRIMARY's direction;
- secondaries: two half-weight walks (+z and -z) from each birth cell, the
  OPB emission being broadly isotropic -- a stated approximation.

Stated limitations of the walk: straight-line along B with no pitch-angle
diffusion; Coulomb slowing on the plasma is the ONLY loss channel (elastic
e-He transfers ~5 meV per collision, negligible); each birth cell's
secondaries walk at their flux-weighted MEAN energy, matching the module's
mean-energy treatment of the primary; a product is born at the near edge of
its birth cell along its own walk direction, so the walk traverses the whole
birth cell (cell-resolution granularity, as everywhere else in the module).

The end ledger ALSO books the transmitted primary's power
``Gamma_t * E_t`` -- computed by this module since B1 and never banked
anywhere. Under ``"nonlocal"`` the per-ray identity is therefore

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + end_loss_low + end_loss_high

with the transmitted primary INSIDE the end-loss channel (its size is
reported separately as ``end_loss_transmitted_erg_s``, so the historical
form with ``transmitted`` as its own term is recoverable by subtraction).
``transmitted_flux`` / ``transmitted_energy_eV`` keep their meaning and their
values. Under ``"local"`` the end-loss fields are identically zero, nothing
new is booked, and the identity stays exactly the historical

    Gamma0*E0 = heating + radiated + ionization_cost + anode_intercepted
                + transmitted

**v1 is ENERGY-ONLY routing.** Particle and charge bookkeeping is UNCHANGED:
``ionization_events`` / ``excitation_events``, the fluid ``n`` rows they feed,
and the circuit currents are identical in both modes. Only where the product
ENERGY lands moves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._cathode_solver import _c_log_ei
from ._cross import He_EII_cross_lkup, He_beam_excitation_channel_lkup

_ERG_PER_EV = 1.602176634e-12
_ME_CGS = 9.1093837015e-28  # electron mass [g]
_E4_CGS = (4.80320425e-10) ** 4  # e^4 [esu^4]
_OMEGA_PE_COEFF = 5.64e4  # omega_pe = 5.64e4 sqrt(n_e) [rad/s] (NRL)

HE_I_ION_EV = 24.587
HE_E_STOP_EV = 20.6158  # lowest inelastic threshold (2^1S)
HE_OPB_EBAR_EV = 15.8  # Opal-Peterson-Beaty shape parameter for He

# --- Non-local product transport (product_transport="nonlocal") -------------
# Thermalization floor of a walking product, as a multiple of the local Te.
# 3/2 is the mean energy of the local Maxwellian: at that energy the product
# is statistically indistinguishable from the bulk it is dragging against, and
# the fast test-particle stopping power (which assumes W >> Te) has stopped
# being a description of anything. NOT a tunable -- it is the bulk mean
# energy, not a fitted number, and it is deliberately not exposed as a config
# key.
_PRODUCT_FLOOR_TE_MULTIPLE = 1.5
# Absolute floor [eV], reusing the module's OWN lnLambda temperature clamp
# (``_c_log_ei(max(Te, 0.1), ne)`` in coulomb_stopping_eV_per_cm): below it the
# Coulomb logarithm is already being evaluated at a substitute temperature, so
# a walk there would be tracking a formula outside its own domain. Also the
# guard that keeps the walk finite when Te -> 0.
_PRODUCT_FLOOR_MIN_EV = 0.1
# Energy exponent p of each closure's stopping power, dE/dx = A(ne,Te) * W**p.
# Both closures are exact power laws in W -- lnLambda depends only on (ne, Te)
# -- which is what makes the product walk integrable in closed form below.
_COULOMB_STOPPING_EXPONENT = {"fast_electron": -1.0, "legacy_tau_ei": 0.5}


def _coulomb_stopping_coefficient(ne, Te, model):
    """Per-cell ``A`` in ``dE/dx = A * W**p`` [eV/cm], from the module's own
    ``coulomb_stopping_eV_per_cm``.

    Evaluated at ``W = 1 eV``, where ``W**p == 1`` for either exponent, so the
    coefficient carries the closure's constants, ``ne``, ``Te`` and lnLambda
    exactly as the primary's stopping power does -- the walk cannot drift away
    from the primary's drag law without ``coulomb_stopping_eV_per_cm`` itself
    changing shape (which the smoke suite checks).
    """
    return np.array(
        [
            coulomb_stopping_eV_per_cm(1.0, float(n), float(t), model=model)
            for n, t in zip(np.asarray(ne).ravel(), np.asarray(Te).ravel())
        ],
        dtype=float,
    )


def _walk_products_forward(W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q):
    """Coulomb slowing walk of per-cell product populations toward +index.

    ``W0_eV[s]`` / ``flux_per_s[s]`` are the birth energy and flux of the
    population born in cell ``s``; the walk proceeds s, s+1, ... Returns
    ``(deposited_eV_per_s, exit_eV_per_s)``: per-cell deposited power in
    ``flux*eV`` units and the scalar power carried out of the last cell.

    The integration is EXACT, not substepped. With ``dE/dx = A_j W**p`` held
    constant across a cell, ``u = W**q`` with ``q = 1 - p`` obeys
    ``du/dz = -q A_j``, so ``u`` falls linearly and a cumulative sum gives the
    energy at every cell exit in closed form. Deposition in a cell is the
    difference of the entry and exit energies, and the entry energy of a cell
    is REUSED as the exit energy of its predecessor (the identical float), so
    the per-ray sum telescopes to the birth energy exactly rather than to
    accumulated roundoff.
    """
    cells = int(dz_cm.size)
    dep = np.zeros(cells)
    W0_eV = np.asarray(W0_eV, dtype=float)
    flux_per_s = np.asarray(flux_per_s, dtype=float)
    active = np.flatnonzero((flux_per_s > 0.0) & (W0_eV > 0.0))
    if active.size == 0:
        return dep, 0.0
    W0v = W0_eV[active]
    fluxv = flux_per_s[active]
    index = np.arange(cells)
    reach = index[None, :] >= active[:, None]
    # u after LEAVING each cell, per birth cell (cumulative sum starts at the
    # birth cell, so cells behind it contribute nothing and read back u = W0**q).
    delta = q * coeff * dz_cm
    u_out = np.power(W0v, q)[:, None] - np.cumsum(
        np.where(reach, delta[None, :], 0.0), axis=1
    )
    W_out = np.power(np.maximum(u_out, 0.0), 1.0 / q)
    W_in = np.empty_like(W_out)
    W_in[:, 0] = W0v
    W_in[:, 1:] = W_out[:, :-1]
    # The birth cell's entry energy is the birth energy by definition; pin it
    # rather than trusting (W0**q)**(1/q) to round back onto W0.
    W_in[np.arange(active.size), active] = W0v
    # Thermalized where the cell-exit energy has fallen to the local floor.
    stop = reach & (W_out <= floor_eV[None, :])
    has_stop = stop.any(axis=1)
    j_stop = np.where(has_stop, np.argmax(stop, axis=1), cells)
    before = reach & (index[None, :] < j_stop[:, None])
    at = reach & (index[None, :] == j_stop[:, None])
    contrib = np.where(before, W_in - W_out, 0.0) + np.where(at, W_in, 0.0)
    dep += (fluxv[:, None] * contrib).sum(axis=0)
    exit_eV = float(
        np.sum(fluxv * np.where(has_stop, 0.0, W_out[:, -1]))
    )
    return dep, exit_eV


def _walk_products(W0_eV, flux_per_s, direction, coeff, dz_cm, floor_eV, q):
    """Direction-aware wrapper around ``_walk_products_forward``.

    Returns ``(deposited_eV_per_s, exit_eV_per_s)``; the exit power leaves the
    HIGH-index end for ``direction > 0`` and the LOW-index end otherwise.
    """
    if direction > 0:
        return _walk_products_forward(
            W0_eV, flux_per_s, coeff, dz_cm, floor_eV, q
        )
    dep, exit_eV = _walk_products_forward(
        W0_eV[::-1], flux_per_s[::-1], coeff[::-1], dz_cm[::-1],
        floor_eV[::-1], q,
    )
    return dep[::-1], exit_eV


def beam_speed_cm_s(E_eV: float) -> float:
    """Nonrelativistic primary speed [cm/s] at kinetic energy E_eV."""
    return math.sqrt(2.0 * E_eV * _ERG_PER_EV / _ME_CGS)


def he_mean_secondary_energy_eV(
    E_eV: float,
    I_ion_eV: float = HE_I_ION_EV,
    ebar_eV: float = HE_OPB_EBAR_EV,
) -> float:
    """Mean secondary-electron energy [eV] per He ionization at E_eV.

    OPB shape over W in [0, (E - I)/2]:
    <W> = Ebar * ln(1 + x^2) / (2 atan(x)), x = W_max / Ebar.
    """
    W_max = 0.5 * (E_eV - I_ion_eV)
    if W_max <= 0.0:
        return 0.0
    x = W_max / ebar_eV
    return ebar_eV * math.log1p(x * x) / (2.0 * math.atan(x))


def coulomb_stopping_eV_per_cm(
    E_eV: float, ne: float, Te: float, model: str = "fast_electron"
) -> float:
    """Coulomb energy-loss rate [eV/cm] of a primary at E_eV."""
    if ne <= 0.0 or E_eV <= 0.0:
        return 0.0
    lnL = _c_log_ei(max(Te, 0.1), ne)
    if model == "fast_electron":
        return 2.0 * math.pi * _E4_CGS * ne * lnL / (E_eV * _ERG_PER_EV) / _ERG_PER_EV
    if model == "legacy_tau_ei":
        tau_ei = 3.44e5 * Te**1.5 / ne / lnL
        return E_eV / (beam_speed_cm_s(E_eV) * tau_ei)
    raise ValueError(
        f"unknown coulomb_model {model!r}; "
        "expected 'fast_electron' or 'legacy_tau_ei'"
    )


def quasilinear_relaxation_length_cm(
    E_eV: float, ne: float, n_b: float
) -> float:
    """Quasilinear beam relaxation length [cm], l_QL = (ne/nb)(v_b/w_pe)ln(ne/nb).

    Valid for a weak beam (n_b << n_e); returns inf outside that domain
    (n_b >= n_e/10) rather than extrapolating the theory into the strong-beam
    regime, and inf for an absent beam or plasma.
    """
    if ne <= 0.0 or n_b <= 0.0 or n_b >= 0.1 * ne:
        return math.inf
    omega_pe = _OMEGA_PE_COEFF * math.sqrt(ne)
    ratio = ne / n_b
    return ratio * (beam_speed_cm_s(E_eV) / omega_pe) * math.log(ratio)


@dataclass(frozen=True)
class BeamDepositionResult:
    """Per-cell deposition of one beam ray; arrays have shape (cells,).

    ionization_events   : ionization events [1/s]
    excitation_events   : excitation events [1/s]
    plasma_heating_erg_s: Coulomb + anomalous drag + secondary-electron
                          energy + sub-threshold residual [erg/s]
    heating_coulomb_erg_s   : DIAGNOSTIC split of ``plasma_heating_erg_s`` --
                          the continuous Coulomb drag on plasma electrons
                          [erg/s]
    heating_anomalous_erg_s : DIAGNOSTIC split -- the anomalous (quasilinear
                          beam-plasma) drag [erg/s]; identically zero under
                          ``anomalous_model="none"``
    heating_secondary_erg_s : DIAGNOSTIC split -- the inelastic-EVENT thermal
                          residue, i.e. the mean secondary-electron energy
                          <W_sec> carried away per ionization ABOVE the
                          ``I_ion`` potential cost [erg/s]
    heating_terminal_erg_s  : DIAGNOSTIC split -- the primary's end-of-range
                          terminal dump: the sub-threshold residual banked
                          whole in the cell where E crosses ``E_stop`` [erg/s]

    The four ``heating_*`` arrays are pure BOOKKEEPING of products the energy
    decrement already forms; they are accumulated alongside, never in place
    of, ``plasma_heating_erg_s``, which keeps its exact historical value and
    is the only one the solver RHS consumes. Their sum reproduces
    ``plasma_heating_erg_s`` to floating-point associativity only (the lumped
    bank adds the three per-substep pieces before multiplying), so compare
    them with a relative tolerance, never with ``==``.

    radiated_erg_s      : excitation line radiation [erg/s]
    ionization_cost_erg_s: I_ion * ionization events [erg/s] (kept separate
                          to map onto the solver's beam_ionization_cost term)
    transmitted_flux    : primary flux leaving the far end [1/s] (reduced by
                          the anode-mesh interception, if any)
    transmitted_energy_eV: mean primary energy at exit [eV] (0 if absorbed)
    anode_intercepted_erg_s: energy the anode mesh intercepts at the anode-face
                          crossing [erg/s] (audit A15). This leaves the plasma
                          (booked to the electrode, NOT plasma_heating_erg_s);
                          0 when no interception is requested or the ray stops
                          before the anode face.
    E_entry_eV          : diagnostic: primary energy entering each cell [eV]
                          (0 for cells the ray never reaches)
    end_loss_low_erg_s  : END LEDGER, low-index end [erg/s]. Identically 0.0
                          under ``product_transport="local"``. Under
                          ``"nonlocal"`` it books (a) the remaining energy of
                          product walks that leave that end without
                          thermalizing and (b) the transmitted PRIMARY's power
                          ``Gamma_t * E_t`` when the ray exits there. This
                          energy LEAVES the system: like
                          ``anode_intercepted_erg_s`` it is NOT part of
                          ``plasma_heating_erg_s`` and must not enter any RHS
                          row.
    end_loss_high_erg_s : END LEDGER, high-index end [erg/s]; same content.
    end_loss_transmitted_erg_s: DIAGNOSTIC split -- how much of
                          ``end_loss_low + end_loss_high`` is the transmitted
                          primary rather than walked products. 0.0 under
                          ``"local"``.
    """

    ionization_events: np.ndarray
    excitation_events: np.ndarray
    plasma_heating_erg_s: np.ndarray
    radiated_erg_s: np.ndarray
    ionization_cost_erg_s: np.ndarray
    transmitted_flux: float
    transmitted_energy_eV: float
    anode_intercepted_erg_s: float
    E_entry_eV: np.ndarray
    heating_coulomb_erg_s: np.ndarray
    heating_anomalous_erg_s: np.ndarray
    heating_secondary_erg_s: np.ndarray
    heating_terminal_erg_s: np.ndarray
    end_loss_low_erg_s: float = 0.0
    end_loss_high_erg_s: float = 0.0
    end_loss_transmitted_erg_s: float = 0.0


def deposit_beam(
    E0_eV: float,
    Gamma0_per_s: float,
    nn: np.ndarray,
    ne: np.ndarray,
    Te: np.ndarray,
    launch: int,
    direction: int,
    dz_cm: np.ndarray,
    *,
    I_ion_eV: float = HE_I_ION_EV,
    E_stop_eV: float = HE_E_STOP_EV,
    coulomb_model: str = "fast_electron",
    anomalous_model: str = "none",
    beam_area_cm2: np.ndarray | float | None = None,
    max_energy_fraction_per_substep: float = 0.02,
    anode_cross_index: int | None = None,
    anode_eta: float = 0.0,
    product_transport: str = "local",
) -> BeamDepositionResult:
    """Deposit one monoenergetic beam ray through the column (He only).

    Parameters mirror plan §2: ``E0_eV`` is the accelerating sheath drop,
    ``Gamma0_per_s`` the accepted emitted electron flux, ``nn/ne/Te`` the
    per-cell column state, and the ray is ``(launch, direction, dz_cm)``
    with ``direction`` +1 (toward increasing index) or -1. Cells behind the
    launch point receive nothing.

    ``anomalous_model``: ``"none"`` (default) or ``"quasilinear"``
    (requires ``beam_area_cm2``, scalar or per-cell, to form n_b).

    **Anode-mesh interception (audit A15).** ``anode_cross_index`` is the first
    cell on the far (column) side of the anode face along the ray; when it is
    given with ``anode_eta`` in ``[0, 1)`` the mesh intercepts the solid
    fraction ``anode_eta`` of the flux STILL STREAMING when the ray reaches
    that face (i.e. the long-mean-free-path beam that survived the gap). The
    intercepted power ``anode_eta * gamma * E`` is booked to
    ``anode_intercepted_erg_s`` (it leaves the plasma, landing on the anode,
    NOT in ``plasma_heating_erg_s``), and the surviving flux is reduced to
    ``(1 - anode_eta) * gamma`` for all subsequent deposition and ionization.
    A ray that stops in the gap never reaches the face and intercepts nothing,
    so only the survived (bypass) fraction is removed -- consistent with the
    circuit's ``eta * beam_bypass_fraction``. Per-ray energy still closes to
    roundoff::

        Gamma0*E0 = heating + radiated + cost + anode_intercepted + transmitted

    Off (``anode_cross_index is None`` or ``anode_eta == 0``) the running flux
    is the constant ``Gamma0_per_s`` throughout, so every bank is byte-for-byte
    the historical result.

    **Product transport (WP-D).** ``product_transport`` is ``"local"``
    (default) or ``"nonlocal"``; see the module docstring for the physics. Off
    (the default, and the value the gap-transmission probe call sites leave
    untouched) not one branch below changes, so the result is byte-for-byte
    the historical one and ``end_loss_*`` are identically zero. On, the
    secondary and terminal-residual banks are withheld from their birth cells
    and re-deposited along mini-CSDA walks, with what escapes an end booked to
    the end ledger. Energy-only: ``ionization_events`` and
    ``excitation_events`` -- and therefore every particle and circuit row
    downstream -- are identical in both modes.
    """
    if product_transport not in ("local", "nonlocal"):
        raise ValueError(
            f"unknown product_transport {product_transport!r}; "
            "expected 'local' or 'nonlocal'"
        )
    if anode_eta != 0.0 and not (0.0 <= anode_eta < 1.0):
        raise ValueError(
            f"anode_eta must be in [0, 1) (got {anode_eta})"
        )
    nn = np.asarray(nn, dtype=float)
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    if anode_cross_index is not None:
        anode_cross_index = int(anode_cross_index)
        if not 0 <= anode_cross_index < cells:
            raise ValueError(
                "anode_cross_index must index a cell in [0, cells) "
                f"(got {anode_cross_index}, cells={cells})"
            )
    if nn.shape != (cells,) or ne.shape != (cells,) or Te.shape != (cells,):
        raise ValueError("nn, ne, Te, dz_cm must share one shape (cells,)")
    if direction not in (-1, 1):
        raise ValueError(f"direction must be +1 or -1 (got {direction})")
    if anomalous_model not in ("none", "quasilinear"):
        raise ValueError(
            f"unknown anomalous_model {anomalous_model!r}; "
            "expected 'none' or 'quasilinear'"
        )
    if anomalous_model == "quasilinear":
        if beam_area_cm2 is None:
            raise ValueError("anomalous_model='quasilinear' needs beam_area_cm2")
        area = np.broadcast_to(
            np.asarray(beam_area_cm2, dtype=float), (cells,)
        )
    frac = float(max_energy_fraction_per_substep)
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "max_energy_fraction_per_substep must be in (0, 1), got "
            f"{max_energy_fraction_per_substep}"
        )

    ionization_events = np.zeros(cells)
    excitation_events = np.zeros(cells)
    heating = np.zeros(cells)  # erg/s
    radiated = np.zeros(cells)
    ionization_cost = np.zeros(cells)
    E_entry = np.zeros(cells)
    # Diagnostic splits of `heating` (see BeamDepositionResult). Accumulated
    # from the SAME products the lumped bank uses; nothing here feeds the RHS.
    heat_coulomb = np.zeros(cells)
    heat_anomalous = np.zeros(cells)
    heat_secondary = np.zeros(cells)
    heat_terminal = np.zeros(cells)
    # --- Non-local product transport (WP-D) -----------------------------
    # Under "nonlocal" the secondary and terminal-residual banks are WITHHELD
    # from their birth cells and accumulated here, then walked after the ray
    # is done. Under "local" none of this is touched or allocated.
    walk_products = product_transport == "nonlocal"
    end_loss_low = 0.0
    end_loss_high = 0.0
    end_loss_transmitted = 0.0
    if walk_products:
        sec_flux = np.zeros(cells)  # secondary electrons born per cell [1/s]
        sec_power_eV = np.zeros(cells)  # their energy [eV/s]
        terminal_cell = -1
        terminal_flux = 0.0
        terminal_E = 0.0

    order = range(launch, cells) if direction > 0 else range(launch, -1, -1)
    E = float(E0_eV)
    absorbed = False
    # Running flux [1/s]. Constant Gamma0 unless the anode mesh intercepts part
    # of the surviving beam at its face (audit A15); every bank below multiplies
    # by this, so the off path is bit-for-bit the historical constant-flux result.
    gamma = float(Gamma0_per_s)
    anode_intercepted = 0.0  # erg/s booked to the anode, not the plasma
    intercept_active = anode_cross_index is not None and anode_eta > 0.0

    if E <= E_stop_eV:
        # Sub-threshold source: nothing inelastic can happen; the module's
        # domain is beam energies, so pass it through untouched. Under
        # "nonlocal" that pass-through IS a transmitted primary, so the end
        # ledger books it and the identity closes here too. (Defensive path:
        # the cathode wiring only launches rays with phi_c > I_ion > E_stop.)
        sub_power = float(Gamma0_per_s) * E * _ERG_PER_EV
        if walk_products and sub_power > 0.0:
            end_loss_transmitted = sub_power
            if direction > 0:
                end_loss_high = sub_power
            else:
                end_loss_low = sub_power
        return BeamDepositionResult(
            ionization_events=ionization_events,
            excitation_events=excitation_events,
            plasma_heating_erg_s=heating,
            radiated_erg_s=radiated,
            ionization_cost_erg_s=ionization_cost,
            transmitted_flux=float(Gamma0_per_s),
            transmitted_energy_eV=E,
            anode_intercepted_erg_s=0.0,
            E_entry_eV=E_entry,
            heating_coulomb_erg_s=heat_coulomb,
            heating_anomalous_erg_s=heat_anomalous,
            heating_secondary_erg_s=heat_secondary,
            heating_terminal_erg_s=heat_terminal,
            end_loss_low_erg_s=end_loss_low,
            end_loss_high_erg_s=end_loss_high,
            end_loss_transmitted_erg_s=end_loss_transmitted,
        )

    for cell in order:
        # Anode-mesh interception (A15): the ray reaches the anode face only if
        # it survived the gap (a stopped beam breaks out before this cell), so
        # removing eta of the flux HERE removes exactly the long-mfp/bypass beam.
        # Book the intercepted primaries' remaining energy to the anode and carry
        # the reduced flux downstream.
        if intercept_active and cell == anode_cross_index:
            anode_intercepted += anode_eta * gamma * E * _ERG_PER_EV
            gamma *= 1.0 - anode_eta
            intercept_active = False
        E_entry[cell] = E
        remaining = float(dz_cm[cell])
        nn_c = float(nn[cell])
        ne_c = float(ne[cell])
        Te_c = float(Te[cell])
        while remaining > 0.0:
            sigma_i = (
                He_EII_cross_lkup(E / I_ion_eV) if E > I_ion_eV else 0.0
            )
            # Table-interpolated manifold channel (see _cross docstring):
            # exact-node table, ~1e-6 relative interp error, ~100x cheaper
            # than the scalar sums this loop used to spend ~80% of total
            # step time in (2026-07-21).
            sigma_x, E_rad = He_beam_excitation_channel_lkup(E)
            W_sec = he_mean_secondary_energy_eV(E, I_ion_eV=I_ion_eV)
            # channel loss rates [eV/cm]
            L_pot = nn_c * sigma_i * I_ion_eV
            L_sec = nn_c * sigma_i * W_sec
            L_exc = nn_c * sigma_x * E_rad
            L_coul = coulomb_stopping_eV_per_cm(
                E, ne_c, Te_c, model=coulomb_model
            )
            L_anom = 0.0
            if anomalous_model == "quasilinear":
                n_b = gamma / (
                    float(area[cell]) * beam_speed_cm_s(E)
                )
                l_ql = quasilinear_relaxation_length_cm(E, ne_c, n_b)
                if math.isfinite(l_ql) and l_ql > 0.0:
                    L_anom = E / l_ql
            L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
            if L_tot <= 0.0:
                break  # vacuum cell: free streaming
            dz_sub = min(remaining, frac * E / L_tot)
            # Land exactly on E_stop rather than overshooting through it.
            if E - L_tot * dz_sub <= E_stop_eV:
                dz_sub = (E - E_stop_eV) / L_tot
            if dz_sub <= 0.0:
                # E sits at E_stop to roundoff: absorb the residual here.
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    heating[cell] += gamma * E * _ERG_PER_EV
                    heat_terminal[cell] += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
            # Bank each channel with the identical products the energy
            # decrement uses, so conservation closes to roundoff.
            d_pot = L_pot * dz_sub
            d_sec = L_sec * dz_sub
            d_exc = L_exc * dz_sub
            d_coul = L_coul * dz_sub
            d_anom = L_anom * dz_sub
            ionization_cost[cell] += gamma * d_pot * _ERG_PER_EV
            if walk_products:
                # Withhold the secondary bank from this cell; accumulate the
                # population (flux and energy) for the walks below. The flux
                # is the SAME product `ionization_events` uses, so the
                # particle rows are untouched by construction.
                heating[cell] += gamma * (d_coul + d_anom) * _ERG_PER_EV
                sec_flux[cell] += gamma * nn_c * sigma_i * dz_sub
                sec_power_eV[cell] += gamma * d_sec
            else:
                heating[cell] += gamma * (d_sec + d_coul + d_anom) * _ERG_PER_EV
                heat_secondary[cell] += gamma * d_sec * _ERG_PER_EV
            heat_coulomb[cell] += gamma * d_coul * _ERG_PER_EV
            heat_anomalous[cell] += gamma * d_anom * _ERG_PER_EV
            radiated[cell] += gamma * d_exc * _ERG_PER_EV
            ionization_events[cell] += gamma * nn_c * sigma_i * dz_sub
            excitation_events[cell] += gamma * nn_c * sigma_x * dz_sub
            E -= d_pot + d_sec + d_exc + d_coul + d_anom
            remaining -= dz_sub
            if E <= E_stop_eV:
                # Sub-threshold residual: the primary can only Coulomb-drag
                # from here; bank the remainder as local plasma heating
                # (plan B1's stated closure) and end the ray. Under "nonlocal"
                # that same residual is instead walked from this cell.
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    heating[cell] += gamma * E * _ERG_PER_EV
                    heat_terminal[cell] += gamma * E * _ERG_PER_EV
                E = 0.0
                absorbed = True
                break
        if absorbed:
            break

    if walk_products:
        # --- Product walks (WP-D) ---------------------------------------
        # One stopping-power coefficient per cell, taken from the module's own
        # coulomb_stopping_eV_per_cm, plus the closure's energy exponent; see
        # _walk_products_forward for the closed-form integration.
        q = 1.0 - _COULOMB_STOPPING_EXPONENT[coulomb_model]
        coeff = _coulomb_stopping_coefficient(ne, Te, coulomb_model)
        floor_eV = np.maximum(
            _PRODUCT_FLOOR_TE_MULTIPLE * Te, _PRODUCT_FLOOR_MIN_EV
        )

        def _bank_walk(W0, flux, walk_direction, split):
            """Walk one product population and book its deposit and escape."""
            nonlocal end_loss_low, end_loss_high
            dep_eV, exit_eV = _walk_products(
                W0, flux, walk_direction, coeff, dz_cm, floor_eV, q
            )
            dep_erg = dep_eV * _ERG_PER_EV
            heating[:] += dep_erg
            split[:] += dep_erg
            if walk_direction > 0:
                end_loss_high += exit_eV * _ERG_PER_EV
            else:
                end_loss_low += exit_eV * _ERG_PER_EV

        if np.any(sec_flux > 0.0):
            # Flux-weighted mean secondary energy per birth cell (the module
            # carries mean energies, not distributions -- stated limitation),
            # emitted 50/50 along +z and -z (OPB emission is broadly
            # isotropic -- stated approximation).
            W_sec_cell = np.zeros(cells)
            born = sec_flux > 0.0
            W_sec_cell[born] = sec_power_eV[born] / sec_flux[born]
            half = 0.5 * sec_flux
            for walk_direction in (1, -1):
                _bank_walk(W_sec_cell, half, walk_direction, heat_secondary)
        if terminal_flux > 0.0 and terminal_E > 0.0:
            # The terminal residual keeps the primary's direction.
            term_flux = np.zeros(cells)
            term_W = np.zeros(cells)
            term_flux[terminal_cell] = terminal_flux
            term_W[terminal_cell] = terminal_E
            _bank_walk(term_W, term_flux, direction, heat_terminal)
        if not absorbed and gamma > 0.0 and E > 0.0:
            # The transmitted primary: computed since B1, never banked. It
            # leaves through the end the ray was heading for.
            end_loss_transmitted = gamma * E * _ERG_PER_EV
            if direction > 0:
                end_loss_high += end_loss_transmitted
            else:
                end_loss_low += end_loss_transmitted

    return BeamDepositionResult(
        ionization_events=ionization_events,
        excitation_events=excitation_events,
        plasma_heating_erg_s=heating,
        radiated_erg_s=radiated,
        ionization_cost_erg_s=ionization_cost,
        transmitted_flux=0.0 if absorbed else gamma,
        transmitted_energy_eV=0.0 if absorbed else E,
        anode_intercepted_erg_s=anode_intercepted,
        E_entry_eV=E_entry,
        heating_coulomb_erg_s=heat_coulomb,
        heating_anomalous_erg_s=heat_anomalous,
        heating_secondary_erg_s=heat_secondary,
        heating_terminal_erg_s=heat_terminal,
        end_loss_low_erg_s=end_loss_low,
        end_loss_high_erg_s=end_loss_high,
        end_loss_transmitted_erg_s=end_loss_transmitted,
    )
