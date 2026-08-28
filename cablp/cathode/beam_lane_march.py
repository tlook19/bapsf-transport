"""Many independent CSDA walkers marched together over ONE shared column.

``deposit_beam``'s tail-ionization closure gives every QL tail population its
own CSDA march: one recursive ``deposit_beam`` call per (plateau group, birth
cell, direction), and a second per reflection. Those legs share the whole
column -- the same windowed ``nn``/``ne``/``Te``/``dz_cm``, the same
thresholds, the same substep fraction -- and differ only in where they start,
which way they go, at what energy, and with how much flux. They never
interact.

This module marches all of them AT ONCE. Each walker is a LANE with its own
energy, cell, remaining path length and per-cell banks; a round advances every
active lane by ONE of its OWN substeps. **The substep sequence is not shared:**
``dz_sub = min(remaining, frac*E/L_tot)`` and the exact landing on ``E_stop``
are evaluated per lane, from that lane's own energy, exactly as the scalar
march evaluates them. What is shared is the arithmetic's dispatch -- one numpy
call per quantity per round instead of one Python call per quantity per lane
per substep.

Bit-exactness is by CONSTRUCTION, not by tolerance:

* every expression below is the scalar march's expression with its
  associativity preserved (``(nn*sigma)*I_ion``, not ``nn*(sigma*I_ion)``);
* the interpolated cross sections go through
  :func:`cablp.numerics.interp.interp_array_fused`, which reconstructs
  ``math.fma``'s single rounding, so the lookups agree with the scalar lookups
  at raw uint64 rather than to 1 ULP (numpy's own lerp is unfused on the
  x86-64 baseline -- that is the whole reason the fused scalar lerp exists);
* every transcendental used here (``log``, ``exp``, ``log1p``, ``arctan``,
  ``sqrt``) is IEEE-754 float64 and numpy dispatches it to the same libm the
  ``math`` module does;
* a lane's per-cell accumulators are advanced by the same additions in the
  same order the scalar march performs, and each accumulator is non-negative,
  so masking an inactive lane's increment to ``0.0`` is an exact no-op;
* the per-cell banks are written once per (lane, cell), which each lane visits
  at most once, so the caller can replay the legs' bank order verbatim.

Scope is deliberately narrow. This marches the configuration the tail-walk
legs are launched in and NOTHING else: no anomalous channel, no anode mesh, no
withheld product or tail banks, ``product_transport="local"``. Anything else
is refused at :func:`lane_march` rather than approximated, and the caller
keeps the scalar march for it.
"""

import math

import numpy as np

from ..atomic.cross_sections import (
    _HE_LOG_EPS,
    _HE_LOG_SIGMA,
    _he_beam_excitation_table,
)
from ..cathode.circuit import _c_log_ei
from ..numerics.interp import (
    _interp_array_unchecked_multi,
    check_fma_domain,
    check_interp_table,
)

__all__ = [
    "LaneMarchResult",
    "check_constants",
    "lane_march",
    "lane_march_energy_ceiling_eV",
    "LANE_MARCH_MODELS",
]

_ERG_PER_EV = 1.602176634e-12
_ME_CGS = 9.1093837015e-28
_E4_CGS = (4.80320425e-10) ** 4

#: Coulomb closures this module reproduces. A model outside it is refused: the
#: stopping power's energy dependence is written out below per closure, and a
#: closure that is not written out cannot be marched here.
LANE_MARCH_MODELS = ("fast_electron", "legacy_tau_ei")

# The He Opal-Peterson-Beaty shape parameter, spelled here rather than
# imported so this module does not import its own caller.
_HE_OPB_EBAR_EV = 15.8

# One-time validation of the two fixed tables, so the per-round lookups can
# skip it. Both are module constants of ``atomic.cross_sections``; the
# excitation table is built lazily and is rebuilt only for a different
# ``n_max``, which nothing here asks for.
_EII_LOG_EPS = np.ascontiguousarray(_HE_LOG_EPS, dtype=float)
_EII_LOG_SIGMA = np.ascontiguousarray(_HE_LOG_SIGMA, dtype=float)
check_interp_table(_EII_LOG_EPS, _EII_LOG_SIGMA)
_EII_LEFT = float(_EII_LOG_SIGMA[0])
_EII_RIGHT = float(_EII_LOG_SIGMA[-1])

_EXC_TABLE_SRC = None
_EXC_GRID = _EXC_SIGMA = _EXC_SIGMA_E = None


def _exc_table():
    """The validated excitation table, rebuilt only if its source is replaced."""
    global _EXC_TABLE_SRC, _EXC_GRID, _EXC_SIGMA, _EXC_SIGMA_E
    src = _he_beam_excitation_table(20)
    if _EXC_TABLE_SRC is not src:
        grid = np.ascontiguousarray(src[1], dtype=float)
        sig = np.ascontiguousarray(src[2], dtype=float)
        sig_e = np.ascontiguousarray(src[3], dtype=float)
        check_interp_table(grid, sig)
        check_interp_table(grid, sig_e)
        _EXC_GRID, _EXC_SIGMA, _EXC_SIGMA_E = grid, sig, sig_e
        _EXC_TABLE_SRC = src
    return _EXC_GRID, _EXC_SIGMA, _EXC_SIGMA_E


def check_constants(erg_per_ev, me_cgs, e4_cgs, opb_ebar_ev):
    """Raise unless the caller's physical constants are this module's, bit for bit.

    The lane march needs four constants that ``beam_deposition`` also spells
    out, and it cannot import them from there because that module imports this
    one. A re-spelling can drift, and a drifted constant would move the march
    silently, so the importer asserts the agreement AT IMPORT -- the same
    question the compiled kernel module answers with ``check_constants_beam``.
    """
    mismatched = [
        name
        for name, mine, theirs in (
            ("_ERG_PER_EV", _ERG_PER_EV, erg_per_ev),
            ("_ME_CGS", _ME_CGS, me_cgs),
            ("_E4_CGS", _E4_CGS, e4_cgs),
            ("HE_OPB_EBAR_EV", _HE_OPB_EBAR_EV, opb_ebar_ev),
        )
        if mine != float(theirs)
    ]
    if mismatched:
        raise ValueError(
            "cablp.cathode.beam_lane_march disagrees with its caller on "
            f"{', '.join(mismatched)}. The lane march reproduces the scalar "
            "CSDA march bit for bit only while the two spell the same "
            "constants; fix the divergence rather than tolerating it"
        )


def lane_march_energy_ceiling_eV():
    """Launch energy [eV] at and above which :func:`lane_march` refuses.

    The top of the tabulated He excitation manifold. At or above it the scalar
    lookup falls back to the exact sum over the singlet manifold, which the
    vectorised lookup does not reproduce; a caller tests against this before
    batching, rather than discovering the refusal as an exception.
    """
    return float(_exc_table()[0][-1])


class LaneMarchResult:
    """Per-lane outcome of :func:`lane_march`.

    Attributes
    ----------
    ionization_events, excitation_events : ndarray, shape (lanes, cells)
        Events per second banked by each lane in each cell.
    plasma_heating_erg_s, radiated_erg_s, ionization_cost_erg_s : ndarray, shape (lanes, cells)
        The three energy banks [erg/s], per lane and cell.
    transmitted_flux : ndarray, shape (lanes,)
        Flux [1/s] leaving the domain, zero for an absorbed lane.
    transmitted_energy_eV : ndarray, shape (lanes,)
        Energy [eV] it leaves with, zero for an absorbed lane.
    absorbed : ndarray of bool, shape (lanes,)
        True where the lane's energy crossed ``E_stop_eV`` inside the domain.
    substeps : ndarray of int, shape (lanes,)
        CSDA substep iterations the lane entered -- the same count the scalar
        march's per-substep cross-section lookup reports.
    """

    __slots__ = (
        "ionization_events",
        "excitation_events",
        "plasma_heating_erg_s",
        "radiated_erg_s",
        "ionization_cost_erg_s",
        "transmitted_flux",
        "transmitted_energy_eV",
        "absorbed",
        "substeps",
    )

    def __init__(self, **fields):
        for name in self.__slots__:
            setattr(self, name, fields[name])


def lane_march(
    E0_eV,
    gamma0_per_s,
    launch,
    direction,
    nn,
    ne,
    Te,
    dz_cm,
    *,
    I_ion_eV,
    E_stop_eV,
    coulomb_model,
    max_energy_fraction_per_substep,
):
    """March ``lanes`` independent CSDA walkers through one shared column.

    Parameters
    ----------
    E0_eV, gamma0_per_s : ndarray, shape (lanes,)
        Launch energy [eV] and flux [1/s] of each lane. Both must be finite
        and strictly positive.
    launch : ndarray of int, shape (lanes,)
        Launch cell of each lane, in ``[0, cells)``.
    direction : ndarray of int, shape (lanes,)
        ``+1`` (toward increasing index) or ``-1``, per lane.
    nn, ne, Te, dz_cm : ndarray, shape (cells,)
        The shared column: neutral density [cm^-3], electron density [cm^-3],
        electron temperature [eV] and cell length [cm].
    I_ion_eV : float
        Ionization potential [eV]; must be finite and > 0. Threaded from the
        caller -- this module has no default for it.
    E_stop_eV : float
        Lowest inelastic threshold [eV]; must be finite and > 0. A lane at or
        below it at launch is transmitted untouched.
    coulomb_model : str
        One of :data:`LANE_MARCH_MODELS`.
    max_energy_fraction_per_substep : float
        The substep fraction, in ``(0, 1)``.

    Returns
    -------
    LaneMarchResult

    Raises
    ------
    ValueError
        On any argument outside the domain above. Nothing is defaulted or
        approximated: the caller keeps a scalar march for the cases this one
        refuses.
    """
    if coulomb_model not in LANE_MARCH_MODELS:
        raise ValueError(
            f"lane_march does not implement coulomb_model {coulomb_model!r}; "
            f"expected one of {sorted(LANE_MARCH_MODELS)}"
        )
    I_ion_eV = float(I_ion_eV)
    E_stop_eV = float(E_stop_eV)
    frac = float(max_energy_fraction_per_substep)
    if not math.isfinite(I_ion_eV) or I_ion_eV <= 0.0:
        raise ValueError(f"lane_march needs a finite I_ion_eV > 0 (got {I_ion_eV})")
    if not math.isfinite(E_stop_eV) or E_stop_eV <= 0.0:
        raise ValueError(f"lane_march needs a finite E_stop_eV > 0 (got {E_stop_eV})")
    if not 0.0 < frac < 1.0:
        raise ValueError(
            "lane_march needs max_energy_fraction_per_substep in (0, 1), got "
            f"{frac}"
        )
    nn = np.asarray(nn, dtype=float)
    ne = np.asarray(ne, dtype=float)
    Te = np.asarray(Te, dtype=float)
    dz_cm = np.asarray(dz_cm, dtype=float)
    cells = dz_cm.size
    if nn.shape != (cells,) or ne.shape != (cells,) or Te.shape != (cells,):
        raise ValueError("nn, ne, Te, dz_cm must share one shape (cells,)")
    for name, arr in (("nn", nn), ("ne", ne), ("Te", Te), ("dz_cm", dz_cm)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"lane_march needs a finite {name}")
    E = np.array(E0_eV, dtype=float)
    gamma = np.array(gamma0_per_s, dtype=float)
    cell = np.array(launch, dtype=np.intp)
    step_dir = np.array(direction, dtype=np.intp)
    lanes = E.size
    if not (gamma.shape == cell.shape == step_dir.shape == (lanes,)):
        raise ValueError(
            "E0_eV, gamma0_per_s, launch and direction must share one shape "
            "(lanes,)"
        )
    if lanes and not (np.all(np.isfinite(E)) and np.all(np.isfinite(gamma))):
        raise ValueError("lane_march needs finite E0_eV and gamma0_per_s")
    if lanes and not (np.all(E > 0.0) and np.all(gamma > 0.0)):
        raise ValueError("lane_march needs E0_eV > 0 and gamma0_per_s > 0")
    if lanes and not np.all((cell >= 0) & (cell < cells)):
        raise ValueError(
            f"lane_march launch cells must lie in [0, {cells})"
        )
    if lanes and not np.all((step_dir == 1) | (step_dir == -1)):
        raise ValueError("lane_march direction entries must be +1 or -1")
    check_fma_domain(E, "E0_eV")

    exc_grid, exc_sigma, exc_sigma_E = _exc_table()
    exc_lo = float(exc_grid[0])
    exc_hi = float(exc_grid[-1])

    ion_events = np.zeros((lanes, cells))
    exc_events = np.zeros((lanes, cells))
    heating = np.zeros((lanes, cells))
    radiated = np.zeros((lanes, cells))
    ion_cost = np.zeros((lanes, cells))
    out_flux = np.zeros(lanes)
    out_E = np.zeros(lanes)
    absorbed = np.zeros(lanes, dtype=bool)
    substeps = np.zeros(lanes, dtype=np.int64)
    if lanes == 0:
        return LaneMarchResult(
            ionization_events=ion_events,
            excitation_events=exc_events,
            plasma_heating_erg_s=heating,
            radiated_erg_s=radiated,
            ionization_cost_erg_s=ion_cost,
            transmitted_flux=out_flux,
            transmitted_energy_eV=out_E,
            absorbed=absorbed,
            substeps=substeps,
        )

    # --- per-cell constants -------------------------------------------------
    # ``coulomb_stopping_eV_per_cm`` re-evaluates lnLambda on every substep
    # from (Te, ne) alone, so its value is a property of the CELL; hoisting it
    # gives the identical float. Cells with ne <= 0 return 0.0 there and are
    # masked here for the same reason.
    ne_live = ne > 0.0
    coul_num = np.zeros(cells)
    coul_tau = np.ones(cells)
    for c in range(cells):
        if not ne_live[c]:
            continue
        lnL = _c_log_ei(max(float(Te[c]), 0.1), float(ne[c]))
        if coulomb_model == "fast_electron":
            coul_num[c] = 2.0 * math.pi * _E4_CGS * float(ne[c]) * lnL
        else:
            coul_tau[c] = (
                3.44e5 * float(Te[c]) ** 1.5 / float(ne[c]) / lnL
            )
    fast = coulomb_model == "fast_electron"

    # --- lane state ---------------------------------------------------------
    idx = np.arange(lanes)
    # A lane launched at or below the stop threshold is the scalar march's
    # sub-threshold short circuit: transmitted untouched, no bank written.
    live = E > E_stop_eV
    out_flux[~live] = gamma[~live]
    out_E[~live] = E[~live]
    idx = idx[live]
    E = E[live]
    gamma = gamma[live]
    cell = cell[live]
    step_dir = step_dir[live]
    remaining = dz_cm[cell].copy()
    acc_ion = np.zeros(idx.size)
    acc_exc = np.zeros(idx.size)
    acc_heat = np.zeros(idx.size)
    acc_rad = np.zeros(idx.size)
    acc_cost = np.zeros(idx.size)

    # A launch cell of zero (or negative) length has no substep to take; the
    # scalar march flushes an all-zero bank and moves on.
    if idx.size:
        (idx, E, gamma, cell, step_dir, remaining, acc_ion, acc_exc, acc_heat,
         acc_rad, acc_cost) = _skip_empty_cells(
            idx, E, gamma, cell, step_dir, remaining, acc_ion, acc_exc,
            acc_heat, acc_rad, acc_cost, dz_cm, cells, out_flux, out_E,
        )

    while idx.size:
        if not np.all(np.isfinite(E)):
            raise ValueError(
                "lane_march produced a non-finite lane energy; the vectorised "
                "cross-section lookup has no NaN passthrough and refuses "
                "rather than diverging from the scalar march"
            )
        substeps[idx] += 1
        nn_c = nn[cell]
        if np.any(E >= exc_hi):
            raise ValueError(
                "lane_march reached the He excitation table ceiling "
                f"({exc_hi} eV), where the scalar lookup falls back to the "
                "exact manifold sum; that fallback is not reproduced here"
            )
        # --- the three table lookups, in one fused-multiply-add -------------
        # The ionization cross section on the log-log EII table, and the
        # summed singlet excitation manifold's sigma and sigma*E_rad on one
        # energy grid. Batched only to save numpy dispatch; each is the value
        # the scalar lookup would return.
        eps = E / I_ion_eV
        open_i = E > I_ion_eV
        log_eps = np.log(np.where(open_i, eps, 1.0))
        in_band = E > exc_lo
        E_probe = np.where(in_band, E, exc_hi)
        eii_log, s_raw, sE_raw = _interp_array_unchecked_multi(
            (
                (log_eps, _EII_LOG_EPS, _EII_LOG_SIGMA, _EII_LEFT, _EII_RIGHT),
                (E_probe, exc_grid, exc_sigma, None, None),
                (E_probe, exc_grid, exc_sigma_E, None, None),
            )
        )
        sigma_i = np.where(open_i, np.exp(eii_log), 0.0)
        live_x = in_band & (s_raw > 0.0)
        sigma_x = np.where(live_x, s_raw, 0.0)
        E_rad = np.where(live_x, sE_raw / np.where(live_x, s_raw, 1.0), 0.0)
        # --- mean secondary energy (OPB) ------------------------------------
        W_max = 0.5 * (E - I_ion_eV)
        open_w = W_max > 0.0
        xx = np.where(open_w, W_max, 1.0) / _HE_OPB_EBAR_EV
        W_sec = np.where(
            open_w,
            (_HE_OPB_EBAR_EV * np.log1p(xx * xx)) / (2.0 * np.arctan(xx)),
            0.0,
        )
        # --- Coulomb drag on the bulk ---------------------------------------
        cell_live = ne_live[cell]
        if fast:
            L_coul = np.where(
                cell_live, (coul_num[cell] / (E * _ERG_PER_EV)) / _ERG_PER_EV,
                0.0,
            )
        else:
            speed = np.sqrt(((2.0 * E) * _ERG_PER_EV) / _ME_CGS)
            L_coul = np.where(
                cell_live, E / (speed * coul_tau[cell]), 0.0
            )
        # --- the substep ------------------------------------------------------
        L_pot = (nn_c * sigma_i) * I_ion_eV
        L_sec = (nn_c * sigma_i) * W_sec
        L_exc = (nn_c * sigma_x) * E_rad
        L_tot = (((L_pot + L_sec) + L_exc) + L_coul)
        vacuum = L_tot <= 0.0
        L_safe = np.where(vacuum, 1.0, L_tot)
        dz_sub = np.minimum(remaining, (frac * E) / L_safe)
        land = (E - (L_tot * dz_sub)) <= E_stop_eV
        dz_sub = np.where(land, (E - E_stop_eV) / L_safe, dz_sub)
        terminal = (~vacuum) & (dz_sub <= 0.0)
        stepping = (~vacuum) & (~terminal)
        d_pot = np.where(stepping, L_pot * dz_sub, 0.0)
        d_sec = np.where(stepping, L_sec * dz_sub, 0.0)
        d_exc = np.where(stepping, L_exc * dz_sub, 0.0)
        d_coul = np.where(stepping, L_coul * dz_sub, 0.0)
        # Banked with the identical products the energy decrement uses. An
        # inactive lane adds 0.0, which is exact on a non-negative accumulator.
        acc_cost += (gamma * d_pot) * _ERG_PER_EV
        acc_heat += (gamma * (d_sec + d_coul)) * _ERG_PER_EV
        acc_rad += (gamma * d_exc) * _ERG_PER_EV
        acc_ion += np.where(
            stepping, ((gamma * nn_c) * sigma_i) * dz_sub, 0.0
        )
        acc_exc += np.where(
            stepping, ((gamma * nn_c) * sigma_x) * dz_sub, 0.0
        )
        E = np.where(stepping, E - (((d_pot + d_sec) + d_exc) + d_coul), E)
        remaining = np.where(stepping, remaining - dz_sub, remaining)
        stopped = stepping & (E <= E_stop_eV)
        done = terminal | stopped
        # The sub-threshold residual, banked in the crossing cell -- the last
        # addition this lane makes to its heating accumulator in this cell.
        acc_heat += np.where(done, (gamma * E) * _ERG_PER_EV, 0.0)
        E = np.where(done, 0.0, E)
        leaving = done | vacuum | (remaining <= 0.0)
        if np.any(leaving):
            rows = idx[leaving]
            cols = cell[leaving]
            ion_events[rows, cols] += acc_ion[leaving]
            exc_events[rows, cols] += acc_exc[leaving]
            heating[rows, cols] += acc_heat[leaving]
            radiated[rows, cols] += acc_rad[leaving]
            ion_cost[rows, cols] += acc_cost[leaving]
            # Compaction is a copy of every lane array, so it runs only when
            # a lane has actually finished; a round where lanes merely cross a
            # cell boundary keeps the arrays it has.
            advance = leaving
            if np.any(done):
                absorbed[rows[done[leaving]]] = True
                keep = ~done
                (idx, E, gamma, cell, step_dir, remaining, acc_ion, acc_exc,
                 acc_heat, acc_rad, acc_cost, advance) = (
                    idx[keep], E[keep], gamma[keep], cell[keep],
                    step_dir[keep], remaining[keep], acc_ion[keep],
                    acc_exc[keep], acc_heat[keep], acc_rad[keep],
                    acc_cost[keep], leaving[keep],
                )
            if np.any(advance):
                acc_ion[advance] = 0.0
                acc_exc[advance] = 0.0
                acc_heat[advance] = 0.0
                acc_rad[advance] = 0.0
                acc_cost[advance] = 0.0
                cell = cell.copy()
                cell[advance] += step_dir[advance]
                remaining = remaining.copy()
                gone = advance & ((cell < 0) | (cell >= cells))
                if np.any(gone):
                    out_flux[idx[gone]] = gamma[gone]
                    out_E[idx[gone]] = E[gone]
                    stay = ~gone
                    (idx, E, gamma, cell, step_dir, remaining, acc_ion,
                     acc_exc, acc_heat, acc_rad, acc_cost, advance) = (
                        idx[stay], E[stay], gamma[stay], cell[stay],
                        step_dir[stay], remaining[stay], acc_ion[stay],
                        acc_exc[stay], acc_heat[stay], acc_rad[stay],
                        acc_cost[stay], advance[stay],
                    )
                if idx.size and np.any(advance):
                    remaining[advance] = dz_cm[cell[advance]]
                    (idx, E, gamma, cell, step_dir, remaining, acc_ion,
                     acc_exc, acc_heat, acc_rad, acc_cost) = (
                        _skip_empty_cells(
                            idx, E, gamma, cell, step_dir, remaining, acc_ion,
                            acc_exc, acc_heat, acc_rad, acc_cost, dz_cm, cells,
                            out_flux, out_E,
                        )
                    )

    return LaneMarchResult(
        ionization_events=ion_events,
        excitation_events=exc_events,
        plasma_heating_erg_s=heating,
        radiated_erg_s=radiated,
        ionization_cost_erg_s=ion_cost,
        transmitted_flux=out_flux,
        transmitted_energy_eV=out_E,
        absorbed=absorbed,
        substeps=substeps,
    )


def _skip_empty_cells(idx, E, gamma, cell, step_dir, remaining, acc_ion,
                      acc_exc, acc_heat, acc_rad, acc_cost, dz_cm, cells,
                      out_flux, out_E):
    """Walk lanes past cells of non-positive length, banking nothing.

    ``while remaining > 0.0`` is false on entry to such a cell, so the scalar
    march flushes an all-zero bank and steps to the next one without entering
    a substep. Reproduced here so those cells cost a lane no substep either.
    """
    while True:
        empty = remaining <= 0.0
        if not np.any(empty):
            return (idx, E, gamma, cell, step_dir, remaining, acc_ion,
                    acc_exc, acc_heat, acc_rad, acc_cost)
        cell = cell.copy()
        cell[empty] += step_dir[empty]
        inside = (cell >= 0) & (cell < cells)
        gone = empty & ~inside
        if np.any(gone):
            out_flux[idx[gone]] = gamma[gone]
            out_E[idx[gone]] = E[gone]
        stay = ~gone
        idx, E, gamma, cell, step_dir = (
            idx[stay], E[stay], gamma[stay], cell[stay], step_dir[stay]
        )
        remaining, acc_ion, acc_exc, acc_heat, acc_rad, acc_cost = (
            remaining[stay], acc_ion[stay], acc_exc[stay], acc_heat[stay],
            acc_rad[stay], acc_cost[stay]
        )
        empty = empty[stay]
        if not idx.size:
            return (idx, E, gamma, cell, step_dir, remaining, acc_ion,
                    acc_exc, acc_heat, acc_rad, acc_cost)
        remaining = remaining.copy()
        remaining[empty] = dz_cm[cell[empty]]
