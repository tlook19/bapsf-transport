"""OPEN-ADAS adf11 helium rate coefficients.

Parses the iso-nuclear master files in ``cablp/vars/adas`` (see the README
there for provenance and conventions) and exposes vectorized (n_e, T_e)
interpolators for the generalized collisional-radiative coefficients the
sim1d ``atomic_rate_model = "adas"`` path consumes.

All interpolation is bilinear in (log10 n_e, log10 T_e) on the file's own
grid, clamped to the grid edges outside it (the grid spans 5e7-2e15 cm^-3
and 0.2-1.5e4 eV, so clamping only engages in regimes where the physics the
tables carry is either frozen out or unreachable for LAPD).

Units follow the historical fit helpers so call sites are symmetric:
ionization/recombination coefficients in cm^3/s, radiated-power
coefficients in eV cm^3/s (the file's W cm^3 divided by the electron
charge), to be multiplied by the appropriate density product and, for
energy, ``ev_to_erg``.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np

from ..vars._cons import qe_SI

ADAS_DIR = Path(__file__).resolve().parent.parent / "vars" / "adas"


def read_adf11(path):
    """Parse an unresolved adf11 file.

    Returns ``(log10_ne, log10_te, stages)`` where ``stages`` maps the stage
    index z1 to an ``(nte, ndens)`` array of log10 coefficients (density
    varying fastest in the file, hence columns here).
    """
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("/")[0].split()
    ndens, nte = int(header[1]), int(header[2])
    z1min, z1max = int(header[3]), int(header[4])

    values = []
    for line in lines[1:]:
        if line.startswith("C"):
            break  # trailing comment section
        stripped = line.strip()
        if stripped and set(stripped) <= {"-"}:
            values.append(None)  # block delimiter
            continue
        if "/" in line or "=" in line:
            values.append(None)  # delimiter with stage metadata
            continue
        for token in line.split():
            values.append(float(token))
    while values and values[0] is None:
        values.pop(0)

    log_ne = np.array(values[:ndens], dtype=float)
    log_te = np.array(values[ndens : ndens + nte], dtype=float)
    rest = values[ndens + nte :]

    stages = {}
    idx = 0
    for z1 in range(z1min, z1max + 1):
        while idx < len(rest) and rest[idx] is None:
            idx += 1
        block = rest[idx : idx + ndens * nte]
        if len(block) != ndens * nte or any(v is None for v in block):
            raise ValueError(f"{path}: stage {z1} block is malformed")
        stages[z1] = np.array(block, dtype=float).reshape(nte, ndens)
        idx += ndens * nte
    return log_ne, log_te, stages


@lru_cache(maxsize=None)
def _load(name):
    log_ne, log_te, stages = read_adf11(ADAS_DIR / name)
    if not (np.all(np.diff(log_ne) > 0) and np.all(np.diff(log_te) > 0)):
        raise ValueError(f"{name}: adf11 axes are not strictly increasing")
    return log_ne, log_te, stages


def _interp_coords(log_x_grid, log_y_grid, log_x, log_y):
    """Return clamped bilinear indices and weights for the shared grid."""
    x = np.clip(log_x, log_x_grid[0], log_x_grid[-1])
    y = np.clip(log_y, log_y_grid[0], log_y_grid[-1])
    ix = np.clip(np.searchsorted(log_x_grid, x) - 1, 0, log_x_grid.size - 2)
    iy = np.clip(np.searchsorted(log_y_grid, y) - 1, 0, log_y_grid.size - 2)
    fx = (x - log_x_grid[ix]) / (log_x_grid[ix + 1] - log_x_grid[ix])
    fy = (y - log_y_grid[iy]) / (log_y_grid[iy + 1] - log_y_grid[iy])
    return ix, iy, fx, fy


def _interp_blend(table, ix, iy, fx, fy):
    """Blend one table at precomputed bilinear coordinates."""
    c00 = table[iy, ix]
    c01 = table[iy, ix + 1]
    c10 = table[iy + 1, ix]
    c11 = table[iy + 1, ix + 1]
    return (
        c00 * (1.0 - fy) * (1.0 - fx)
        + c01 * (1.0 - fy) * fx
        + c10 * fy * (1.0 - fx)
        + c11 * fy * fx
    )


def _interp_log_bilinear(log_x_grid, log_y_grid, table, log_x, log_y):
    """Vectorized bilinear interpolation of ``table[y, x]`` in log space.

    Coordinates are clamped to the grid, so evaluation outside it returns
    the nearest-edge value rather than an extrapolation.
    """
    ix, iy, fx, fy = _interp_coords(log_x_grid, log_y_grid, log_x, log_y)
    return _interp_blend(table, ix, iy, fx, fy)


def _evaluate(name, z1, ne_cm3, Te_eV):
    log_ne_grid, log_te_grid, stages = _load(name)
    ne = np.asarray(ne_cm3, dtype=float)
    Te = np.asarray(Te_eV, dtype=float)
    if np.any(ne <= 0.0) or np.any(Te <= 0.0):
        raise ValueError("ne and Te must be positive")
    log_coeff = _interp_log_bilinear(
        log_ne_grid,
        log_te_grid,
        stages[z1],
        np.log10(ne),
        np.log10(Te),
    )
    return 10.0**log_coeff


# (file, stage, unit) per fused-quantity name. "rate" stays in cm^3/s;
# "power" converts the file's W cm^3 to eV cm^3/s, matching the single-table
# helpers below.
_HE_QUANTITIES = {
    "scd": ("scd96_he.dat", 1, "rate"),
    "acd": ("acd96_he.dat", 1, "rate"),
    "plt1": ("plt96_he.dat", 1, "power"),
    "plt2": ("plt96_he.dat", 2, "power"),
    "prb1": ("prb96_he.dat", 1, "power"),
}


@lru_cache(maxsize=None)
def _shared_grid_tables():
    """Return the shared (log_ne, log_te) axes and every quantity's table.

    The four '96 GCR files are tabulated on one identical grid; asserting that
    here is what makes the fused lookup's single coordinate computation valid.
    """
    axes = None
    tables = {}
    for name, (fname, z1, unit) in _HE_QUANTITIES.items():
        log_ne, log_te, stages = _load(fname)
        if axes is None:
            axes = (log_ne, log_te)
        elif not (
            np.array_equal(log_ne, axes[0]) and np.array_equal(log_te, axes[1])
        ):
            raise ValueError(
                f"{fname}: adf11 axes differ from {next(iter(_HE_QUANTITIES.values()))[0]}; "
                "the fused lookup requires one shared grid"
            )
        tables[name] = (stages[z1], unit)
    return axes, tables


def he_rates(ne_cm3, Te_eV, quantities, low_te_extension=False):
    """Fused lookup of several He coefficients at one (ne, Te).

    Computes the bilinear grid coordinates once and blends each requested
    table at them, so N quantities cost one coordinate solve plus N cheap
    gathers instead of N full interpolations. ``quantities`` is an iterable
    drawn from {"scd", "acd", "plt1", "plt2", "prb1"}; returns a dict in the
    same units as the corresponding single-table helpers (bit-identical
    values -- both paths share the same blend arithmetic).

    ``low_te_extension`` (default False, bit-exact off): below the adf11
    Te grid edge (0.2 eV) the standard lookup clamps nearest-edge, which
    freezes ACD exactly where the detachment-regime recombination
    explodes (three-body ~ Te^-9/2 * ne). With the extension on, "acd"
    (and "prb1", holding the per-event radiated energy at its edge
    value) are scaled below the edge by the in-repo janev shape anchored
    continuously at the edge:

        R(Te, ne) = [alpha_r(Te) + ne*alpha_3(Te)]
                    / [alpha_r(edge) + ne*alpha_3(edge)]

    -- the model's own radiative + classical three-body coefficients, so
    the extension carries the known power laws with no new data and no
    free parameters (KINETIC_TWOZONE_PLAN/§5b afterglow ledger audit,
    2026-07-22). Quantities other than acd/prb1 keep the clamp (SCD at
    sub-edge Te is exponentially dead regardless).
    """
    (log_ne_grid, log_te_grid), tables = _shared_grid_tables()
    ne = np.asarray(ne_cm3, dtype=float)
    Te = np.asarray(Te_eV, dtype=float)
    if np.any(ne <= 0.0) or np.any(Te <= 0.0):
        raise ValueError("ne and Te must be positive")
    ix, iy, fx, fy = _interp_coords(
        log_ne_grid, log_te_grid, np.log10(ne), np.log10(Te)
    )
    out = {}
    for name in quantities:
        table, unit = tables[name]
        value = 10.0 ** _interp_blend(table, ix, iy, fx, fy)
        out[name] = value / qe_SI if unit == "power" else value
    if low_te_extension:
        te_edge = 10.0 ** log_te_grid[0]
        below = Te < te_edge
        if np.any(below):
            from cablp.funcs._cross import alpha_3, alpha_r

            I_he = 24.587
            ne_b = np.maximum(ne, 1.0)
            alpha_lo = alpha_r(Te, I=I_he) + ne_b * alpha_3(Te)
            alpha_edge = (
                alpha_r(np.full_like(Te, te_edge), I=I_he)
                + ne_b * alpha_3(np.full_like(Te, te_edge))
            )
            ratio = np.where(
                below, alpha_lo / np.maximum(alpha_edge, 1e-300), 1.0
            )
            for name in ("acd", "prb1"):
                if name in out:
                    out[name] = out[name] * ratio
    return out


def he_ionization_rate(ne_cm3, Te_eV):
    """Effective He0 -> He+ ionization coefficient [cm^3/s] (SCD, z1=1)."""
    return _evaluate("scd96_he.dat", 1, ne_cm3, Te_eV)


def he_recombination_rate(ne_cm3, Te_eV):
    """Effective He+ -> He0 recombination coefficient [cm^3/s] (ACD, z1=1).

    Includes radiative, dielectronic, and three-body recombination at the
    tabulated density, so it replaces the historical alpha_r + ne*alpha_3
    pair as a single coefficient.
    """
    return _evaluate("acd96_he.dat", 1, ne_cm3, Te_eV)


def he_neutral_line_power(ne_cm3, Te_eV):
    """He0 excitation line-power coefficient [eV cm^3/s] (PLT, z1=1).

    Radiation only: unlike the historical IAEA_exp1 fit, this does NOT
    include the ionization-potential loss, which sim1d books separately as
    ``I_ion * S_ion``.
    """
    return _evaluate("plt96_he.dat", 1, ne_cm3, Te_eV) / qe_SI


def he_ion_line_power(ne_cm3, Te_eV):
    """He+ excitation line-power coefficient [eV cm^3/s] (PLT, z1=2)."""
    return _evaluate("plt96_he.dat", 2, ne_cm3, Te_eV) / qe_SI


def he_recombination_power(ne_cm3, Te_eV):
    """He+ recombination + bremsstrahlung power [eV cm^3/s] (PRB, z1=1)."""
    return _evaluate("prb96_he.dat", 1, ne_cm3, Te_eV) / qe_SI
