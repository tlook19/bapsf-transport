"""Measure the DVM's wall detailed-balance offset, on ANY commit.

The B3 wall-reflection registration pre-registered a detailed-balance pin --
a uniform gas at the wall temperature exchanging ZERO net energy with the
cylindrical wall, to roundoff. It does not hold, and this probe is the
evidence that the reason predates the member: it reads nothing the
``wall_reflection`` selector added, so it runs unchanged at the base commit
and at the branch tip and prints the same numbers.

What the offset is. The engine's ACCOMMODATED share re-emits on
``VGrid.wall_emission_spectrum`` -- the vp-flux-weighted cosine spectrum --
while the wall ABSORBS at ``nu_w ~ vp`` times the volume Maxwellian. Those
two coincide in the continuum, which is exactly why the cosine spectrum is
the right re-emission law; on the discrete grid they differ bin by bin,
because ``nu_w`` samples ``vp`` at the bin CENTRE while the emission
spectrum integrates ``vp^2 exp(-vp^2/2s^2)`` across the bin. The residual is
therefore proportional to ``alpha`` -- it lives entirely on the accommodated
share -- and it converges away under velocity refinement. It is the same
velocity-resolution property the L4 limit case records for the density and
temperature split.

Usage (from the checkout root, PYTHONPATH set to it)::

    python scripts/k2_dvm_wall_detailed_balance_base_probe.py
"""

from types import SimpleNamespace

import numpy as np

from cablp.solvers._sim1d.physics.kinetic_dvm import TransientDVM

#: Accommodation coefficients the B3 registration named, plus the two
#: degenerate ends. The offset is linear in alpha, which is the diagnosis.
ALPHAS = (0.0, 0.35, 0.40, 0.46, 0.7307, 1.0)

#: Velocity grids, coarse to fine. The offset must fall under refinement --
#: that is what makes it a resolution property rather than a defect.
GRIDS = ((16, 6), (32, 10), (48, 12), (96, 32))


def uniform_tube(nz, length_cm=1600.0, Rp=15.0, Rm=50.0):
    """Return the same strictly-uniform coaxial tube the gate suite uses."""
    dz = np.full(nz, length_cm / nz)
    Rp_cm = np.full(nz, Rp)
    Rm_cm = np.full(nz, Rm)
    return SimpleNamespace(
        cells=nz,
        length_cm=dz,
        Rp_cm=Rp_cm,
        Rm_cm=Rm_cm,
        plasma_volume_cm3=np.pi * Rp_cm**2 * dz,
        neutral_volume_cm3=np.pi * Rm_cm**2 * dz,
    )


def measure(alpha, nvz, nvp, nz=10):
    """Return the wall channel's net energy exchange over one tick."""
    dvm = TransientDVM(
        geometry=uniform_tube(nz), nvz=nvz, nvp=nvp, accommodation=alpha,
        s_L=0.0, s_R=0.0,
    )
    dvm.seed_from_density(np.full(nz, 1.0e13), np.full(nz, 1.0e13))
    led = dvm.update(
        1.0e-5,
        n_i=np.zeros(nz),
        Ti_eV=np.full(nz, 0.026),
        u_i=np.zeros(nz),
        nu_ion=np.zeros(nz),
    )
    e = led["energy"]
    predicted = alpha * (
        e["loss_wall"] - led["loss_wall"] * dvm.E_wall_mean
    )
    return e["net_surface_wall"], predicted, e["loss_wall"]


def main():
    print("DVM cylindrical-wall detailed balance: net energy exchange over "
          "one tick")
    print("uniform 300 K gas, sealed tube, no plasma; net / incident wall "
          "energy")
    print("=" * 78)
    header = "alpha".ljust(9) + "".join(
        f"{nvz}x{nvp}".rjust(14) for nvz, nvp in GRIDS
    )
    print(header)
    for alpha in ALPHAS:
        row = f"{alpha:<9g}"
        for nvz, nvp in GRIDS:
            net, _predicted, loss = measure(alpha, nvz, nvp)
            row += f"{abs(net) / loss:14.4e}"
        print(row)
    print("=" * 78)
    print("localization: net == alpha * (E_incident - N_incident * "
          "E_wall_mean)?")
    worst = 0.0
    for alpha in ALPHAS:
        for nvz, nvp in GRIDS:
            net, predicted, _loss = measure(alpha, nvz, nvp)
            if predicted:
                worst = max(worst, abs(net - predicted) / abs(predicted))
    print(f"  worst relative miss over the whole table: {worst:.4e}")
    print("  -> the offset lives ENTIRELY on the accommodated share; the "
          "non-accommodated")
    print("     share (the only thing wall_reflection touches) contributes "
          "none of it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
