"""Probe the initial timestep bound of the conducting-phase window config.

Cheap, no-solve diagnostic: build the conducting-phase configuration at a
range of plasma seeds and report the suggested timestep and its binding
limiter at t = 0. A seed placed exactly AT ne_floor leaves the drain-margin
bound no margin at all and pins the run at dt_min, which is a run-cost
property of the configuration rather than a physics one -- this is the probe
that says where that edge is.
"""

import argparse

from cablp.solvers._sim1d import LAPDSim1D

from covbuild_run_conducting_phase import build_config


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=60)
    p.add_argument("--seeds", nargs="*", type=float,
                   default=(1e8, 2e8, 5e8, 1e9, 1e10))
    args = p.parse_args(argv)

    print(f"conducting-phase window probe, nx={args.nx}")
    for ne0 in args.seeds:
        params, flags = build_config(args.nx, extra={"ne0": ne0})
        sim = LAPDSim1D(params, flags)
        diag = sim.suggest_timestep(include_heat_conduction=False)
        bounds = {
            name[3:]: float(getattr(diag, name))
            for name in vars(diag)
            if name.startswith("dt_") and name not in ("dt_max", "dt_raw")
        }
        limiters = ", ".join(
            f"{name}={value:.3e}"
            for name, value in sorted(bounds.items(), key=lambda kv: kv[1])[:4]
        )
        print(
            f"  ne0={ne0:.3e}  dt={diag.dt:.3e} s  "
            f"constraint={diag.active_constraint}  "
            f"clamped_to_dt_min={diag.clamped_to_dt_min:g}  "
            f"dt_raw={diag.dt_raw:.3e}"
        )
        print(f"      limiters: {limiters}")


if __name__ == "__main__":
    main()
