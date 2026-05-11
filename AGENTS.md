# AGENTS.md

This file provides guidance to Codex and other coding agents when working in this
repository.

## Project Overview

`cablp` (Cathode And Basic Linear Plasma) is a 0D/pseudo-1D plasma transport
simulation package for the LAPD (Large Plasma Device).  The current primary
model evolves per-cell electron density, neutral density, electron temperature,
ion temperature, and optional parallel plasma velocity.  It includes bulk and
beam ionization, recombination, neutral molecular-flow transport, parallel and
perpendicular heat transport, electron-ion energy exchange, charge-exchange
cooling, cathode sheath/circuit physics, and optional twin-cathode operation.

## Package Setup

This repository is currently being developed on a Windows PC as part of the
workspace at `D:\bapsf`.  Use the `bapsf-app` mamba environment for Python when
working locally:

```powershell
mamba activate bapsf-app
# or, if this environment is managed through conda:
conda activate bapsf-app
```

The Python package lives under `cablp/` and uses Poetry:

```powershell
cd D:\bapsf\bapsf-transport\cablp
poetry install
poetry build
```

`pyproject.toml` currently requires Python `>=3.14`.  If the local `bapsf-app`
environment has an older Python, update or recreate the environment before
installing.  Core package dependencies are `numpy`, `matplotlib`, `mpmath`, and
`scipy`.

If `mamba`/`conda` is not visible in the active shell, do not assume the
environment is missing.  On Windows, the shell may simply not have been
initialized for conda/mamba; use an initialized PowerShell session or the full
path to the environment's Python executable.

There is no formal test suite or CI in the repository.  Development validation
is done with scripts and notebooks under `scripts/`.  For quick sanity checks,
prefer targeted imports, short simulation runs, and `python -m compileall` over
full parameter sweeps.  Run these checks inside the project environment; importing
`cablp.solvers` also imports the deprecated `_sim.py`, which requires
`matplotlib` at import time.

## Current Entry Point

The current primary solver is:

```python
from cablp.solvers._sim3 import LAPDSim, load_config, default_config

params, flags = load_config("scripts/example.toml")
sim = LAPDSim(params, flags)
sim.start_simulation()
results = sim.get_results()
```

Important current API details:

- The class name is `LAPDSim`, not `LAPDSim3`.
- The solver method is `start_simulation()`, not `solve()`.
- Results are returned in memory as a `types.SimpleNamespace` from
  `get_results()`.  There is no current `_sim3.py` `.save()` method.
- `load_config(path)` reads TOML sections `[params]` and `[flags]`, overlaying
  values onto `input_dict_template` and `input_flags_template`.
- `default_config()` returns copies of the current default parameter and flag
  dictionaries.

## Core Simulation Flow

```text
TOML config or input_dict/input_flags
        ↓
LAPDSim in cablp/cablp/solvers/_sim3.py
        ↓
Adaptive Dormand-Prince RK45 integration
        ↓
State variables per cell:
  ne, nn, Te, Ti, v_plasma
        ↓
SimpleNamespace results from get_results()
```

For `Plasma=True`, the solver runs three adaptive phases:

1. `pre_breakdown`: cathode and gas puff on until `I_tot >= I_breakdown`, or
   raises `BreakdownError` after `tau_prebreakdown`.
2. `main_discharge`: cathode and gas puff on for `tau_discharge` after
   breakdown.
3. `afterglow`: cathode and gas puff off for `tau_afterglow`; cathode solver is
   run in floating mode.

For `Plasma=False`, only the neutral equation is advanced through repeated
puff/off equilibrium cycles controlled by `cycles`, `tau_discharge`, and
`tau_cycle`.

## Module Map

- `cablp/cablp/solvers/_sim3.py` - current primary pseudo-1D solver.  Contains
  `LAPDSim`, `load_config`, `default_config`, `BreakdownError`, adaptive RK45
  stepping, density/heat/velocity RHS assembly, cathode/beam coupling, optional
  adaptive mesh, and result packaging.
- `cablp/cablp/solvers/_sim.py` - deprecated older 0D solver kept for reference.
  Do not extend it for new work unless explicitly asked.
- `cablp/cablp/solvers/_rk.py` - legacy standalone Runge-Kutta helpers.
- `cablp/cablp/funcs/_cathode_solver.py` - thermionic cathode sheath and circuit
  model.  Provides `DeviceConfig`, `PlasmaState`, `solve`, and
  `solve_beam_system`; computes sheath drops, admitted thermionic current,
  plasma/load/compliance power, beam speed/density, beam cross section, and beam
  mean-free-path profiles.
- `cablp/cablp/funcs/_heat.py` - Braginskii parallel/perpendicular heat
  conductivities, finite-volume parallel heat diffusion, electron-ion exchange,
  electron-neutral/ion inelastic cooling, and charge-exchange cooling.
- `cablp/cablp/funcs/_cross.py` - electron-impact ionization cross sections,
  lookup-table interpolation for H/He EII tables, approximate recombination
  coefficients, heavy-particle reaction fits, and charge-exchange rate tables.
- `cablp/cablp/funcs/_fits.py` - IAEA empirical rate/cooling fit expressions and
  Maxwellian rate coefficient helper.
- `cablp/cablp/funcs/_plasmaparams.py` - collision times, thermal velocities,
  gyrofrequencies, and Coulomb logarithms.
- `cablp/cablp/vars/_cons.py` - physical constants in SI/CGS plus selected
  arbitrary-precision values.
- `cablp/cablp/vars/_coeff.py` - IAEA/Janev coefficient tables for cooling,
  ionization, excitation, and heavy-particle reactions.
- `cablp/cablp/vars/_nn_table.py` and `nn_table.csv` - equilibrated neutral
  density lookup table used to infer `nn0` from `S_gp` when `nn0` is omitted.
- `cablp/generate_eii_tables.py` - helper for generating EII lookup CSVs.
- `scripts/generate_nn_table.py` - helper for regenerating `nn_table.csv`.
- `scripts/profile_sim.py` - example/profiling runner for `_sim3.py`.
- `scripts/lecroydaq.py` - LeCroy oscilloscope data acquisition helper.

## Important Parameters and Flags

The source of truth for current parameters and flags is
`input_dict_template` / `input_flags_template` in
`cablp/cablp/solvers/_sim3.py`.  `scripts/example.toml` is useful, but check it
against `_sim3.py` before assuming it is exhaustive or current.

Frequently used parameters:

- Initial state: `ne0`, optional `nn0`, `Te0`, `Ti0`, `Tn_fit`, `gas_type`.
- Geometry: `Lm`, `Rm`, `Lp`, `Rp`, `Rhf`, `Bz0`, fixed 100 cm cathode cells.
- Cathode/circuit: `V_bank`, `T_s`, `phi_wf`, `C_R`, `R_comp`, `eta`,
  `L_cath`, `R_cath`.
- Gas/pumping: `S_gp`, `Twin_S_gp`, `gp_puff_factor`, `tau_gp_ramp`,
  `S_pump_L`, `S_pump_R`.
- Timing: `tau_prebreakdown`, `tau_discharge`, `tau_afterglow`, `tau_cycle`,
  `cycles`, `I_breakdown`, `h0`, `h_max_discharge`, `h_max_afterglow`,
  `rtol`, `h_min`.
- Scaling factors: `b_epara`, `b_ipara`, `b_eperp`, `b_iperp`, `b_ioniz`,
  `b_rec_rad`, `b_rec_3b`, `b_Qcx`, `b_source`, `b_Qie`, `b_Qei`, `b_Qen`.
- Adaptive mesh controls: `max_cells`, `min_cells`, `mfp_refine_threshold`,
  `mfp_coarsen_threshold`.

Current flags include:

- `Plasma` - enable coupled plasma equations; false means neutral-only
  equilibrium cycling.
- `TwinCathode` - enable a second cathode at the far boundary.
- `Velocity` - evolve parallel flow velocity.
- `advection` - include `v dot grad v` in the velocity equation.
- `adaptive_mesh` - dynamically refine/coarsen interior cells by bulk electron
  mean-free-path criteria.
- `nonlocal_ne` / `sonic_ne` - alternate density transport schemes.
- `eperp`, `iperp`, `icool`, `ncool`, `cx`, `icool_recomb` - heat/cooling
  physics toggles.
- `C_imp`, `O_imp`, `mit_el` - placeholders or currently inactive paths.

## Result Object Shape

`get_results()` returns arrays padded to `max_cells` with `NaN` when adaptive
mesh is enabled.  Time is reported in milliseconds and, for plasma runs, shifted
so breakdown is at `t=0` after finalization.

Useful result fields include:

- State: `time`, `t_breakdown`, `ne`, `nn`, `n_beam`, `Te`, `Ti`, `v_plasma`.
- Density terms: `Ne_flux`, `Nn_flux`, `S_ion_bulk`, `S_rec_rad`, `S_rec_3b`,
  `S_ion_beam`.
- Heat terms: `e_par_flux`, `i_par_flux`, `e_perp_hl`, `i_perp_hl`, `Qie`,
  `Qei`, `Qen`, `Qcx`, `Qeb`, `Qib`, `div_v_elec`, `div_v_ions`, `Te_conv`,
  `Ti_conv`.
- Diagnostics: `isat`, `primary_mfp`, `bulk_mfp`, `ln_lambda`,
  `cells_at_time`, `refinement_events`.
- Cathode diagnostics: `cathode` and `cathode_twin`, each with fields such as
  `phi_c`, `phi_a`, `V_b`, `I_i`, `I_eth_star`, `I_tot`, `P_prim`,
  `P_ohmic`, `P_net`, and `P_loss`.

## Development Notes

- The code is physics-heavy and unit-sensitive.  Preserve existing CGS/SI
  conventions and comments when modifying formulas.
- `_sim3.py` uses per-cell NumPy arrays throughout.  Avoid scalar-only
  assumptions unless working in the deprecated `_sim.py`.
- Several names are historical: `icool` controls electron cooling on ions,
  `ncool` controls electron cooling on neutrals, and `Qie` is the electron-to-ion
  energy transfer term that appears with opposite signs in the electron and ion
  temperature equations.
- `lookup_nn0()` is only valid for tabulated `S_gp` values from 100 to 16000 SCCM
  and will raise outside that range.
- Scripts and notebooks under `scripts/` are a mix of current utilities,
  exploratory analysis, and older examples.  Prefer `_sim3.py` templates over
  notebook assumptions when behavior conflicts.
- The repository currently contains generated artifacts and ignored paths.  Do
  not clean or revert unrelated files unless explicitly asked.

## Suggested Sanity Checks

From the repository root:

```powershell
cd D:\bapsf\bapsf-transport\cablp
python -m compileall cablp
python -c "from cablp.solvers._sim3 import default_config, LAPDSim; params, flags = default_config(); flags['Plasma'] = False; params['cycles'] = 1; params['tau_cycle'] = 1e-3; params['tau_discharge'] = 2e-4; params['h_max_discharge'] = 1e-4; params['h_max_afterglow'] = 1e-4; sim = LAPDSim(params, flags); sim.start_simulation(); print(sim.get_results().nn[-1])"
```

For Git Bash or another POSIX-style shell, the same neutral-only smoke test can
be written as:

```bash
python - <<'PY'
from cablp.solvers._sim3 import default_config, LAPDSim

params, flags = default_config()
flags["Plasma"] = False
params["cycles"] = 1
params["tau_cycle"] = 1e-3
params["tau_discharge"] = 2e-4
params["h_max_discharge"] = 1e-4
params["h_max_afterglow"] = 1e-4
sim = LAPDSim(params, flags)
sim.start_simulation()
print(sim.get_results().nn[-1])
PY
```

Short plasma runs may fail to break down if `tau_prebreakdown` or cathode/gas
parameters are too small; this is model behavior, not necessarily a code error.
