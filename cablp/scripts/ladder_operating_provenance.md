# Provenance of the per-setting operating points (`run_mechanism_ladder.ES_OPERATING`)

`ES_OPERATING` in `scripts/run_mechanism_ladder.py` carries the two machine
inputs that vary between discharge settings: the open-circuit bank voltage and
the heater-maintained standby cathode surface temperature. Everything else in
the model configuration is held fixed across the ladder.

It is imported by `scripts/baseline_sim1d.py` as well, so a change here reaches
the regression fixture. Parameter meanings are in the docstrings of
`cablp/solvers/_sim1d/core/config.py`; provenance classes are defined in
`cablp/solvers/_sim1d/core/config_defaults_provenance.md`.

| setting | `V_bank` [V] | `Ts_standby_K` [K] |
|---|---|---|
| 1 | 177.843 | 1910.0 |
| 2 | 138.303 | 1949.0 |
| 3 | 98.814 | 1972.0 |
| 4 | 98.978 | 1972.0 |

## `V_bank` — MEASURED

All four are pre-shot open-circuit readings taken on the same probe channel as
V_dis. Bars: **+/-0.03 V SEM**, plus a **+/-1.2% multiplicative instrumental
systematic** that is unresolved between supply regulation and probe gain.

These are a different quantity from the supply setpoint that `config.py`
defaults `V_bank` to (180 V, the dial). Any run that means the machine rather
than the dial must take its `V_bank` from here.

*Correction of record.* An earlier version of this table mixed conventions:
setting 1 carried a FITTED 173.6 from the near-singular single-trace circuit
fit while settings 2 and 3 were measured. They are now uniformly measured. See
`circuit_constrained_refit.md` (an untracked working memo alongside these
scripts) for the fit that superseded it.

**Settings 3 and 4 previously shared the literal 99.0 and now differ by
0.164 V.** Setting 4's reading is window-corrected. Do not re-collapse them.

## `Ts_standby_K` — MEASURED

Digitized standby surface temperatures. They are operational machine setpoints
and are **not to be tuned**: the cathode calibration is carried on the effective
Richardson constant `C_R` instead, which is the same flat direction (~100 K of
standby per e-fold of emission) parameterized on the constant the cathode
literature already treats as effective. See
`scripts/production_stance_provenance.md`.

Setting 4 runs the same heater as setting 3 and the bank set to the same dial;
only the puff drive differs (110 V versus 76.4 V).

## Driver defaults that are not operating points

`run_mechanism_ladder.py`'s command-line defaults are not machine
measurements:

- `--g-cond` (skin-to-substrate conduction, W/K) — **FITTED**; the one fitted
  knob of the cathode power balance, frozen after calibration at setting 1. The
  production stance uses a different value.
- `--c-th` (skin-layer heat capacity, J/K) — **ASSUMED**; shapes only the ramp
  timescale, and the steady state is independent of it.
- `--emissivity` — literature value for LaB6.
- `--standby-offset-K` — a stability-derivative probe, deliberately NOT a tuning
  knob; it offsets the measured standby to measure a sensitivity.
