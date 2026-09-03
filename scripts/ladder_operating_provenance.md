# Provenance of the per-setting operating points (`run_mechanism_ladder.ES_OPERATING`)

`ES_OPERATING` in `scripts/run_mechanism_ladder.py` carries the two machine
inputs that vary between discharge settings: the open-circuit bank voltage and
the heater-maintained standby cathode surface temperature. Everything else in
the model configuration is held fixed across the ladder.

It is imported by the campaign drivers — `scripts/run_m6_point.py` and
`scripts/run_es1_r5_iter.py` among them — so a change here reaches every
campaign arm. It does NOT reach the regression fixture: `baseline_sim1d.py`
imports no campaign driver at all, deliberately, so these override dicts cannot
touch the golden. Parameter meanings are in the docstrings of
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

## `Ts_standby_K` — DERIVED

**Not a temperature the machine reports.** The quantity that is set is the
HEATER CURRENT — 1775 / 1865 / 1920 A for settings 1 / 2 / 3, operator-set on
the console — and the temperatures in the table are those currents read through
the source paper's Fig-10 heater-current → surface-temperature map (that
paper's pyrometer, digitized 2026-07-20). The heater currents are the inputs of
record because the machine's own instrumentation cannot supply the temperature:
its pyrometer channel reads 0.0 in every shot, and its heater-current data
channel is invalid-flagged and uncalibrated.

The map's slope is ≈ 0.45 K/A with a **[0.43, 0.50] K/A bracket**, and the
rungs recorded above sit at the 0.43 K/A edge of it, to rounding. The map is
read in °C: the three heater currents interpolate to 1637 / 1676 / 1699 °C,
i.e. 1910.15 / 1949.15 / 1972.15 K, and `ES_OPERATING` stores them rounded to
1910.0 / 1949.0 / 1972.0 — which is why a bracket quoted to a tenth of a kelvin
can otherwise appear to exclude the shipped rung.

*Anchor for the ± 3 K.* ES1 interpolates the Fig-10 map between its
1700 A ≈ 1603 °C and 1800 A ≈ 1648 °C points: at 1775 A that is 1637 °C. Under
the slope bracket the interpolation moves 1603 + 75 × [0.43, 0.50] =
**[1635.3, 1640.5] °C**, i.e. within about ± 3 K of 1910 K.

*Sensitivities.* Under the slope bracket alone, setting 1 moves by at most
**± 3 K** — absorbed into `C_R` along the flat direction below as a factor
**× [0.97, 1.03]**. Stated to the record's own precision, setting 2 spans
**[1949, 1955] K** and setting 3 **[1972, 1983] K**; each contains its shipped
rung. The slope bracket is not the real uncertainty. The
**ABSOLUTE pyrometer bar of ± 50–100 K** is, and along the same flat direction
it maps to `C_R` **× [0.61, 1.65]** and **× [0.37, 2.72]** respectively. That
bar is invisible to every ES1 score, because `C_R` is the fit target — the
calibration absorbs it by construction, which is why it must be carried as a
stated bracket rather than read off the agreement.

The heater settings behind them are operational machine settings, and the
temperatures are **not to be tuned**: the cathode calibration is carried on
the effective Richardson constant `C_R` instead, which is the same flat
direction (~100 K of standby per e-fold of emission) parameterized on the
constant the cathode literature already treats as effective. See
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
  knob; it offsets the map-derived standby to measure a sensitivity.
