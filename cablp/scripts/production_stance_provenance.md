# Provenance of the production stance (`scripts/stances/g1atrim.toml`)

The production stance of record is the committed stance file
`scripts/stances/g1atrim.toml`, loaded by `scripts/stance_config.py`
(R1, 2026-08-20; it replaced the former `PARAM_OVERRIDES` / `FLAG_OVERRIDES`
dicts in `scripts/compare_sim1d_es1.py`). This file records where each stance
value came from. Parameter *meanings* are in the docstrings of
`cablp/solvers/_sim1d/core/config.py`; defaults provenance is in
`cablp/solvers/_sim1d/core/config_defaults_provenance.md`, which also defines
the provenance classes MEASURED / DERIVED / FITTED / ASSUMED used here.

**Where the values live changed at R2a (2026-08-20), and this note now spans
both homes.** The fold-in pass made the g1atrim package the shipped
`default_config()` defaults; the stance file retains only the keys that could
not fold (per-mesh arrays, per-rung operating measurements, provenance-excluded
and consequence keys — the R2a exception list). Entries below whose key folded
now describe a **config default**; their provenance (value, class, bar, memo)
is unchanged by the move, and `config_defaults_provenance.md` carries the
default-side entry. The golden fixture pins every pre-fold value as a literal
(`golden_baseline_provenance.md`), so no historical artifact tracks the fold.

Analysis memos named below are working files kept alongside these scripts and
are not tracked in this repository.

## Circuit

Memos: `circuit_constrained_refit.md`, `circuit_vdis_three_rung_read.md`,
`l_parasitic_reconciliation.md`.

All four circuit constants come from one V0-pinned constrained refit — V0 fixed
per setting at its measured pre-shot reading, with C, R and L shared across four
settings (N = 1952, window 0.3-19.8 ms). It replaced a free 4-parameter fit to a
single trace which was near-singular (corr(V0, R) = 0.997, R swinging
1.9-5.7 mOhm with the window). The defect was invisible on the setting it was
fitted to and appeared only on transfer; see the defaults provenance note for
the residual comparison.

| key | value | class | bar |
|---|---|---|---|
| `V_bank` | 177.843 V | MEASURED | +/-0.03 V SEM, plus a +/-1.2% multiplicative instrumental systematic |
| `R_comp` | 7.2244e-3 Ohm | MEASURED | 7.213 +/- 0.043 mOhm jackknife; settings agree to 1.8% |
| `C_bank_F` | 9.5 F | MEASURED | 9.5187 +/- 0.66 F; hardware-bounded to [7.56, 12.60] F |
| `L_parasitic_H` | 8.1e-6 H | DERIVED | bracket 7.6-8.4 uH from two independent instruments |

`V_bank` here is the measured pre-shot open-circuit bank voltage on the same
probe channel as V_dis. It is NOT the 180 V supply setpoint that
`config.py` defaults to; those are different quantities. The +/-1.2%
systematic is unresolved between supply regulation and probe gain.

`R_comp` and `C_bank_F` come from ONE joint fit and must move together.

`L_parasitic_H` was 6.6e-6 until the reconciliation. 6.6e-6 was the orphaned
fourth member of the same retracted free fit as the old `R_comp`/`C_bank_F`,
not a considered choice; it is still inside the plateau refit's jackknife bar
(6.7 +/- 2.5 uH) and is therefore not excluded, only unsupported.

L is inert for every scored row -- and, MEASURED on the production reference
run 2026-08-03, inert on the UNSCORED timing observables too: across the
6.6 -> 8.1 uH change t90 moved 0.00000 ms and ignition -0.0035 ms.

*(Corrected 2026-08-17, THESIS_NOTES chain pass. This paragraph previously
read "its measurable consequences are confined to the unscored reported
fingerprints (t90 +0.05..0.11 ms, ignition +0.02..0.07 ms, both toward the
measurement)." That projection is WITHDRAWN: it extrapolated
d(t90)/d(ln L) ~ 0.24 ms from a 12-vs-20 uH artifact pair, and the
sensitivity does not hold down at 6.6-8.1 uH. The direct measurement above
postdates and supersedes it. This matters more here than in a narrative
document: THESIS_NOTES defers to this note by policy when the two
disagree, so a retracted number left standing here would have WON that
disagreement. The adoption of 8.1 uH is a pure consistency and provenance
correction with no measurable physical consequence whatsoever.)*

The regression fixture has pinned 8.1e-6 all along, so it is bit-exact
across the change.

## Cathode emission

**`C_R = 7.09` — FITTED (the one drive-side fit knob; 2026-08-19 stance
event).** Value chain: the 14.25 derivation below was superseded when the
L2 geometry rebaseline moved the operating point to the measured
18.415 cm aperture (the l2a4 free fit landed 7.26 there; the G1
measured-geometry arms ran 6.94); the 2026-08-19 D-phase conservation
fixes (see "Conservation fixes" below) changed the plasma load, and the
REGISTERED one-knob re-trim (drive-band target ONLY, once at ES1, frozen
and transferred; campaign log 2026-08-19 zzzz) landed **7.09** — peak
2997 A / plateau 2964 A, ratio 1.000 against the measured band, matched
on the first log-knob interpolation (measured local exponent 0.99966;
the super-linear 1.392 figure is kinetic-era only). Port scores were
reported unconditionally and never entered the selection: n 0.91,
Isat 0.84 (1.9σ), Te 0.88 at the matched drive. `cathode_Ts_base_K`
remains pinned at the measured standby (the flat-direction rule below
stands: only one member carries the calibration).

*(Historical derivation of the pre-rebaseline 14.25, retained as record —
the flat-direction and residual-honesty analysis below still applies to
the current knob:)* `C_R` is treated by the cathode
literature as an effective emission constant (surface state, patch fields,
non-ideal emitting fraction), not the 120 A/cm^2/K^2 Richardson-Dushman
universal. The value is obtained by matching the emission at the operating
point in the code's own expression `J = C_R T^2 exp(-e phi/(kB T))`:

    J(C_R_eff, T + dT) = J(29.0, T)
    =>  C_R_eff = 29.0 * (T/(T+dT))^2 * exp(-(e phi/kB)(1/T - 1/(T+dT)))

    T   = 1859.02 K   plateau surface temperature, mean over 15-19.5 ms on the
                      main-discharge clock of the reference run
    dT  = +70 K       the base-temperature move, taken to propagate ~1:1 into
                      the plateau at fixed heater power (the warming balance's
                      restoring term is G_cond*(T_s - T_base), so a rigid base
                      shift translates the operating point). ASSUMPTION.
    phi = 2.809 eV    the work function the emission actually evaluates at the
                      plateau: `cathode_phiwf_clean_eV`, since the ads_des
                      surface is fully cleaned there (recorded
                      phi_wf_eff = 2.809, theta ~ 1e-19). The uncleaned
                      shot-start 2.869 eV would give 14.06, a 1.4% difference,
                      well inside the residual below.

    => C_R_eff = 14.2546 -> 14.25 adopted (14.3 to 3 s.f.; the extra digit
       keeps the point-emission match at 0.03%, inside the 0.1% the derivation
       was pre-registered to hit).

**`cathode_Ts_base_K` is deliberately NOT pinned here.** It is inherited from
the config default, which is the MEASURED standby temperature. An earlier
stance pinned 1840 K — 70 K below the measurement — which mislabelled a measured
quantity as a fit. `C_R` and `cathode_Ts_base_K` are the same flat direction
(~100 K of standby per e-fold of emission), so only one may carry a
calibration, and it is the constant the literature already treats as effective.

Residual honesty: the match is exact only at the plateau point. The flat
direction is not perfectly flat (~103 K per e-fold, recorded), so shifts of
order the ~10% ramp-gain slope across it are ACCEPTED and revalidated by run
rather than tuned away. Standby emission is not matched exactly either: 1910 K
on `C_R_eff` versus 1840 K on 29.0 emits 1.4% more at phi = 2.809 and 2.8% at
2.869 — the same flat-direction residual seen off the operating point.

**`cathode_conduction_W_per_K = 12058.0`, `cathode_heat_capacity_J_per_K =
181.0`** — DERIVED by areal transcription (L2 geometry rebaseline, 2026-08-17).
Both are extensive in the cathode face area, so moving the face from the
fitted 15.0 cm to the measured 18.415 cm aperture scales them by
`(18.415/15)^2 = 1.5072`: 8000 -> 12058 W/K and 120 -> 181 J/K. The underlying
calibration is UNCHANGED — the conduction remains the one fitted knob of the
cathode power balance, co-tuned with `S_gp` at the reference setting and
frozen thereafter; only the area it is quoted per has been corrected. The heat
capacity was previously inherited from the config default and now differs from
it, so it is pinned in the stance dict.

Disclosure: the physically conducting and radiating body is the whole 19.05 cm
disc, not the exposed aperture, which would give `(19.05/15)^2 = 1.613` —
about 7% above the transcription adopted here. That 7% sits well inside the
hand-tuned class both knobs already carry, and the aperture factor is used for
consistency with the emitting/collecting area the rest of the stance keys off.
Neither number is re-fitted to absorb the difference.

## Conservation fixes (2026-08-19 stance event — ARMED stance members)

**`neutral_hot_internal_wall = true`** (flags) and
**`cathode_jet_energy_convention = "total_reflected"`** (params) are
armed in the production stance as of the 2026-08-19 adoption
(campaign @ c1947d8 carries the code; both keys defaulted OFF in the
config until the R2a fold, 2026-08-20 — they are now the shipped
defaults, with the golden pinning the historical OFF/"legacy" values
for bit-exactness of historical artifacts). Class: neither is a
calibration — both restore conservation. The wall flag ends a measured
particle deletion (~2e21 atoms/s, ~25 % of puff scale, at the plenum;
kernel flights now terminate at plasma-dead faces); the convention key
ends a measured ~24 kW energy destruction (the debit and the launch now
share one TRIM-convention spec; per-particle identity exact). Joint
effect at the stance: mid-machine density +6–8 % toward measured on two
geometries at zero tuning, established with input power FALLING
(particle/fueling-side, not a power artifact). Record: campaign log
2026-08-19 rrrr–zzzz; the plan of record is POWER_COUPLING_PLAN.md
(local docs). The production GEOMETRY is the G1a measured-machine
package (grid of record nx 268 / Lm 2117.8 / collector 7.8 + the sss
fidelity rulings; coils-ON droop-min ansatz f_end 2.221 — always quoted
as the ansatz, never "coils on" generically).

## Neutral source

**`S_gp = 9010` sccm per valve × 2 valves, `gas_puff_delivery_fraction
(f_gp) ≡ 1.0` — MEASURED (hardware), the L2 operating point (adopted at
l2a7b, carried unchanged by the 2026-08-19 stance event).** The 9010 is
the flowcal hardware measurement of the East valve's sustained plateau
flow at the recorded 76.4 V piezo drive (censored forward-model fit of
`flow_data_2026-02-19.hdf5`; A_v(76.4 V) = 9.01 slm, stat [9.00, 9.04],
full syst [8.65, 9.50]; West = East ASSUMED; total delivered
[7.2, 8.1]e21 atoms/s; the meter sits upstream so the number is the
sustained valve flow — no droop term against a square-waveform source).
The derivation chain of record: `scripts/flowcal_{map,sensor,
censored_fit,conservation,figs}.py` + the 2026-08-13 campaign-log
entries (local docs). **History:** the earlier FITTED values (3000
production / 5200 kinetic-operating; the previous text of this entry
said 3000 and predated the L2 adoption — corrected 2026-08-19) sat
inside the hardware envelope as an implied delivery fraction
f_gp ≈ 0.577 — the "1.5–1.7× box." The L2 ruling retired the fitted
level in favor of hardware-true fueling at f_gp ≡ 1.0. **Post-D2
reconciliation pending (thread 23 phase 0):** the restored ~2e21
atoms/s recycle stream changes the model's effective fueling economy;
the box-vs-stance reconciliation re-derives against the adopted-stance
ledger and lands here when ruled.
`S_gp_decay_target`, `tau_gp_pulse_duration`, `tau_gp_decay_duration` are read
only by the deprecated pulse/decay waveform modes and are inert under the
square waveform.

**`equilibration_gas_puff_on_s = 25e-3`** — MEASURED, boxed. The total gas-puff
pulse width across the four settings was ~25 ms: the operator fires the valve,
waits out the machine breakdown delay (~4-6 ms), holds 20 ms from 1 kA, and
rounds up. It is measurable from the V_dis traces, so the number is refinable,
not fitted. Without it the equilibration inherits `tau_discharge` (20 ms) as its
per-cycle puff window — a double duty with no physical basis. It changes the
scorer's runs (the equilibrated seed rises ~x1.25 in delivered fuel) and not the
regression fixture, which pins the key back to `None`.

## Geometry

**`Rp = 18.415`, `R_cath = 18.415`** — MEASURED (caliper, 2026-08-17; L2
geometry rebaseline). The cathode is a 15.0 in x 0.25 in LaB6 disc
(R = 19.050 cm) held by a backside carbon ring against a graphite front panel
whose opening is 14.5 in — r = 18.415 cm. The ring's lip fit on the disc is
self-confirming: the overlap the ring needs is exactly the annulus the panel
covers. The exposed aperture, not the disc, is the emitting, collecting and
conducting face, so both radii identify with it and are equal by measurement
rather than by the previous coincidence of one fitted 15.0. Mapping the
aperture along the field to the plasma-column radius is ASSUMED 1:1 (recorded
ruling: no flux-tube expansion or compression is modelled between the cathode
face and the column).

Both values were 15.0 before this pass — a fit, not a measurement, and one
that conflated the emitting radius with the plasma-column radius. The golden
fixture pins 15.0/15.0 explicitly (`baseline_sim1d.BASELINE_PARAM_OVERRIDES`)
so the regression anchor does not track this stance.

**`end_expansion_*`, `Rcs`, `Lcs`, `Rsup` — RETIRED FROM THE STANCE (R1,
2026-08-20).** The G1 measured-geometry adoption replaced the parametric flare
with the prescribed-area profiles, and the R1 de-staling deleted these keys
from the stance (flag `end_expansion_geometry = False`; params at their
inert defaults). The entry below is retained as the historical record of the
pre-G1 interim geometry it described:

*(historical)* `end_expansion_cells = 10`, `end_expansion_machine_radius_cm
= 100.0`, `end_expansion_plasma_radius_cm = 18.415`, `Rcs = 40.0`,
`Lcs = 25.0`, `Rsup = 0.0` — ASSUMED, an interim geometry pending a 2D model. The end vessel
expands to a 1 m neutral radius over 10 cells with no plasma flare (the plasma
stays at `Rp`); the plenum choke is an obstruction with no support rods; no
baffles; collector length unchanged. `end_expansion_plasma_radius_cm` carries
no independent content: it RIDES `Rp` by construction of the no-flare arm, and
moved to 18.415 with it. The regression fixture needs no pin for it — the
baseline pops the `end_expansion_*` params when the flag is off.

**`source_region_length_cm = 100.0`, `source_region_dz_cm = 10.0`** (with
`source_fixed_grid = True`) — ASSUMED, interim geometry. *(All three FOLDED to
config defaults at R2a, 2026-08-20; no longer stance keys.)* The 100 cm column in
front of the anode is meshed at exactly 10 cm regardless of `nx`, so refining
`nx` refines only the far column and no longer moves the source cells or the
puff cell underneath the source terms. All three must travel together: the
geometry is presence-gated both ways and raises if either parameter is set
without the flag. `nx_gap` is deliberately not pinned — it is already the config
default, so it never appears in the delta.

## Transport and closures

**`atomic_rate_model = "adas"`** — published inputs, not tuned. Because the ADAS
cooling coefficients are radiation-only, `b_Qei = b_Qen = b_Qcx = 1` is
meaningful under this model rather than a null setting.

**`b_beam_excitation = 1.4`** — ASSUMED. 1.0 books the He 2^1P channel alone and
the extra 0.4 approximates the rest of the singlet manifold. It is inert under
`beam_deposition_model = "csda"`, which uses the measured manifold knob-free.

**`heat_flux_limiter_f = 0.1`** — ASSUMED. The free-streaming cap on the
parallel electron heat flux. It combines harmonically (Cowie-McKee) with the
Braginskii flux at `heat_flux_limiter_exponent = 1`, which is already the
config default. This coefficient is a bracket, not a measurement. *(Split at
R2a, 2026-08-20 — the FLAG `electron_heat_flux_limit` folded to a config
default of `True` while the VALUE stayed stance-side — and rejoined at R2b the
same day: `heat_flux_limiter_f = 0.1` is now the config default too, so the
stance file names neither. It was held back from R2a on the expectation that
folding it would move the golden; it does not — the R1 literal pin had already
made the fixture immune, verified byte-identical. See
`config_defaults_provenance.md`.)*

**`beam_deposition_smoothing_cm = 50.0`** — **ASSUMED**, nominally a physical
straggling width. The CSDA range profile is sharp on the mesh scale; smoothing
over a fixed physical length keeps the deposited power from following cell
edges and makes the deposition profile mesh-convergent.

**Honest bar: this kernel IS the applied axial deposition geometry, not a
numerical tidy-up of it, and its width is not measured.** Measured on the
`es1_k6d_sgp5200_nx240` plateau (5.0-19.5 ms, `I_tot` 1904 A, CSDA), the RAW
stopping profile is **2 cells wide at half maximum** — FWHM 15.0 cm on 7.5 cm
cells — and puts 7.6% of the beam power in a SINGLE cell. The 50 cm kernel
widens that to FWHM 377.5 cm, a factor of ~25, and drops the peak cell to
1.6%. Total deposited power is unchanged (4.01313e+08 erg/s either way), so
the kernel moves only WHERE the power lands, which is the whole axial
structure of the beam heating. A result that depends on the axial shape of
beam deposition is therefore reporting this assumed 50 cm, not the stopping
physics.

Two consequences follow and are NOT yet discharged:

- The 50 cm is not derived from a straggling calculation anywhere in this
  repository or its history; it is a round number chosen for the stated
  mesh-convergence purpose. It should be treated as a **bracket arm**, not a
  value, until a straggling width is computed or measured.
- Because the raw support is sub-cell to 2 cells, "mesh-convergent" here means
  convergent to the KERNEL, not to the deposition. Refining `nx` cannot expose
  the assumption; only varying `beam_deposition_smoothing_cm` can.

Evidence: `scripts/smallbatch_beam_smoothing_support.txt` (the raw-vs-applied
support measurement above); the sub-cell finding it confirms is the deposition
discriminator of 2026-08-05.

**`implicit_heat_scheme = "tr_bdf2"`, `operator_splitting = "strang"`,
`heat_picard_iterations = 2`, `heat_picard_tol = 1e-10`** — DERIVED accuracy
choices; all three are needed together for second order.

**`ion_neutral_drag_cx_only = False`** — the legacy ion-neutral thermalization
arm, subsumed by the Phelps moment operator that is now the config default. The
legacy drag keys are deprecated and are deliberately no longer set here.

## Deliberately absent

- `T_s` — the config default is identical, and under `power_balance` it is only
  the initial surface temperature.
- `beam_product_transport` — `"local"` is both the stance and the config
  default; the non-default arm must travel with the run it scored.
- Run-cost settings (`tau_afterglow`, `max_steps_action`, `density_dt_fraction`)
  — they buy runtime, not physics, and belong on the command line of the run
  that wants them.
