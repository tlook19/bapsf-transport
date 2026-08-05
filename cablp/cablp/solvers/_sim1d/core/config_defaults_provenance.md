# Provenance of the `LAPDSim1D` config defaults

Companion to `core/config.py`. The docstrings in that file say what each
parameter **means**: units, sign convention, which term consumes it, what
`0`/`None` does, what raises, which flag gates it. This file says where the
shipped **numbers** came from.

The split is deliberate. A parameter's meaning is the same under every
configuration, which is exactly why a fitted or measured value does not belong
in the place that defines the meaning. Value history in a docstring has twice
been mistaken for a specification and "corrected" in the wrong direction.

## Provenance classes

| class | meaning |
|---|---|
| **MEASURED** | read off an instrument, a trace, or a hardware spec sheet |
| **DERIVED** | computed from measured or literature quantities through a stated relation |
| **FITTED** | obtained by matching model output to data |
| **ASSUMED** | chosen; no instrument pins it. A bracket, not a point, is the honest claim |

Where a quantity cannot be pinned, the **bracket is the claim** and a result
using it must report the bracket rather than a single number.

Analysis memos are named by filename below. They are working files kept
alongside the scripts and are not tracked in this repository; the numbers that
matter are reproduced here.

Configuration-specific pins live with their configuration, not here:

- `scripts/production_stance_provenance.md` — `compare_sim1d_es1.PARAM_OVERRIDES`
- `scripts/golden_baseline_provenance.md` — `baseline_sim1d.BASELINE_PARAM_OVERRIDES`
- `scripts/ladder_operating_provenance.md` — `run_mechanism_ladder.ES_OPERATING`

---

## `initial_condition_defaults`

**`nn0 = 2.0e13` cm^-3 — ASSUMED.** A representative pre-shot neutral
background, so a bare `LAPDSim1D(...).run()` starts from a physical fill. It
replaced a `1e9` near-vacuum value that only ever made sense as a seed for the
neutral equilibration. The equilibrated path does not read it, so this default
can move without disturbing any equilibrated run.

**`Te0 = 0.21` eV — DERIVED.** Just above the exact bundled He ADF11 low-Te
edge (~0.200092 eV), below which the rate lookups clamp.

**`Ti0 = 0.026` eV — DERIVED.** ~300 K, essentially the fill temperature, and a
hair above `Ti_floor` so the raw-stage validator's strict `Ti0 > Ti_floor`
holds.

**`Tn_fit = 0.1` — FITTED, DEPRECATED.** A fitted neutral collision temperature
for the legacy IAEA fits. Superseded by the single cold-gas `Tn_K = 300` K.

## `geometry_defaults`

**`Rp = 15.0` cm — MEASURED.** The LAPD plasma-column radius. Before it was
made the default, every campaign run overrode it per-run.

**`Rm = 50.0` cm, `Lm = 2000.0` cm, `cathode_anode_gap_cm = 50.0`,
`plenum_length_cm = 100.0`, `collector_length_cm = 100.0` — MEASURED**
machine dimensions.

## `floor_defaults`

**`Ti_floor = 0.02585` eV — DERIVED.** 300 K in eV. Relaxed from 0.1 eV once
the only consumer that required 0.1 eV — the retired legacy IAEA CX table —
left the default path. Every remaining Ti consumer (`kappa_par_ion`, pressure,
sound speed) needs only `Ti > 0`.

**`Te_floor = 0.1` eV — ASSUMED, numerical.** Below the 0.2 eV ADF11 edge so
the afterglow can cool. It is a positivity floor, not a physical temperature.

## `neutral_source_defaults`

**`S_gp = 3400` sccm — FITTED.** The one calibration constant of the puff
model: the sccm-versus-drive-voltage relation of the valve is uncalibrated, so
the level cannot be read off the hardware. Everything else in the waveform is a
hardware timing. It feeds back on the discharge through S_gp -> ne -> current,
so it cannot be calibrated independently of the cathode power balance.

**`gas_puff_mode = "square"` — MEASURED shape.** The valve is driven by a
square voltage pulse from the same trigger that closes the cathode circuit and
held for the discharge; the supply line (45 PSI, 1/4") is hydraulically stiff,
with conductance and stored inventory orders of magnitude beyond the delivery,
so the delivered flow is flat with only piezo-opening and entry-transit edges.

**`gas_puff_rise_center_s`, `gas_puff_rise_width_s`, `gas_puff_close_lag_s`
= 5e-4 s — MEASURED, hardware-boxed** to ~0.5-1 ms. Not fit knobs.

**`gas_puff_profile = "cosine_pipe"`, `gas_puff_z_cm = 60.0` cm,
`gas_puff_throw_cm = 100.0` cm — DERIVED from geometry.** The physical source
is a small pipe at the chamber wall about 10 cm in front of the anode
(anode at 50 cm, hence 60), pointing radially inward with a Lambertian outlet;
the throw is of order the chord across the chamber, ~2*Rm. Neither centre nor
width is tunable.

**`S_pump_L = S_pump_R = 4000` L/s — ASSUMED.** The source side was previously
2000. Matching them reflects that the plenum aperture, not the pump speed,
throttles the source-side rate.

**Gas-puff clump density (motivating `gas_puff_local_ionization_fraction` and
`beam_clump_enhancement`) — DERIVED, boxed** at ~1-2e15 cm^-3 from the 45 psi
line, 1/4" choke and KF40 jet. Both parameters default to their off values.

## `timing_defaults`

**`tau_neutral_prebreakdown = 0.0` — MEASURED (device fact), boxed.**

The LAPD fires ONE global trigger: the capacitor bank is connected and the gas
puff starts at the same instant. The bank-connect step is directly visible in
all 64 shots as a single-sample 0 -> full-bank voltage step. There is no
interval in which neutrals accumulate with the drive withheld, so the model
must not run one. This is a statement about the DEVICE, which is why the value
lives in the config defaults rather than in a campaign override.

Superseded: a 0.002 s default, a run-sequencing artifact with no hardware
counterpart, which sat at the scale of a whole bracket step in the machine
timing comparison. It was measurably inert on the reference configuration —
across those 2 ms the neutral inventory built 0.031% and the plasma changed
0.000%, sitting at the `ne0 = 1e9` seed, because the 25 ms neutral
equilibration had already made the fill — so removing it was a pure 2 ms
sequencing shift. Validated against a pre-registered ignition gate of
0.591 +/- 0.010 ms: ignition moved 2.5913 -> 0.5968 ms with every scored stage
row unchanged except peak current 3017 -> 3016 A.

**`tau_prebreakdown = 0.05` s — MEASURED, hardware-boxed.** Not a knob.

**`equilibration_gas_puff_on_s = None` — historical default.** `None` makes the
equilibration inherit `tau_discharge` as its per-cycle puff window, a double
duty with no physical basis. The machine's total gas-puff pulse width is an
independent MEASURED quantity — see `scripts/production_stance_provenance.md`.

## `model_mode_defaults`

The shipped selectors — `hyperbolic_wave_speed = "adiabatic"`,
`ionization_birth_energy_model = "conservative"`, `operator_splitting =
"strang"`, `implicit_heat_scheme = "tr_bdf2"` — are DERIVED correctness and
accuracy choices, not calibrations: the signal speed matches the implemented
gamma=5/3 energy system, births book no unphysical electron thermal energy, and
the split step is second order. The regression fixture pins the historical
first-order values instead; see `scripts/golden_baseline_provenance.md`.

### Transient DVM neutral arm (`neutral_kinetic_dvm_*`)

All six keys are inert under the shipped `neutral_model = "moment"`; they are
read only by the K2a transient velocity-grid arm.

**`neutral_kinetic_dvm_cadence_s = 2.5e-5` — ASSUMED, and PROVISIONAL.** The
neutral clock's tick. This value was NOT selected from an accuracy study: the
multirate convergence measurement that would select it has not been run, and no
result may present this cadence as accuracy-chosen. It is a conservative
placeholder inside the `25-50 us` band the neutral-architecture assessment
named for ordinary discharge evolution, chosen because the split implicit step
is unconditionally stable, so the cadence buys accuracy rather than stability.
Honest bar: an ordinary discharge interval only; a rapid source transient or a
hot charge-exchange transient may need a shorter tick, and the afterglow may
tolerate a longer one. Replace with the measured value when the convergence
study lands.

**`neutral_kinetic_dvm_nvz = 48`, `neutral_kinetic_dvm_nvp = 12` — ASSUMED.**
The velocity-grid resolution, matched to `neutral_kinetic_nvz` /
`neutral_kinetic_nvp` so the transient arm and the K4a quasi-static engine are
compared on one grid rather than on two. `nvz` must be even (an odd count
places a bin at exactly `v_z = 0`). Honest bar: measured consequence, not a
free choice — on this grid the operator's discrete equilibrium sits about
2.4e-2 (density split) and 2.4e-2 (temperature) away from the continuum
Maxwellian at the 300 K wall, because the shared axis is stretched to hold the
10 eV charge-exchange tail and a 300 K gas then occupies only the few bins
inside `v_fine`. Refining to 96x32 takes the temperature offset to 8.4e-3 while
the density split converges to ~2.1e-2. Recorded by
`scripts/verify_sim1d_k2_dvm.py`, gate L4.

**`neutral_kinetic_dvm_accommodation = 1.0` — ASSUMED, boxed.** Thermal
accommodation coefficient of the chamber surfaces. Unity reproduces the
assumption the fluid neutral model already makes everywhere (every wall return
is a 300 K re-emission), so the arm's default introduces no new surface
physics and the arm is comparable to the moment model on this axis. It is a
surface property, never a fit parameter: it is not to be adjusted to move a
residual. Honest bar: helium on technical (unbaked, oxidized) stainless steel
is reported well below unity in the literature, so unity is the conservative
END of a bracket, not a measurement. The incomplete-accommodation arm is the
A/B; where data cannot pin the coefficient, the bracket over it is the claim.

**`neutral_kinetic_dvm_elastic = "phelps_iso"` — DERIVED.** Whether the
polarization-elastic ion-neutral channel is carried alongside charge exchange.
Derived from the requirement that the arm not silently drop physics the fluid
model has: the arm supersedes the fluid ion-neutral collision family whole, and
that operator's momentum-transfer cross section is `Qi + 2 Qb`, so carrying
only the backscatter `Qb` would delete the `Qi` half. `"phelps_iso"` restores
it as a mass-matched relaxation toward the local ion Maxwellian at HALF the
Phelps isotropic rate. The one-half is itself DERIVED, not fitted: a BGK
full-replacement event transfers the whole `m (v - u_i)`, which is the correct
weight for backscatter (`mu (1 - cos th) g` at `cos th = -1`, `mu = m/2`, is
`m g`) but exactly twice the isotropic angular average `<1 - cos th> = 1`,
i.e. `mu g = m g / 2`. The factor is the equal-mass reduced-mass ratio
`mu/m = 1/2` and nothing else. With it the arm's effective momentum-transfer
rate is `k_b + 0.5 k_iso`, identical to the superseded fluid operator
`phelps_momentum_transfer_rate_cm3_s`, so the kinetic and moment arms are an
honest A/B on the SAME collision strength rather than a 2x change of the
elastic channel smuggled in with the discretization. Both channels use the
same archived LXCat cross sections as the moment-closure operator, so nothing
new is fitted. `"off"` is the declared A/B arm. Corrected 2026-08-05 (the K2a
build shipped the unhalved rate); the correspondence is measured by
`scripts/verify_sim1d_k2_dvm.py`, gate C5.

**`neutral_kinetic_dvm_tn_feedback = False` — ASSUMED (a stance, not a value).**
Off means the fluid keeps its fixed cold-gas neutral temperature where it needs
one, exactly as today, while the arm reports its measured `Tn(z)` as a
diagnostic. Default-off is what makes the assumed-300 K versus measured
comparison a clean A/B rather than a confounded change of two things at once.

## `fudge_factor_defaults`

**`atomic_rate_model = "adas"` — MEASURED/published inputs.** The OPEN-ADAS GCR
'96 effective coefficients (`cablp/vars/adas`, see its README) are used as
published. They are citable inputs, **not calibration knobs**: a residual misfit
belongs to the plasma state and transport, not to the rate coefficients.
Consequently all `b_*` rate and cooling scale factors ship at 1 and are inert;
they remain readable so the `"janev"` A/B arm and the "set 0 to disable a term"
diagnostics still work.

**`alpha_isat = 0.6065306597126334` — DERIVED.** `exp(-1/2)`, the Bohm
presheath density ratio.

**`sigma_in_cm2 = 5.0e-15` cm^2 — literature (MEASURED).** Read only by
`sigma_in_model = "constant"`; the shipped `"phelps"` model uses the Phelps
He+/He cross section directly.

**`heat_flux_limiter_f = 0.3` — ASSUMED.** The free-streaming fraction is not
pinned by any measurement; the limiter is a closure-family instrument and the
coefficient is properly reported as a bracket. The campaign stance uses a
different value — see `scripts/production_stance_provenance.md`.

**`source_surface_area_scale = 1.8` — FITTED, DEPRECATED.** A 0D artifact that
stood in for un-separated cathode/anode I_sat. Never consumed by the resolved
geometry.

## `cathode_defaults`

### Circuit

The circuit constants come from ONE V0-pinned constrained refit over four
discharge settings: V0 fixed per setting at its measured pre-shot reading, with
C, R and L shared across the four (N = 1952, window 0.3-19.8 ms). Design
conditioning 4.7. Memo: `circuit_constrained_refit.md`.

It replaced a free 4-parameter fit to a single trace, which was near-singular:
corr(V0, R) = 0.997, and R swung 1.9-5.7 mOhm with the fit window, so its
formal +/-0.079 mOhm bar was meaningless and its 0.14 V rms was in-sample. The
defect was invisible on the single setting it was fitted to and appeared only
on transfer: reconstructing measured plateau V_dis, the old parameterization
left residuals -0.136/+6.329/+5.677/+5.786 V across the four settings, against
+0.010/+0.139/-0.053/-0.309 V for the corrected one.

**`R_comp = 7.2244e-3` Ohm — MEASURED.** Jackknife bar 7.213 +/- 0.043 mOhm;
the four settings (2965-4411 A) agree to 1.8%. Supersedes 5.72e-3 from the
free fit. It does not set the discharge current — the emission ceiling does;
at the reference operating point `dlnI/dlnR_comp` is -0.0034 frozen and -0.017
coupled.

**`C_bank_F = 9.5` F — MEASURED, hardware-bounded.** Fit value 9.5187 +/- 0.66 F.
Hardware: 10 IGBT switches x 2 minibanks x 35 Nippon Chemi-Con 36DY cans
(12,000 uF, 200 V) = 700 cans, NOMINAL 8.40 F. Per-can tolerance -10/+50%, so
the allowed total is [7.56, 12.60] F and the nominal sits near the FLOOR, not a
ceiling. With N = 700 the random part of the sum is only 0.65% (0.055 F), so
the trace measures the bank ~12x more precisely than the nameplate bounds it.
9.5 F is an ordinary interior value of the allowed band (39th percentile).

*Retraction.* An earlier record asserted that the fit's effective ~8.9 F was
anomalous "even though the hardware bank is nominally <= 4 F", and attributed
the gap to ~7 V of unexplained slow EMF recovery. There was never an anomaly.
The 4.2 F figure counted one minibank per switch — a factor-of-2 miscount — and
the 8.40 F nominal it was compared against is a near-floor, not a limit. 4.2 F
is excluded twice over (5 sigma by the fit, and below the band floor). The
"~7 V of slow EMF recovery" was an artifact of leaving V0 free in a
near-singular fit and dissolves once V0 is pinned to its measured pre-shot
value. The historical 8.9 F was inside tolerance all along; no past run is
invalidated and there is no caveat to carry.

**`R_comp` and `C_bank_F` are one joint fit and must move together.**
7.2244 mOhm is self-consistent with C = 9.5187 F; pinning C = 8.40 F instead
would pair with R = 7.079 mOhm.

**`L_parasitic_H = 8.1e-6` H — DERIVED from measurement, bracket 7.6-8.4 uH.**
Memo: `l_parasitic_reconciliation.md`. Two instruments with disjoint time
windows and no shared fitted parameters agree at ~8 uH:

- the flyback VOLT-SECOND BALANCE over the current fall, `L = int V dt / dI`,
  giving 7.2-8.4 uH. The switch hardware paper (Pribyl & Gekelman, RSI 75, 669
  (2004)) confirms the fall is a real freewheel, and this arm is INVARIANT to
  the circuit constants: the fall branch of `scripts/fit_circuit_edges.py`
  drives the loop with `emf = -V_meas` and touches neither V0, R nor C. Its
  fall-only answer (~8.2 uH) was unchanged, to better than the scan resolution,
  across the circuit correction above — the strongest single number here.
- the edge ODE fit over the current RISE (same script, driven by the measured
  V_dis with the corrected V0/R/C): 7.6 uH at 38 A rms. Joint rise+fall
  8.2 uH, +10%-rms band 7.9-8.5 uH.

The plateau refit independently prefers 8.06 uH, but the plateau is nearly
L-blind (dI/dt ~ 0 there): its jackknife bar is 6.7 +/- 2.5 uH and the fitted
value ranges 2.1-9.4 uH with the window. The instruments cannot discriminate
8.06/8.1/8.23, so 8.1e-6 is adopted — the value the regression fixture and the
pre-regression config default already carried, collapsing three distinct
recorded values to one.

*Retraction.* An earlier record said 6.6e-6 was left "pending" a better-posed
edge fit that "boxes 15-25 uH". Both halves were false. 15-25 uH was never the
edge fit's answer — it is `L = tau_fall * R_load` with the plasma treated as a
CONSTANT RESISTOR, and the measured V_dis collapses 16x within 0.2 ms during
the fall (88.8 -> 15.5 -> 9.3 -> 5.3 V at t = 20.0/20.2/20.5/21.0 ms), so a
constant ~50 mOhm load is off by an order of magnitude. That estimate was
retracted the day `fit_circuit_edges.py` was written to test it and refuted it;
the edge instrument excludes 15-25 uH at 4.6-7.1x its minimum residual.

*Correction of a correction.* A later note claimed the docstring "claimed
8.1 uH and contradicted the code" and resolved it toward the code. That was
backwards. The docstring was right and the code was wrong: the default was
8.1e-6, was overwritten to 6.6e-6 in the same three-line hunk that installed
the since-corrected `R_comp` 5.72e-3 and `C_bank_F` 8.4, and 6.6e-6 was the
orphan of the same retracted near-singular free fit — a regression, never a
considered choice. The docstring was the surviving true record.

*Honest limit.* 6.6e-6 is still INSIDE the plateau refit's jackknife bar
(6.7 +/- 2.5 uH). The claim is that 6.6 has no evidence behind it while 8.1 has
two independent lines — NOT that 6.6 is excluded. The move is a consistency and
provenance correction, not a physics correction: L is inert for every scored
quantity (the plateau inductive term moves 0.055 V mean against a -5.7 V
fingerprint signal, under 1%), and the measurable consequences are confined to
unscored reported fingerprints — t90 +0.05 to +0.11 ms and ignition +0.02 to
+0.07 ms, both toward the measurement.

**`R_comp_partition = 1.0` — all external.** Bit-exact with the historical
behaviour. Memo: `circuit_vdis_three_rung_read.md`.

*Retraction.* An earlier docstring claimed the internal part "lowers the
current, which RAISES V_dis". That is algebraically FALSE — `x` cancels
identically from the loop equation — and it made a campaign design look more
promising than it was. The corrected three-part statement (dynamically inert,
observationally active, therefore a calibration knob) is in the `config.py`
docstring, where it belongs: it is a fact about what the parameter does, not
about which value was chosen.

**`R_mesh_ohm = 0.0` Ohm — MEASURED bound, consistent with zero.** The measured
internal resistance at the reference operating point is
`R_int = (V_dis_meas - V_b)/I = 0.0 +/- 0.3 mOhm`, which is BELOW the
0.5-1.5 mOhm physical bracket for the anode mesh (0.64 mm Mo wire on a 2.58 mm
pitch, rising with anode temperature). A temperature-dependent `R_mesh(T_anode)`
from an anode power balance was designed and declined; only a constant value is
implemented.

### Emission

**`C_R = 29.0` A cm^-2 K^-2 — literature nominal for LaB6**, in the effective
sense described in the docstring, not the Richardson-Dushman universal 120.
`C_R` and `cathode_Ts_base_K` are degenerate along one flat direction of
roughly 100 K of surface temperature per e-fold of emission, so only one of the
two may carry a calibration. The campaign stance calibrates `C_R` and leaves
`cathode_Ts_base_K` at its measured value; see
`scripts/production_stance_provenance.md`.

**`cathode_Ts_base_K = 1910.0` K — MEASURED**, and not to be tuned. The standby
surface temperature is an operational machine setpoint. Per-setting values are
in `run_mechanism_ladder.ES_OPERATING`; see
`scripts/ladder_operating_provenance.md`.

**`T_s = 1998.15` K — MEASURED setpoint.** Under `power_balance` this is only
the static-model fallback; the live input is `cathode_Ts_base_K`.

**`phi_wf = 2.869` eV — FITTED**, the contaminated shot-start work function.
**`cathode_phiwf_clean_eV = 2.809` eV — FITTED**, the per-shot-accessible depth
of the re-adsorbed layer, NOT the literature clean-LaB6 value. Both are only
meaningful together with the `cathode_schottky` flag, which shifts the
effective barrier.

**`cathode_Ts_fwhm_cm = 28.0` cm — MEASURED/DERIVED.** The emission footprint
measures 28.8-31.2 cm at the diagnostic ports; back-extrapolating the axial
broadening to the cathode gives ~27.8 cm, and radial transport argues for the
steeper end. The implied centre-to-edge temperature drop is of order
150-200 K.

**`R_cath = 15.0` cm — the plasma-channel radius, not the cathode disc.** The
physical cathode radius is 19 cm. The gaussian emission profile should be used
with the physical radius; the default matches `Rp` instead.

**`cathode_heat_capacity_J_per_K = 120.0` — ASSUMED (hand-tuned).** It shapes
only the ramp timescale; the steady state is independent of it. Physical scale:
the thermal skin depth over a ~20 ms transient is 0.3-0.5 mm of LaB6, a few
J/K, well below the disc's bulk heat capacity of hundreds of J/K.

**`cathode_conduction_W_per_K = 1200.0` — ASSUMED (hand-tuned), pending a
heater-current fit.** Physical scale: quasi-static `kappa*A/delta` for LaB6 is
~10 kW/K at a 0.4 mm skin depth, and the effective value over a ~20 ms
transient is lower. It is load-bearing: at 0 (the pure-radiation limit) the
bombardment feedback gain d(P_ion)/dT_s, of order kW/K through the emission
loop against a ~230 W/K radiation+emission stiffness at ~250 kW bombardment
power, runs the discharge away to ~13 kA. Around 2000 W/K reproduces a ~110 K
plateau rise at the measured bombardment power. The campaign stance uses a
different value; see `scripts/production_stance_provenance.md`.

**`cathode_cleaning_E_th_eV = 20.0` eV — DERIVED, bracket 18-26 eV** from
He -> O kinematics for chemisorbed oxygen.
**`cathode_cleaning_sigma_cm2 = 3.5e-16` cm^2 — FITTED.**

**`cathode_desorption_energy_eV = 3.0` eV — ASSUMED.** No independent bar is on
record and no memo was written; the value is latent rather than load-bearing.
It is read only when `cathode_desorption_prefactor_per_s > 0`, which ships `0`,
so at the shipped defaults the thermal-desorption exponential is never
evaluated and this number is inert. It acquires a bar only when the ads/des
arm (`cathode_surface_model = "ads_des"`) is first exercised with a positive
prefactor; a bracket must be established before any result leans on it.

**`cathode_jet_R_N = 0.5`, `cathode_jet_R_E = 0.2`, `anode_jet_R_N = 0.5`,
`anode_jet_R_E = 0.25` — literature-boxed (MEASURED class).** Particle and
energy reflection coefficients of the Eckstein/Thomas class: the cathode pair
is the mid-box for He -> LaB6, with the B-rich versus La surface-termination
spread as the honest uncertainty; the anode pair sits at the He -> Mo
heavy-target corner. Not fit knobs. Both jets default off.

### Beam

**`beam_excitation_energy_eV = 21.218` eV — MEASURED**, the He 2^1P excitation
energy.

**`b_beam_excitation = 0.0`** ships off. Under `"2p_scalar"` a value of 1.0
books the 2^1P channel alone and ~1.4 was a historical ASSUMED estimate of the
full singlet manifold; the measured replacement is
`beam_excitation_model = "manifold"` (Ralchenko et al. 2008 singlet manifold),
which over 60-180 eV gives 1.65-1.75x the 2^1P events and 1.71-1.81x its
radiated power. Both are inert under `beam_deposition_model = "csda"`, which
always uses the measured manifold.

**`heating_anomalous_tail_energy_eV = 75.0` eV — ASSUMED, NEVER fitted.** The
quasilinear plateau energy is a kinetic quantity a fluid model cannot pin, so
the BRACKET is the claim: the central arm is 75 eV with 30 and 150 eV as the
bracket arms, and all three are reported together. Inert unless
`heating_anomalous_transport = "tail_walk"`.

**`beam_deposition_smoothing_cm = 0.0`** ships off (bit-exact). The campaign
stance uses a physical straggling width; see
`scripts/production_stance_provenance.md`.

## `physics_fit_defaults`

**`Tn_K = 300.0` K — MEASURED.** The single cold-gas neutral temperature, also
the Phelps operator's `T_eff` input. It ended a term-by-term mix of `Tn_K` and
the fitted `Tn_fit`.

**`heat_picard_iterations = 2` — DERIVED.** A positive value is required for
`tr_bdf2` + `strang` to express second order; 0 freezes the conductivity and
caps the whole step at first order.

## `timestep_defaults`

**`surface_loss_floor_exempt_rtol = 1e-3` — DERIVED from measured scales.**
Instrumented on a floor-pinned afterglow state: a pinned cell hovers at ~5e-6
relative margin (the clip plus one step of re-heating residue, not float
round-off), while every healthy drained cell sampled across drive and afterglow
sits at >= ~2e1 relative. 1e-3 splits those scales by more than two decades on
each side; physically it means Te within 0.1% of `Te_floor`.

**`dt_min_lock_max_steps = 250000` — DERIVED from a census of saved runs.**
Memo: `scripts/dtmin_census_runlengths.txt` (2026-08-05). All 209 result h5
files then present in `cablp/scripts/` were scanned for the per-step
`active_constraint` label, which under the pre-2026-08-05 semantics read
`"dt_min"` on exactly the steps that were clamped. 80 files clamp at least
once, and the two populations separate cleanly on CONSECUTIVENESS:

- self-releasing episodes (79 files), median run length 1–2 steps, with the
  longest at **23296 consecutive steps** in `es1_r5_hflim01_exp2.h5` (t =
  2.002–2.005 ms, `dt_surface_loss` small but strictly positive throughout).
  That run released, completed, and was scored;
- one permanent lock, `es1_r5_f01_ag26ms.h5`: a run of 36690 steps still open
  at the last recorded step, with `dt_surface_loss` exactly `0.0` on 1446 of
  them, truncated at 22.3 ms against a 26.0 ms target.

The pre-registered rule was **≥ 10× the longest run in any completed
(non-pathological) run**, i.e. ≥ 232960; 250000 is the round number above it.
The margin buys immunity for the known-good family: since the guard is
default-ON, aborting a healthy arm would be the worse failure, and a genuine
lock is unbounded by construction so a large threshold costs only detection
latency, never detection itself.

The remaining timestep parameters (`cfl`, the `*_dt_fraction` limits, growth
and retry factors) are ASSUMED numerical-control values with no measurement
behind them.
