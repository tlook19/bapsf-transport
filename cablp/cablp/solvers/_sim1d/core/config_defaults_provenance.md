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

**`nn0_profile = None`, `nn0_annulus_profile = None` — INSTRUMENT, NO VALUE.**
The shaped initial neutral fill (`neutral_initial_profile` flag) carries no
number and can acquire none: its content is a per-cell array of absolute
densities [cm^-3] computed OUTSIDE the solver, and `None` is the only defensible
default because a shaped initial condition has no shape to inherit. Nothing here
is measured, derived, fitted or assumed — the capability is plumbing, and every
physical quantity that decides the array lives in the producing script, which is
where the provenance for a given run's fill belongs.

The value producer of record is **`scripts/sp3_build_nn0.py`** (the sp3 foot-shape
IC arm): it composes the shipped-convention base fill with the accumulated
first-flight gas-puff lobe over a foot duration, spread by one of two declared
kernels, and writes the array plus a provenance header into an `.npz`. The
`.npz` header — not this file — records that run's base, throughput convention,
foot duration, kernel, mean free path and cross-section source. A different
construction is a different instrument and carries its own header; that is the
point of the key having no value here.

Two rules keep the instrument honest and are enforced at construction rather
than documented here as a convention: the scalar `nn0` must be `None` when the
flag is armed (no silent precedence between a scalar and an array claiming the
same field), and `neutral_equilibration` / `restart_from` are refused (both
replace `nn` after construction, so a shaped fill under either would be built
and discarded without a trace).

**`Te0 = 0.21` eV — DERIVED.** Just above the exact bundled He ADF11 low-Te
edge (~0.200092 eV), below which the rate lookups clamp.

**`Ti0 = 0.026` eV — DERIVED.** ~300 K, essentially the fill temperature, and a
hair above `Ti_floor` so the raw-stage validator's strict `Ti0 > Ti_floor`
holds.

**`Tn_fit = 0.1` — FITTED, DEPRECATED.** A fitted neutral collision temperature
for the legacy IAEA fits. Superseded by the single cold-gas `Tn_K = 300` K.

## `geometry_defaults`

**`Rp = 18.415` cm — MEASURED.** The LAPD plasma-column radius, identified
with the exposed cathode aperture: the graphite front panel's 14.5 in opening
in front of the 15.0 in LaB6 disc (caliper, 2026-08-17). Mapping the aperture
along the field to the column is ASSUMED 1:1. The previous default 15.0 was a
fit that also conflated this radius with the emitting radius `R_cath`; both
moved onto the measured aperture in the L2 geometry rebaseline, and the golden
fixture pins 15.0 explicitly so the regression anchor does not follow.

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

**`gas_puff_delivery_fraction = 1.0` — STRUCTURAL IDENTITY, no value claimed.**
The shipped default is the identity element of the decomposition it enables,
not a measurement or a fit: at `1.0` the injected flow is `S_gp` exactly, so
every configuration that predates the key is bit-exact under it. The key
exists because `S_gp` had been carrying two distinct quantities at once — the
flow the valve delivers, which is measured, and the share of that flow that
enters the modelled volume, which is not — and a single lumped constant cannot
report either honestly. Splitting them is what makes the measured half
citable; moving the value off `1.0` is a separate, registered calibration
event and is not part of this decomposition.

The complement `1 - f_gp` is a real gap and not a modelling convenience. The
inventory read `scripts/fa0_neutral_budget.txt` closes the neutral throughput
budget on the production ES1 stance and finds the pump venting only 2.0-2.6%
of the delivered puff across every window, so the model has **no sink capable
of absorbing a 30-40% shortfall**: whatever the complement represents (valve
plenum, transit line, entry aperture, gas that never reaches the column) it is
outside the modelled volume by construction, which is exactly why it belongs
in a delivery fraction rather than in a pumping speed. Its eventual class will
be FITTED-with-a-measured-envelope: the measured half is the per-valve plateau
flow from the censored valve fit (`scripts/flowcal_censored_fit.txt`, which
gives `A(76.4 V) = 9.01` slm at the ES1 operating point, systematic envelope
`[8.80, 9.50]` slm, plus 4.8% pass-to-pass valve reproducibility), and the
fitted half is whatever `S_gp` the discharge calibration lands on, so the
implied `f_gp` is a ratio of one to the other and inherits the envelope of
both. The bracket registered for the calibration leg is
**`f_gp = [0.578, 0.650]`, full envelope `[0.556, 0.667]`**. Note that only
its lower end follows from the artifact above — `5200/9010 = 0.577` — while
`0.650` implies a per-valve delivery near 8.0 slm, which that fit's own
envelope does not reach; **reconciling the upper end is owed by the
calibration leg and is not established here.** Nothing in this repository is
calibrated to any of it yet, and **the bracket, not a point, is the claim** if
the leg cannot pin it.

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

**`neutral_kinetic_dvm_exchange = "cauchy_chord"` — DERIVED (both arms), the
default is a REPRODUCIBILITY stance.** Which closed form carries the
column/annulus zone exchange and the radial-wall rate. Both arms are derived,
neither is fitted, and the choice between them is not a free parameter — they
are two different pieces of integral geometry applied to the same cell, and one
of them is the right one:

- `"cauchy_chord"` is the shipped K2a transcription of `KN2Zone`: the
  three-dimensional Cauchy mean chord `4V/S = 2 (Rm - Rp)` evaluated at the
  perpendicular speed `vp`, with one surface encounter split between the two
  cylinders as `Rp/Rm : (1 - Rp/Rm)`.
- `"geometric"` is derived from the fact that crossings of two COAXIAL
  cylinders are a two-dimensional problem — the axial coordinate never enters
  the radial dynamics — so the mean chord is the planar Cauchy chord
  `pi A / P = pi (Rm - Rp) / 2`, and the encounter splits between the two
  circles in proportion to their PERIMETERS, `Rp : Rm`. That gives
  `nu_a->c = 2 vp Rp / (pi (Rm^2 - Rp^2))`,
  `nu_a->wall = 2 vp Rm / (pi (Rm^2 - Rp^2))` and, on the true cell volumes,
  `nu_c->a = 2 vp / (pi Rp)`.

Honest bar: `"geometric"` is the one that reproduces the geometry. It is
measured against the reference geometry by the free-flight billiard probe in
`scripts/k2_dvm_exchange_measure.py` — the committed E2 ray tracer with
collisions, sources, pumping and accommodation all switched off, so the
measurement is a property of `(Rp, Rm)` and the velocity and of nothing else —
and the record is `scripts/k2_dvm_exchange_measured.txt`. Independently,
`nu_c->a = 2 vp / (pi Rp)` averages over a Maxwellian to `vbar / (2 Rp)`,
exactly the free-molecular rate `neutrals.neutral_zone_exchange_conductance`
already carries in the fluid arm; the `"cauchy_chord"` form does not reduce to
it. The correction factor is `4 Rm / (pi (Rp + Rm))` on the exchange channels
and `4 Rm^2 / (pi (Rm^2 - Rp^2))` on the wall channel, so it is
CELL-dependent, not a constant.

The DEFAULT nonetheless stays `"cauchy_chord"`, because the pre-registered
acceptance gate that would have promoted `"geometric"` was MISSED
(`scripts/k2_dvm_exchange_acceptance.txt`, a reduced-statistics E2 rerun with
both arms scored against one reference). At this device's main-column ratio
`Rp/Rm = 0.3` the two errors in the shipped exchange rate nearly cancel — the
mean chord is too long by `4/pi` and the return fraction too large by
`(Rp+Rm)/Rm` — leaving the exchange channels only -2.1 % apart; the wall
channel, where they do not cancel, moves +39.9 %. The gate required the worst
matched-time mid-machine `n_ann` deviation to shrink 3x and it shrank 1.32x
(+184.97 % to +139.80 %), so the mid-machine annulus divergence is NOT caused
by the zone-rate coefficients. The two arms are a declared A/B; promotion is
gated on evidence that has not been produced.

**`neutral_kinetic_dvm_annulus_flights = "rates"` — DERIVED (both arms), the
default is a REPRODUCIBILITY stance.** How the annulus zone's wall interaction
and radial exchange are taken. Neither arm has a fitted number in it.

- `"rates"` is the shipped algebraic treatment: the
  `neutral_kinetic_dvm_exchange` frequencies `nuw` and `nuxp` in the implicit
  march, with the annulus advected by the same upwind sweep as the column. A
  per-cell rate is a memoryless process, so the flight time it implies is
  EXPONENTIAL — `mean^2/var = 1`.
- `"bounded_chord"` replaces the annulus-side wall and annulus-to-column rates
  with the K1b jump kernel's three flight classes. The mean chords are the
  cosine-weighted class means computed numerically from the local `(Rp, Rm)` by
  `kinetic_neutrals.annulus_chord_classes`, the same function `KN2ZoneJump`
  uses: outer wall to inner surface with the view factor `Rp/Rm` at `c_wi`,
  outer wall to outer wall at `c_ww`, inner surface outward at `c_io`. On the
  production geometry's duct (`Rp = 15`, `Rm = 50`) they are 37.46, 69.99 and
  37.46 cm. Their sampled distributions carry `mean^2/var` 196.0, 10.5 and
  195.7 — the measured chord statistics of the duct, ten to two hundred times
  narrower than the exponential the rate arm implies, which is the whole
  reason the arm exists. The classes are checked against the two-dimensional
  mean-chord theorem `pi (Rm - Rp) / 2`, which nothing in their derivation was
  fitted to, by gate J1 of `scripts/verify_sim1d_k2_dvm.py`.

Honest bar: the transient-ification is NOT the steady K1b kernel. It keeps the
axial displacement per flight deterministic and equal to `v_z c / v_perp`, and
the mean flight time equal to `c / v_perp`, but the flight is TERMINATED at
that mean rate rather than at a fixed elapsed time, and the in-flight atom is
held at the flight's midpoint rather than spread along its path. An exact
fixed-duration delay is not representable in an Eulerian `f_a` without an age
coordinate, whose cost on the production grid is prohibitive; what is
recovered exactly is the per-flight axial step and hence the annulus's axial
dispersion, which is the quantity the exponential tail was over-carrying. The
E2 read of both arms against the step-face ray-traced Monte Carlo on
`scripts/es1_k3a_cal2_nx240.h5` is `scripts/k5build_report.txt`; the numbers
there are RATIFICATION-PENDING and nothing in the kernel was tuned to them.

The DEFAULT stays `"rates"` because the frozen production arm ran on it and
must stay bit-reproducible; `scripts/k5_frozen_bitexact.py` is the check.

**`neutral_kinetic_dvm_tn_feedback = False` — ASSUMED (a stance, not a value).**
Off means the fluid keeps its fixed cold-gas neutral temperature where it needs
one, exactly as today, while the arm reports its measured `Tn(z)` as a
diagnostic. Default-off is what makes the assumed-300 K versus measured
comparison a clean A/B rather than a confounded change of two things at once.

**`neutral_kinetic_dvm_transfer_relax_fraction = 0.5` — ASSUMED (a numerical
safety margin, not a physical quantity).** The share of a cell's ion-energy
margin above its `Ti` floor that the tick-frozen coupling drain may consume in
one plasma step. It exists because the transfer is frozen for a whole neutral
tick while the plasma steps ~1e3 times inside it and can flip sign across one
tick: at the 2026-08-05 K3a crash the frozen drain was -1.57e10 erg/cm^3/s
against a margin of 0.746 erg/cm^3, an explicit e-fold of 9.5e-11 s, below the
run's `dt_min` of 1e-10 s — no admissible step existed. `0.5` leaves half the
margin for every other term in the same step; it was NOT selected by a
convergence study, and nothing in the model depends on the value being 0.5
rather than, say, 0.25 — a smaller value defers more transfer and a larger one
less, with the deferred amount conserved either way (`applied + debt ==
booked`). The physically meaningful statement is the one the cap makes
regardless of its value: the applied drain cannot carry a cell through its
floor inside a step. Raise the value only with the ledger's outstanding debt
reported alongside.

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

**`eta = 0.358` — MEASURED (hardware: anode mesh geometry).** Calipers on the
anode mesh (Tom, 2026-08-11): molybdenum wire diameter `d = 0.64` mm, square
weave, clear opening `a = 2.58` mm. The solid fraction follows from the
geometry alone,

    T = (a/(a + d))^2 = (2.58/3.22)^2 = 0.641989,  eta = 1 - T = 0.358011,

i.e. the shipped `0.358` to every digit it quotes.

**The exact match also settles which spacing convention the caliper number
used, and that check IS the verification.** Read instead as a
centre-to-centre pitch of 2.58 mm, the same wire gives
`eta = 1 - ((2.58 - 0.64)/2.58)^2 = 0.434589` — nowhere near the shipped
value. Only the clear-opening reading reproduces it, so `2.58` mm is the
OPENING and the pitch is `a + d = 3.22` mm. (Note the `R_mesh_ohm` entry above
calls 2.58 mm a "pitch"; on this check that wording is the loose one.)

Honest bar **+/- 0.01**, carried almost entirely by wire-DIAMETER variation:
at a caliper resolution of +/- 0.02 mm, `d` moves eta by `+/- 0.008` and `a`
by `-/+ 0.002` (opposite in sign, and the smaller term).

**Spacing NONUNIFORMITY contributes ~zero to the mean, exactly rather than
approximately.** For straight wires the open-area fraction factorizes,
`T = (sum of x-openings / L) * (sum of y-openings / L)`, and displacing a wire
within the plane conserves each sum — every wire blocks exactly its own
diameter wherever it sits — so the area-averaged transparency is INVARIANT to
where the wires are. The invariance fails only for wires that touch or overlap
(visually excluded on this mesh) and for out-of-plane weave at oblique
incidence, which does not arise here.

The value is geometric and assumes NORMAL INCIDENCE: the magnetized electrons
follow field lines perpendicular to the mesh plane, so the projected solid
fraction is the collection fraction, independent of energy and of current.

**`eta` is NOT overloaded: every consumer implements the one meaning.** The
symbol does double duty across modules, so the consumers were enumerated
before the docstrings were unified. Four classes, all of them the mesh solid
fraction:

1. **Geometric transmission, `1 - eta`** — the anode face's open fraction
   (`geometry.anode_open_fraction`, `1 - eta*(Ra/Rm)^2`), the neutral
   transparency (`physics/neutrals.py`, `physics/kinetic_neutrals.py`) and the
   parallel heat transmission across the mesh (`physics/conduction.py`).
2. **Bohm ion collection at the wires** — `sources.anode_collection_rhs` takes
   the collecting area as `eta * plasma_area_cm2` in each flanking cell, and
   `cathode.anode_circuit_sample` sums the two faces, which is the historical
   `I_i_a = 2*eta*I_i`. The `2` is the mesh's two faces, not a second quantity.
3. **Beam interception** — `eta*beam_bypass_fraction` in both cathode solvers
   (`J_anode`, `P_prim`, the bypass power) and the CSDA ray's anode-face event
   (`anode_eta`), where the wires remove `eta` of the gap-surviving flux and
   `1 - eta` transmits.
4. **Particle-loss bookkeeping** — `(1 + 2*eta) * I_i / qe`: the cathode face
   plus the mesh's two.

**The competing "anode area / cathode area" reading is excluded twice over,
quantitatively and structurally.** At the shipped stance `R_cath = 15` cm,
`Rp = 15` cm and `anode_radius_cm = None` — the mesh spans the chamber, with
`Ra >= Rp` enforced — so an anode-to-cathode AREA RATIO is `>= 1` and cannot
be `0.358`. Independently, class 1 uses the quantity as `1 - eta`, which is a
transparency only for a fraction: for a ratio `>= 1` it would be `<= 0`.

**`cathode_phi_c_cap_V = 1000.0` V — DERIVED from the atomic data's domain, and
a numerical REGIME guard.** Both are true of it, and the second is what it is
usually read as. It is not a device-physics voltage and no measurement of a
device bounds it — but it is also not a free numerical choice: **1000.0 V IS the
top of the tabulated He EII cross section** the beam deposition reads. From the
modules' own constants, `HE_EII_EPS_TOP = 40.671258069120896` times
`I_ion = 24.58738793623` eV is `1000.0000000000002` eV, and the shipped value
sits **sub-ULP below that edge** — relative offset `-1.75e-16` in the `eps`
space the guard actually tests (`-2.27e-16`, i.e. two ULP, in energy space).
With `HE_EII_EDGE_REL_TOL = 1e-12` the largest admissible cap is
`1000.0000000010` V: one part in `1e9` of headroom. **The cap is movable
DOWNWARD only.** The edge is INCLUSIVE by deliberate design
(`_cathode_solver_idriven.py:930-941`, `_beam_deposition.py:298-307`): at the
edge the clamped lookup IS the table's endpoint node and nothing is
extrapolated, while a larger excess is REFUSED with a `ValueError` naming the
table top rather than silently clamped onto an extrapolated cross section. The
relation is therefore `cap = HE_EII_EPS_TOP * I_ion`, and the data it is pinned
to is the same He atomic data the standing no-tuning policy protects — raising
the cap is not a re-choice of a number, it requires cross-section data above
1000 eV.

What it ALSO bounds — the regime reading — is the current-driven sheath solve:
the bracket ladder doubles `psi_top` looking for a
`psi` that carries the imposed current, and this is the ceiling at which that
search stops and the solve returns the solution AT the cap tagged
`regime = "capability_limited"` — the well-posed form of the inductive kick,
which the circuit then rides down at ~V/L. Since the returned-root fix
(2026-08-09) it also bounds the located J-root, not only the ladder's doubling
grid points, so a root at or above the ceiling now falls through to the same
branch; the two routes are indistinguishable in the exported result and both are
flagged by `{source,end}_phi_c_at_cap`.

*Honest bar: one-sided and exact.* Upward there is no bar to state and no
bracket to explore — the admissible headroom is `1e-9` relative, so the value is
its own upper bound. Downward the value is unconstrained by the data and the
solve is deliberately insensitive to it wherever the ceiling is not reached: a
free root below the cap is returned bit for bit independently of where the cap
sits. It is load-bearing only in the at-cap regime, and there it sets the
reported `phi_c` outright, so every `phi_c`-keyed consumer inherits it (notably
`E_tail` under `heating_anomalous_tail_energy_keying = "phi_c"`, where
`f = 1.0` puts `E_tail` on the He EII table's top edge to the last bit — which
is the same edge, reached from the consumer side, and why it is checked
inclusively rather than clamped). A result that spends material time at the cap
is quoting a guard AND the table's last node, not a device voltage.

Two facts recorded 2026-08-11, both MEASURED, both bearing on how much weight
the value carries:

- **At-cap exposure on the build leg is 40-65 % of saves** in the
  conducting-phase coverage arms (40.0 / 46.3 / 64.7 % of pre-breakdown saves
  across the three F2-family shots), and those saves carry 54-78 % of the beam
  channel's pedestal log-gain. The cap is not a latent bound in that regime; it
  is on the calibration path.
- **The ignition-frame sheath demand is at least 10 kV.** A trial run with the
  cap raised to 10 kV reported `phi_c` pinned at the *new* ceiling
  (`10000.000000000002` V) at the FIRST cathode solve — before the table-top
  guard refused the step. Raising the cap by 10x does not un-pin `phi_c`, so the
  at-cap regime cannot be exited by raising the ceiling even where the atomic
  data would allow it.

*Memo line:* adjudicated 2026-08-09 — accepted as-is, with the at-cap regime
flag (`{source,end}_phi_c_at_cap`, added the same day) as the standing
instrument for reading how much of a run rides it. The REGISTERED REVISIT
("if any arm is found riding the cap materially, the value must acquire a
bracket") TRIGGERED on 2026-08-11 and is resolved by the class correction above:
the arms do ride it materially, and the bracket that answer demands does not
exist upward, because the value is the data's edge rather than a choice.
Memos: `covcap_memo.md`, `covcal_efold_read.md`.

**`cathode_circuit_bound_object = "device_voltage"` — DERIVED; a selector, not
a magnitude.** It carries no number and no bar: it names which quantity
`cathode_circuit_voltage_bound` holds at the circuit-available voltage. The
derivation is the loop equation itself. What the source and the series
resistance can sustain is the DEVICE voltage `V_b = phi_c - phi_a + V_p`, and
that is the quantity the circuit integrates, so `"device_voltage"` is the
object the relation actually contains; `"phi_c"` is the proxy R1 shipped,
correct only where `phi_a` and `V_p` are negligible. The shipped value is
therefore the derived one and not a preference. `"phi_c"` is retained as a
declared A/B arm (it reproduces R1 bit for bit) so that results taken under
the R1 composition remain reproducible, exactly like the other closure
families. Honest bar: none applies to a selector; what a RESULT must state is
which arm produced it, because the two differ by `phi_a - V_p` in the returned
sheath drop and therefore in the beam birth energy keyed to it (measured on a
plateau-class point: 190.36 V vs 177.84 V at `phi_a = 12.90` V).

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

**`R_cath = 18.415` cm — MEASURED.** The cathode is a 15.0 in x 0.25 in LaB6
disc (R = 19.050 cm) held by a backside carbon ring against a graphite front
panel whose opening is 14.5 in, r = 18.415 cm (caliper, 2026-08-17); the
ring's lip fit on the disc covers exactly the annulus the panel hides, which
confirms the two measurements against each other. The EXPOSED APERTURE — not
the disc — is the emitting, collecting and conducting face, so `R_cath` is
that aperture. It equals `Rp` by measurement plus the assumed 1:1 field
mapping, no longer by the old coincidence of one fitted 15.0. The physical
disc radius 19.050 cm remains the right number for anything that is about the
whole body rather than the exposed face.

**`cathode_emission_profile = "uniform"` — the zero-shape-parameter choice on
the measured aperture.** The default was `"gaussian"`, whose radial falloff
was carried by the fitted `cathode_Ts_fwhm_cm = 28.0` footprint. That
empirical basis is RETIRED by the caliper: the measured emission footprint
identifies with the aperture the panel defines, not with an emission droop
across a larger disc, so the falloff no longer has a measurement behind it.
`"uniform"` introduces no shape parameter and is fully boxed by hardware.
`"gaussian"` remains a selectable arm and is what the golden fixture pins, so
the historical trajectory stays reachable and bit-exact.

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

**`cathode_jet_energy_convention = "legacy"` — a REPRODUCIBILITY PIN, not a
physical claim.** The two settings disagree about what `cathode_jet_R_E`
means, and only one of them is consistent with `cathode_jet_surface_debit`.
The debit is written in the total-reflected (Eckstein/TRIM) convention that
the coefficient itself is quoted in — reflected energy over incident, summed
over particles — so the surface gives up `R_E` of its ion bombardment power.
`"legacy"` reads the same `R_E` per backscattered particle and lets only the
`R_N` reflected fraction carry it, so the gas receives `R_N R_E` and the
`(1 - R_N) R_E` remainder is debited from the surface and received by nobody.
`"total_reflected"` gives each reflected particle `R_E/R_N` of the incident
energy, and the exported per-recycled-particle energy then equals the debited
one exactly. `"legacy"` remains the shipped default ONLY so that jet-armed
results predating the convention key stay bit-reproducible; it is not the
defensible reading of the coefficient, and any new energy-carrying jet arm
selects `"total_reflected"`. Inert when `cathode_neutral_jet` is off.

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
`heating_anomalous_transport = "tail_walk"` AND
`heating_anomalous_tail_energy_keying = "fixed"`.

**`heating_anomalous_tail_energy_keying = "phi_c"` — DERIVED.** The plateau is
filled by a beam whose energy IS the cathode accelerating drop, so keying the
tail birth energy to that drop rather than to a constant follows from the
channel's own driver rather than from a fit. It is not a free choice between
equals: with a FIXED rung the walkers' margin against the sheath that reflects
them is set by how far the drive happens to sit from the rung, and the
sheathwalk read (2026-08-05) measured up to 60% spurious ES1<->ES2 dependence
from that alone, against <3% under keying. `"fixed"` remains selectable and is
bit-exact, and is what every pre-K7 arm must name to reproduce.
Memo: `scripts/sheathwalk_report.txt`.

**`heating_anomalous_tail_phi_c_fraction = None` (the `f = 0.25` arm) —
ASSUMED, NEVER fitted.** `f` in `E_tail = f * e*phi_c(t)` is a DECLARED
BRACKET `{0.25, 0.5, 1.0}` and any other value is refused at construction,
precisely so it cannot become a knob. The shipped arm is 0.25 on CONTINUITY
grounds and nothing stronger: at ES1 drive it reproduces the shipped 75 eV rung
to within a factor 0.75-1.04, so adopting the keying does not silently move the
plateau energy at the same time. Honest bar: none — the plateau energy is
kinetic and a fluid model cannot pin it, which is why the bracket is the claim
and all three arms are reported together, exactly as for the fixed rungs.
Memo: `scripts/sheathwalk_report.txt`.

**`heating_anomalous_tail_cathode_boundary = "reflect"` — DERIVED from a
MEASURED potential.** The cathode sheath drop measured through drive is
190-310 V, above every plateau energy either bracket carries, so the cathode
reflects essentially the whole `-z` tail flux; the shipped free-escape
convention deleted it instead (the sheathwalk read measures the deleted share
at ~0.50 of `P_QL`, and reflection recovering it at x1.80-2.06 on the total
in-domain deposition). This is a boundary condition read off a solved quantity,
not a tuned one, and there is deliberately NO partial-reflection coefficient.
Standing rider, UNSIZED: a real cathode end is a grounded wall with an emitting
disc in it, so some of the returning tail misses the disc radially and is lost
at the wall's much smaller potential. The 1D walk has no radial coordinate and
cannot size that fraction; it is the one assumption that could pull the ~2x
below exact, and it is a documented limitation rather than a knob.
`"escape"` remains selectable, is bit-exact, and is what every pre-K7 arm must
name to reproduce. Memo: `scripts/sheathwalk_report.txt`,
`scripts/k7build_sheath_crosscheck.txt`.

**`heating_anomalous_tail_ionization = "off"`** ships off (bit-exact); the
default needs no physical justification, but the band treatment its `"on"` arm
uses does. The two depth-1 bars are DERIVED (the lowest He inelastic threshold
20.6158 eV, and the `<W_sec>` crossing at 221.406 eV, both computed from the
thresholds rather than tabulated). Since K7b neither refuses: below the lower
bar the march reverts to the energy-only walk, which is exact rather than
approximate because no inelastic channel is open there; above the upper bar it
marches with the depth-1 truncation, whose cascade understatement is MEASURED
at 0.08-2.03% (variant-B-over-variant-A ratios 1.0008-1.0203 over the six
above-bar frames at `f = 1.0`), i.e. the disclosed bar is <= 2.0%. Honest bar:
that spread is over the sampled frames, not a proof of a bound. The one
surviving refusal, a tail energy past the tabulated He EII cross section
(eps = 40.671258, ~999.98 eV at the module's `I_ion`), is a DOMAIN limit read
off the table itself and is unreachable at the 190-310 V drop this device
produces. Memo: `scripts/sheathwalk_report.txt`.

**`heating_anomalous_disposal = "local"`** ships off (bit-exact, verified
raw-uint64 on both kernel paths); the default needs no physical justification,
but the branching its `"landau_branched"` arm computes does.

**The branching is DERIVED — computed from boxed inputs, with NO new constant
and nothing tunable in it.** Per cell
`f_Landau = gamma_L / (gamma_L + nu_en/2)` splits each cell's extracted
anomalous power between the nonlocal tail and local bulk heat. Both rates are
already in the model:

- `nu_en/2 = 0.5 nn K_m(Te)` is the collisional Langmuir AMPLITUDE damping the
  `ql_relaxation` onset gate already weighs growth against (Ginzburg 1970), on
  the boxed two-node He e-n momentum-transfer table
  (`_cross.he_electron_momentum_transfer_rate_cm3_s`; its own provenance and
  bracket are the entry for that table);
- `gamma_L = sqrt(pi/8) w_pe (v_phi/v_te)^3 exp(-v_phi^2/2v_te^2 - 3/2)` is the
  Maxwellian Landau damping rate at the beam-resonant phase velocity, Krall &
  Trivelpiece §8 with the Bohm-Gross term, evaluated with the module's existing
  `w_pe = 5.64e4 sqrt(n_e)` and `v_phi` the ray's own launch energy. The
  `sqrt(pi/8)` prefactor is arithmetic inside the cited expression, not a
  description-class constant.

CROSS-CHECKED before use, and the anchors are smoke-pinned: the same expression
reproduces the QL-onset memo's independent §4 anchors — the `e^-37.0` Landau
exponent at Te 5 eV, the ~4e14 Landau-limited threshold at Te 25 eV, and
`nu_en(25 eV) = 1.405e6` s^-1 at the stance neutral density — and the stance
branching table `f_Landau = 0.8316 / 0.9398 / 0.9802 / 0.9936` at
`n_e = 1e8 / 1e9 / 1e10 / 1e11`. Sensitivity to the `K_m` 25 eV bracket is
<= 0.04 in `f_Landau`, i.e. immaterial. Memos:
`QL_ONSET_MEMO_2026-08-12.md` (including its 2026-08-13 addendum, which records
that the memo's §4 prose puts the 50% crossing higher than its own formula
gives — the error is in the conservative direction) and the pd0 read
`scripts/pd0_branching.txt`, which is the artifact these anchors are taken from.

Honest bars, both DOCUMENTED rather than sized: (i) the asymptotic Landau
expression is a large-argument expansion, quantitative for `v_phi/v_te`
above roughly 2.4 and marginal below it; the formula is used AS-IS across the
whole range rather than switched, and a cell in the marginal band is reported
as indicative. (ii) `v_phi` uses the ray's LAUNCH energy — CSDA slowing along
the column is not tracked into the resonance, the same convention the pd0 read
disclosed.

The birth energy of the Landau share introduces nothing either: it is the
EXISTING `heating_anomalous_tail_energy_keying = "phi_c"` path, whose `f`
bracket entry is above, and the branched arm requires that arm to be STATED
(the registered central arm is `f = 1.0`, while the shipped default `None`
would silently select 0.25). The cathode `reflect` convention, the free-escape
collector and the tail end ledger are likewise the existing ones.

REFUSED under `coverage_closure`, by design and not by omission: the two-stream
march shares ONE withholding bank between the channel and reservoir arms, so
the reservoir's extraction cannot be branched on the reservoir's own state; and
the reservoir carries `n_e = ` the density floor (a numerical constant standing
for "no plasma") against the mean-field `Te`, which returns `f_Landau` ~ 0.83
at Te 25 eV and ~0.98 at Te 55 eV — a branching owned by the floor convention
rather than by the plasma. The coverage arms are deferred until that stance is
designed.

#### QL relaxation closure (`beam_anomalous_model = "ql_relaxation"`)

The middle leg of the anomalous closure bracket. One config key, three boxed
module constants and a two-node table; all of it read only when this arm is
selected, and all of it from `QL_ONSET_MEMO_2026-08-12.md`. Not the shipped
default — the family is described in NUMERICS.md, "The anomalous closure
bracket".

**`ql_relaxation_coeff = 30.0` — ASSUMED, NEVER fitted.** The O(10-100)
coefficient `c` in the quasilinear plateau-formation time
`tau_QL = c (n_e/n_b)/w_pe` (Vedenov-era scaling as restated in Krall &
Trivelpiece §10, which gives it as an order-of-magnitude class and not a
number), and hence the length `L_rel = tau_QL v_b` the extracted beam power is
spread over. **Bracket `[10, 100]`, and the bracket is the claim**: the shipped
value is the bracket's geometric centre and carries no more standing than the
endpoints, so every headline under this closure is quoted at 10, 30 and 100.
Honest bar: none is possible from within a fluid model — the plateau-formation
time is kinetic, and the source states a decade-wide class rather than a
coefficient. This is the closure's ONLY description-selecting constant and it
is deliberately NOT defaulted at the point of use: `deposit_beam` raises rather
than substituting a value, so a run cannot land behind a published number
without having named its bracket arm. Inert under every other
`beam_anomalous_model`, and byte-inert (smoke-pinned). MEASURED consequence:
the third balance column bins SPLIT across this bracket — no root at `c = 10`,
root at 30 and 100 (NUMERICS.md).

**`_beam_deposition.QL_TRAP_COEFF = 1.0` (module constant, not config) —
DERIVED (cited scaling).** `C_trap` in the reactive-trapping extracted fraction
`f_ext = C_trap (n_b/2n_e)^(1/3)` [O'Neil, Winfrey & Malmberg, Phys. Fluids 14,
1204 (1971)]. The cited result is the SCALING, stated as `~(n_b/2n_e)^(1/3)`;
unity is the adoption of that form with its order-unity prefactor unresolved.
Honest bar: exactly that — the prefactor is not separately measured here, and a
factor-of-a-few in `C_trap` is degenerate with `ql_relaxation_coeff`, which is
the bracket that is reported. Not exposed as config, precisely so it cannot
become a second knob on the same product.

**`_beam_deposition.QL_GROWTH_COEFF = 0.687` (module constant) — DERIVED.** The
cold-beam beam-plasma growth rate `gamma_r = 0.687 w_pe (n_b/n_e)^(1/3)`
[O'Neil & Malmberg 1968; Krall & Trivelpiece §9]. Enters the onset gate only —
it never scales a deposited power. Honest bar: the `min(n_b/n_e, 1)` cap that
keeps it finite in the `n_b >~ n_e` corner is a **FLAGGED INFERENCE**, not part
of the cited result; it holds the rate at `0.687 w_pe` rather than continuing a
curve past its own domain, and the same flag rides `f_ext`'s cap.

**He e-n momentum-transfer table `_cross.HE_EN_MT_SIGMA_CM2` (module data, not
config).** `nu_en = nn K_m(Te)` is the damping side of the onset gate, with
`K_m(Te) = sigma_m(1.5 Te) * <v>(Te)`. Two nodes, log-log inside the span and
CLAMPED outside it.

| node | shipped `sigma_m` | class | bracket carried | honest bar |
|---|---|---|---|---|
| 5 eV | `6.0e-16 cm^2` | **MEASURED (cited)** | `5.7e-16 - 6.3e-16` | the ±3-5% of the source, propagated |
| 25 eV | `2.1e-16 cm^2` | **ASSUMED — bracket** | `1.6e-16 - 2.6e-16` | **not verified against a primary table** |

The 5 eV node is Milloy & Crompton, PRA 15, 1847 (1977), ±3-5%, consistent with
Crompton, Elford & Robertson, Aust. J. Phys. 23, 667 (1970); the shipped value
is the cited central and the bracket is that stated uncertainty. The 25 eV row
is a BRACKET of the Register/Trajmar/Srivastava-class values as carried in the
LXCat sets (per Alves et al., J. Phys. D 46, 334002 (2013)); the shipped value
is its arithmetic midpoint, and **the memo did not verify it against a primary
table — pull the LXCat set before boxing it**, and until then the row must be
quoted as a bracket and never cited as a primary measurement. Both brackets are
published as `HE_EN_MT_SIGMA_BRACKET_CM2` so a result can quote them.
Order-of-magnitude standing overall: `K_m` is formed as `sigma(<E>)·<v>` rather
than by Maxwellian quadrature, because a two-node table cannot support the
precision a quadrature would imply. This is tolerable HERE and only here: the
gate it feeds is open by ×400-2500 across the working range, so no headline
moves anywhere inside either bracket. It must not be reused as a transport
rate on that basis.

**`beam_deposition_smoothing_cm = 0.0`** ships off (bit-exact). The campaign
stance sets a nonzero width, and that width is **load-bearing rather than
cosmetic**: at the production operating point the raw CSDA stopping profile is
only ~2 cells wide, so the kernel sets the applied axial deposition geometry
outright. The value, its ASSUMED class and its honest bar live with the stance,
in `scripts/production_stance_provenance.md`.

## `physics_fit_defaults`

**`Tn_K = 300.0` K — MEASURED.** The single cold-gas neutral temperature, also
the Phelps operator's `T_eff` input. It ended a term-by-term mix of `Tn_K` and
the fitted `Tn_fit`.

**`heat_picard_iterations = 2` — DERIVED.** A positive value is required for
`tr_bdf2` + `strang` to express second order; 0 freezes the conductivity and
caps the whole step at first order.

**`neutral_energy_wall_accommodation = 0.40`, bracket `[0.35, 0.46]` —
MEASURED (literature-boxed).** The thermal accommodation coefficient of helium
on an engineering (unpolished, air-exposed, vacuum-baked) stainless surface,
read only under the default-off `neutral_energy` flag. Three independent
sources box it:

- the Sandia parallel-plate accommodation programme (SAND2005-6084; the same
  apparatus and analysis published as *Rev. Sci. Instrum.* **82**, 035120
  (2011)) measures He on engineering-finish stainless at **0.36–0.46**, and
  reports a downward shift to **0.31–0.38** on plasma-conditioned (sputter-
  cleaned) surfaces — the cleaner the metal, the weaker the coupling;
- Zampella *et al.*, PATRAM 2019, independently obtain **0.35–0.37** for He on
  stainless cask surfaces;
- the Song & Yovanovich (1987) engineering correlation, evaluated for He on
  steel at 300 K, returns **≈0.40**.

The three overlap on 0.36–0.37 and span 0.35–0.46; 0.40 is the round value
inside that overlap and at the correlation's own prediction. **Honest bar:
±0.06**, dominated by surface condition (finish, adsorbate coverage, degree of
plasma conditioning) rather than by any instrument's precision — the
plasma-conditioned shift alone is that large.

**Stated caveat — this number applies to the THERMAL population only.** The
measurements above are equilibrium-gas experiments at or near 300 K. For the
1–5 eV charge-exchange tail the same surface behaves very differently: clean
metal molecular-dynamics (Borovikov/Voter/Tang, *J. Nucl. Mater.* **447**
(2014)) gives **≈0.07–0.09** for energetic He, and no engineering-steel
measurement exists at those energies at all. The shipped single coefficient
therefore over-accommodates the CX tail. If that channel is ever split out, the
two endpoints **0.1 / 0.4** are to be run as a declared family and the bracket
reported as the claim — they are never to be fitted, on either arm.

**UPDATE (NBL pass 2).** That channel IS now split out: the CX-born population
is a separate, collisionally decoupled hot channel
(`physics/hot_neutrals.py`), and `alpha_E` no longer touches it. The
coefficient above now applies to what it was measured on — the thermal bulk at
the wall sink — and nothing else. The hot channel's own surface treatment is
NOT `alpha_E`: under the ratified annulus-cold v1 cut an intercepted hot atom
leaves its whole excess energy on the wall, of which only the `alpha_E` share
is accommodation in the measured sense and the remainder is the cut. The run's
hot-channel diagnostics report both, so the declared **0.1 / 0.4** family
remains the right instrument for the CX tail whenever that cut is relaxed —
still never fitted.

**`neutral_knudsen_temperature = "frozen"` — ASSUMED (stance choice, ratified
v1-primary).** Which temperature the Knudsen conductances take their thermal
speed from; read only under the default-off `neutral_energy` flag, because it
is only there that a second answer exists. `"frozen"` evaluates every
conductance once at `Tn_K`, making neutral transport a fixed property of the
geometry — the behaviour every scored run to date was produced with, which is
why it is the primary. `"local"` scales each conductance by
`sqrt(Tn_local/Tn_K)` and is a DISCLOSED sensitivity arm. **Honest bar: the arm
is unscored.** It also does less than its name suggests: it scales the transport
RATE, not the equilibrium, because the driving potential is still the density
difference — the textbook transpiration relation `n ~ 1/sqrt(T)` would require
changing that potential and is not built. No measurement or fit is implied by
either setting.

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

**`circuit_dt_fraction = 0.25` — ASSUMED numerical-control value**, carrying
the same quarter-of-a-relaxation-time convention as `density_dt_fraction`,
`neutral_dt_fraction` and `heat_dt_fraction`. What is MEASURED is the feature
it exists to resolve, not the fraction: the sheath capability wall's device
slope reaches ~2 kOhm, giving `tau_circuit = L/(R + dV_dis/dI)` ~ 4 ns against
an `L/R_comp` of 1.12 ms and a `dt_max` of 1e-4 s, and the sub-wall slew
crosses the wall in ~45 ns (`scripts/regime_dtq_wallmap.txt`,
`scripts/regime_dtq_frozen_circuit.txt`, 2026-08-12). The bound is an ACCURACY
control — the TR-BDF2 loop advance is L-stable and needs no stability
restriction — and is read only while `cathode_circuit_voltage_bound` is armed.

The remaining timestep parameters (`cfl`, the other `*_dt_fraction` limits,
growth and retry factors) are ASSUMED numerical-control values with no
measurement behind them.

## `coverage_closure_defaults`

All four keys are read only under the default-off `coverage_closure` flag, so
nothing here is on any shipped trajectory. The closure declares exactly TWO
physical constants — `coverage_growth_rate_per_s` and
`coverage_backfill_time_s`. `coverage_initial_fraction` and
`coverage_initial_profile` are initial conditions, not constants, and appear
here only because they also have no default. **Neither the two-medium beam
split nor the v2 z-resolved rebuild introduced a constant of its own: the split
ratio IS `f_cov`, the reservoir's plasma density is the existing `ne_floor`,
and v2's growth driver `w(z,t)` is the beam-ionization rate divided by its own
volume-weighted column mean — a normalization with `<w> = 1` by construction,
so it carries no free scale and leaves `coverage_growth_rate_per_s` meaning
exactly what it meant before.**

**`coverage_growth_rate_per_s = 1390.0` s^-1 — FITTED-on-F2, CALIBRATION
PENDING. The shipped number is a PLACEHOLDER and must not be quoted as a
result.** Under the logistic law `df_cov/dt = r0 w f_cov (1 - f_cov)` — with
`w` the parameter-free normalized driver, so `r0` is the COLUMN-MEAN rate — the
small-`f_cov` limit is exponential with e-fold time `1/r0`, so the placeholder
is the reciprocal of the midpoint of the MEASURED pedestal e-fold window
713–725 µs (`1/719 µs = 1391 s^-1`, rounded to 1390). That is a scale-setting
identification, NOT the calibration: the measured e-fold constrains the growth
of the observable the pedestal is read from, and the mapping from that
observable to `f_cov` has to come from the F2 fit against the pedestal itself,
which happens post-merge. Honest bar until then: the placeholder is right to
within whatever that mapping costs, which is unmeasured, so treat the value as
order-of-magnitude only. The reciprocals of the window edges, 1379–1403 s^-1,
are the spread of the ANCHOR, not an uncertainty on `r`.

**This key is the SHARED PERCOLATION CLOCK (ruling of 2026-08-13).** The
cathode emitting-area closure (`cathode_emitting_area`, below) grows its lit
fraction on the same law and reads THIS key rather than carrying a rate of its
own: the column coverage and the lit cathode disc are declared the same
physical percolation seen from two surfaces, so the constant is fitted once,
against the F2 current-foot waveform, and has exactly one owner. Registering a
second rate would have been a second fit to the same waveform. Two consequences
are structural, not stylistic: a re-calibration of this number moves BOTH
closures, and the key is live — non-default values accepted — whenever EITHER
flag is armed, which is why the coverage validator's inert-key refusal exempts
this key alone under the emitting-area flag.

**`coverage_backfill_time_s = 3.0e-5` s — ASSUMED. Bracket
1.0e-5 – 1.2e-4 s, and the bracket is the claim.** The time over which the
uncovered reservoir refills a burnt channel is a free-molecular transit across
the inter-channel spacing. Helium at the model's `Tn_K = 300` K has mean speed
`sqrt(8 k T / (pi m)) = 1.26e5` cm/s, so a refill path of 1 cm gives 8e-6 s and
one of the full plasma radius `Rp = 15` cm gives 1.2e-4 s; the bracket rounds
those to one significant figure and the shipped value is the ~4 cm spacing in
the middle of it. Nothing in the campaign measures the azimuthal channel
spacing, which is why this is a bracket rather than a point. Two further
approximations sit inside the same constant and are NOT separately
parameterized in v1: the closure carries its neutral deficit on the same
chamber-mean `nn` field the burn debits, so `tau_backfill` absorbs the
column-to-chamber exchange as well as the channel-to-inter-channel one; and
the exchange is a single-rate relaxation rather than a transport operator.

**`coverage_initial_fraction = None` — no default, and an INITIAL CONDITION
rather than a physical constant.** The uniform spelling: one covered fraction
applied to every cell. `None` is not a neutral default — 1.0 is the
fully-covered mean-field limit and would make the flag a silent no-op — so the
flag requires an initial condition explicitly.

**`coverage_initial_profile = None` — no default, and an INITIAL CONDITION
rather than a physical constant.** The per-cell spelling: `f_cov0(z)` as a
sequence of length `nx`, every entry in `(0, 1]`. This is the L3 ENSEMBLE HOOK.
The solver contains no randomness; a realization is one run with one externally
generated profile, so the ensemble's statistics live entirely in whatever
generates the profiles and are reproducible from the saved config alone.

With the flag on, EXACTLY ONE of the two must be given — they are two spellings
of the same initial condition and neither modifies the other, so there is no
composition rule and supplying both is a construction-time `ValueError`. This
is a deliberate choice of the least-surprising rule over a precedence rule
(e.g. "profile wins", or "profile scaled by the fraction"), which would have to
be remembered and could be got wrong silently.

## `emitting_area_defaults`

The one key here is read only under the default-off `cathode_emitting_area`
flag, so nothing in this section is on any shipped trajectory. **The closure
declares NO constant of its own beyond this seed.** Its growth rate is
`coverage_growth_rate_per_s`, the shared percolation clock documented above;
the logistic saturation at `f_em = 1` is structural (the face cannot be more
than fully lit), not a fitted ceiling; and the patch-scale assumption that lit
patches are large compared with the Debye length — which is what lets each
patch carry the same one-dimensional sheath the full disc does — is ASSUMED and
stated, with no number attached.

**`cathode_emitting_area_initial_fraction = 0.0075` — DERIVED-with-bracket.
Bracket 0.0063 – 0.0087, and the bracket is the claim; the shipped value is its
midpoint.** The lit fraction at the start of the machine's current window is
the back-extrapolated window-start discharge current over the current the model
carries when the whole face is lit. The numerator is the machine's
back-extrapolated foot, 0.34 – 0.47 A. The denominator is 54 A — the
CIRCUIT-SET equilibrium current of the model's build end, not a wall
coincidence: it is what the loop equation settles at, which is why a
normalizer that moves with the emission ceiling was rejected. Both endpoints of
the bracket come from dividing the two ends of the measured foot by that one
number (`0.34/54 = 0.0063`, `0.47/54 = 0.0087`), so the bracket carries the
measurement spread of the foot and nothing else.

Two alternative normalizers were considered and are EXCLUDED, not assumed away
(`scripts/ea0_emission_probe.{py,txt}`, 2026-08-13):

* the static Richardson ceiling, 3151 A at the stance, gives `f_em0 ~ 1.3e-4`.
  It is the wrong object: the model never rides the Richardson ceiling — the
  measured release is space-charge/voltage-clamped a factor ~1670 below it
  (annular Richardson 3151 A against 1.9 A actually carried at the 178 V
  circuit ceiling at seed conditions).
* the ~1.5 A the model carries at `t = 0` — the other end of the same
  "1.5 → 54 A" transient — gives `f_em0 ~ 0.23 – 0.31`. That is excluded by the
  closure's own feasibility condition rather than by preference: at those
  values the primary-alone gain is 4.2e3 – 5.9e3 s^-1 against an all-surface
  loss of 4.45e3 s^-1, i.e. already critical or supercritical at the seed, so
  there is no subcritical foot for the closure to describe.

At the shipped value the primary-alone gain is 118 – 163 s^-1 against the same
4.45e3 s^-1 loss — subcritical by a factor 0.026 – 0.037, which is the property
the closure exists to express and is measured, not imposed.

**Composition with `coverage_closure` is PERMITTED, not refused.** The two
closures act on different surfaces — the cathode's emitting face and the
column's cross-section — and the audit found no shared state: `_cathode_f_em`
is one scalar consumed only at the device-config seam, `_coverage_f` is a
per-cell field consumed only by the beam split and the neutral partition, and
they occupy separate restart payload sections (`cathode` vs `coverage`). The
emitting-area clock is autonomous, and a composed run's `f_em(t)` was measured
bit-identical to the same window run with the column closure off (2026-08-13).
Their one common object is the growth constant above, which is shared BY
DESIGN. A composed arm is therefore a legitimate configuration and simply has
to disclose that it is one; nothing about the composition is implicit.

**Honest bar.** The number is DERIVED, not measured: it inherits the
back-extrapolation of the current foot (an extrapolation, not a reading) and
the identification of the model's circuit-set 54 A with "the fully lit face".
Neither the lit area nor the patch count is observed in this machine. Two
physical keyings that WOULD have pinned the growth from measured surface
physics were evaluated and both fail the clock by orders of magnitude at the
shipped constants — ion-bombardment cleaning gives `tau ~ 13 s` at foot flux
(~4 decades slow) and thermal spreading `tau_th ~ 94 ms` (~2 decades) — so they
are registered as arithmetic-excluded alternatives rather than options, and the
logistic with the one shared fitted rate is the minimal honest law that
remains.

## `neutral_probe_source_defaults`

**All ten keys are `None`, and that is the whole entry: this group ships no
number, so there is nothing here to classify.** The ad-hoc probe source is an
INFERENCE INSTRUMENT — an arm with it on measures the plasma's response to a
*hypothesized* neutral source — so every one of its values is the hypothesis
the arm is stating and none of them is a property of the machine. A default
would be a hypothesis nobody made, silently inherited into a run and then read
back out of the saved config as if someone had chosen it. There is accordingly
no default amplitude, no default placement, no default waveform, and (under two
zones) no default zone; each is a construction-time `ValueError` when the flag
is on, and each is refused when the flag is off. Nothing in this group is on
any shipped trajectory.

Two consequences worth stating, because both look like missing entries:

- **There is no injection-temperature key.** The moment neutral model carries
  one neutral temperature, `Tn_K = 300 K` (its own entry above), and no neutral
  energy equation, so probe-injected particles join that single cold-gas
  population exactly as gas-puff particles do. A distinct probe temperature
  would be a new field, not a new parameter, and would need its own provenance
  and its own physics.
- **There is no smoothing constant.** The `square` waveform's edges are hard
  and its window is the half-open `[t_on, t_off)`; the `table` waveform is
  linear between the caller's own nodes and exactly zero outside their span.
  Any rise or fall time here would be an invented number in a group whose whole
  point is that it invents none — a caller who wants a shaped turn-on tabulates
  one, where it is visible in the config.

`neutral_probe_amplitude_cm3_s = 0` is the explicit NULL CONTROL and must be
asked for: it is bit-exact against the flag-off trajectory, which is what makes
it a usable control rather than a silent no-op. An identically-zero
`neutral_probe_profile` is refused instead, because a shape that injects
nowhere is a misconfiguration and the null control already has a key.

## `regime_tracer_defaults`

These eight numbers **select a description**: each one decides where the
pre-breakdown leg stops being an ODE and starts being a fluid. None of them is
a property of the machine, so none is measured, and the registration below IS
the acceptance criterion — a threshold that appeared in the code without an
entry here would be an anonymous knob deciding a physical boundary. Every one
is sweepable by config alone (`eps/3`, `eps`, `3 eps`) with no code edit, and
the `active_criterion` census on every tracer run reports which of them binds,
where, and when.

The three passivity criteria all ship at the same `0.01`. That is not
laziness: they are three spellings of one statement — *the plasma's
back-reaction on the circuit, the beam and the neutral background is a 1%
correction* — and giving them different values would assert a precision about
their relative importance that nothing supports. They are separate keys so that
a sweep can move one at a time and the census can say which one bound.

**`tracer_passivity_current_ratio = 0.01` — DERIVED.** The statement is
"neglected plasma back-reaction on the loop current is below 1% of the
loop-current budget", tied to an existing error bar. The R5.1 fluid/circuit
coupling audit (`NUMERICS.md`, "R5.1 gated fluid<->circuit Picard") measured
the whole sequential-vs-Picard coupling neglect at ~3% on `I_tot` and the
sequential stance was nonetheless **retained as production**. A conducted-plasma
shunt below 1% of the loop current is therefore a third of a neglect the
campaign has already accepted and re-derived on its own terms, which is what
makes this DERIVED rather than ASSUMED. Honest bar: the 3% figure is an audit
measurement at one operating point, so the derivation transfers a *scale*, not
a bound. Bracket `0.003 - 0.03`; at the upper end the tracer leg's neglect
becomes comparable to the accepted coupling neglect rather than small against
it.

**`tracer_passivity_thinness = 0.01` — ASSUMED.** Below a 1% single-pass energy
loss to plasma electrons, the deposition profile the tracer consumes is the
vacuum-column profile. Nothing pins the number: the honest observation is that
the *closure* bracket on beam deposition (Beer-Lambert vs CSDA x
{classical, quasilinear}) moves the deposited profile by far more than 1%, so
any value in the bracket below is invisible against a systematic the campaign
already reports as a bracket. Bracket `0.003 - 0.03`.

**`tracer_passivity_depletion = 0.01` — ASSUMED.** `gamma` is proportional to
`nn`, so a plasma-driven burn of a fraction `f` of the local neutrals is a
fractional error `f` in the growth rate. 1% is chosen to match the other two
criteria. Bracket `0.003 - 0.03`. Note the beam's own neutral debit is
deliberately excluded: it is background, and including it would trip this
criterion on a quantity the plasma did not cause.

**`tracer_passivity_hysteresis = 3.0` — ASSUMED.** A pure numerical guard with
no physical referent. All three ratios rise monotonically while the discharge
builds, so a cell that activates is not expected to return; the width only has
to be wide enough that round-off cannot flip a cell sitting on a criterion.
A factor 3 is comfortably wider than any plausible round-off excursion and
narrow enough that a genuinely non-monotone background would still be tracked.
Bracket `1.5 - 10`; the result must not depend on it, and a run in which it
does is a run whose criteria are chattering and should be reported.

**`tracer_refresh_tol = 0.01` — NUMERICS.** A Picard freeze tolerance in the
same family as `circuit_picard_tol_rel`, not a description-selecting constant:
it trades cost against the size of the frozen-coefficient error. Freezing
`gamma` while the background moves by at most 1% caps the resulting error in
the exact affine update at the same order. `0` refreshes every step and is the
reference against which the shipped cadence is checked. Bracket `0.001 - 0.1`.

**`tracer_activation_ne = 1.0e10` cm^-3 — DUAL ROLE; classed separately per
role.** One number is doing two jobs, and they do not have the same standing.
The value is unchanged by this entry.

*Role (i), the fluid-validity / handoff gate — DERIVED.* Two conditions have to
hold before the fluid can be handed a cell, and this is the larger of them.
(a) The density floor must be inert: `ne_floor = 1e8`, and the fluid's
`_negative_margin_timestep` bound degrades as `n` approaches it, so a 100x
margin is the condition that the clip is not what is holding the cell up.
(b) The value must sit at the bottom of the density range the fluid model is
validated over — the LAPD afterglow/early-discharge scale is `1e10 - 1e11`
cm^-3, so `1e10` is the low edge of the validated window rather than an
extrapolation. Construction enforces only condition (a) (`>= 10 * ne_floor`),
because `ne_floor` is itself configurable and (b) is a judgement about the
model's validity, not an arithmetic relation. Honest bar: the 100x margin is a
sufficiency argument, not a measurement of where the clip stops mattering.
Bracket `3e9 - 3e10`.

*Role (ii), the de-facto quasilinear-onset gate — ASSUMED, and **RESOLVED
BELOW RANGE** (amended 2026-08-12).* Because the passive/active mask is also
what gates the anomalous beam power booking, this same number decided where
quasilinear absorption started being booked. It was never derived for that job.
The bracket as first registered was `[2.9e9 cm^-3, substantially higher]`, its
lower edge being the deposition module's own weak-beam validity floor —
`10 n_b` at stance, the density below which `quasilinear_relaxation_length_cm`
returns `inf` — which is **a numerical validity bound on the closure, not the
physical instability onset**.

*Memo adoption (`QL_ONSET_MEMO_2026-08-12.md`).* The physical onset that was
"under separate derivation" has landed, and it does not bound this number from
either side. The LINEAR beam-plasma onset is **always on** over the working
range `n_e = 1e8 - 1e11 cm^-3`, by ×400-2500 against He collisional damping, so
the onset criterion is satisfied everywhere `n_act` could plausibly sit and
`max(n_QL, n_fluid)` degenerates to `n_fluid`. The ASSUMED bracket is therefore
**retired as resolved below range** rather than narrowed: no split occurs, and
role (ii) leaves this entry. What replaces it is not another threshold but a
CLOSURE FAMILY — the onset question now lives in `beam_anomalous_model`, whose
`"ql_relaxation"` arm evaluates the boxed inequality per cell and whose gating
physics is relaxation, not onset (NUMERICS.md, "The anomalous closure
bracket"). **Role (i), the fluid-validity/handoff gate, is DERIVED and is now
the sole binding role; the VALUE is unchanged by this amendment.**

*Measured consequence of role (ii)'s ignorance interval.* The two-sided overlap
gate FAILS at the shipped value (worst relative disagreement 0.978 against
`tracer_overlap_rtol = 0.05`): the fluid arm is heated across the band by
quasilinear power the tracer arm refuses below `n_act`, so the two descriptions
do not meet. That failure is the measurement of the gap, and neither the gate
nor this value was adjusted to remove it. See NUMERICS.md, "Corrected beam
power booking on passive cells". For role (i) the overlap gate remains the
check that makes the choice falsifiable — if the two descriptions agree across
the band, the boundary inside it did not matter.

**`tracer_overlap_band_ne = (1.0e10, 1.0e11)` cm^-3 — DERIVED.** The band where
BOTH descriptions are valid, and therefore the only place a two-sided check is
meaningful. Its low edge IS `tracer_activation_ne` (below it the fluid is
floor-poisoned); its high edge is one decade up, where at the shipped fill the
conducted-current criterion (a) is approaching its threshold and the tracer
stops being valid. A decade is what the two validity windows actually share;
widening it would put one side outside its own domain and turn a disagreement
into an artefact of the band rather than a finding. Honest bar: the high edge
is read off criterion (a) at the shipped stance and moves with `nn` and the
circuit ramp.

**`tracer_overlap_rtol = 0.05` — ASSUMED.** The two descriptions are not
supposed to be bit-identical: the tracer neglects parallel transport (the
`c_s/(L_n gamma)` term tabulated in `NUMERICS.md`) and holds `Te` quasi-static,
while the fluid resolves both. 5% is chosen as a tolerance that a genuine
implementation defect would exceed while the *stated* physical differences
would not. Honest bar: it is not derived from the neglect bound, and it cannot
be — that bound reaches ~50% at the low-`Te` end. What the gate therefore
proves is agreement *at the operating point it runs*, not agreement in general;
a PASS is evidence about the code, and the neglect table is the statement about
the physics. Bracket `0.02 - 0.10`.

---

## `regime_vessel_node_defaults`

Both values are HARDWARE quantities of the machine's electrical topology, not
model closures. Nothing in this group is tuned, and nothing in it is fitted to
any run.

The topology behind them is hardware-verified: the cathode/anode system floats
with respect to the machine wall; the entire electrically connected stainless
vessel — some 20 m of it — is ONE wall conductor; and the anode is referenced
to that conductor only through FOUR feedthrough capacitors bridging the ceramic
gap insulators.

**The capacitor TYPE is visually UNRESOLVED** (2026-08-12). A first look read
them as axial-wound aluminium electrolytics; a second look read them as axial
**polypropylene film**, with a black mark on one side of the cylinder. The
mark does not settle it and mildly favours film: on a film capacitor a plain
band conventionally marks the OUTER-FOIL terminal — a shielding convention,
not a polarity one — while electrolytics mark polarity with explicit `-`/`+`
symbols. The type sets the leak class below and the tolerance term in
`C_total`'s bar, so both are carried type-conditionally until the **bench
measurement**, which resolves the type and the value together.

**`vessel_capacitance_F = 1.3e-6` F — ESTIMATED. THE BRACKET IS THE CLAIM.**
The four feedthrough capacitors are in parallel, so `C_total = 4 * C_each`.
`C_each` is an ENGINEER'S ESTIMATE — "probably 0.1-1 uF" — with no part number
read and no bench measurement behind it, which puts

    C_total in [0.4, 4.0] uF,

a factor of ten wide. The shipped value is the bracket's geometric midpoint
(`sqrt(0.4*4.0) = 1.265` uF, rounded to 1.3), chosen so that a run left at the
default sits in the middle of the bracket rather than at an edge. **It is not
a measurement and no result may quote it as one.** Any result that depends on
`C_total` must report the bracket, and
`scripts/regime_vcm_r0b_check.py` is the instrument that sweeps it.

*Honest bar: a factor of 10, one-sided in neither direction, and the part
tolerance sits INSIDE it under either type reading.* Polypropylene film runs
`±5-10 %`; aluminium electrolytics of this vintage run `-20/+80 %` (a factor
of ~2.25). Both are dominated by, and already contained in, the factor-of-ten
engineer bracket, so the unresolved type does NOT widen this bar — which is
why the type question is a leak question rather than a capacitance question. A
bench measurement removes every term at once; a part number alone would not.
What the bracket does and does not decide, measured across
`0.4 / 1.3 / 4.0` uF at `V_scale = V_bank = 180` V:

- The PHASE SEQUENCE is bracket-stable in kind. Early build is
  wall-referenced at every capacitance (at 1 mA seed current the node reaches
  only 1.4-13.9 % of the bank scale over a 10 ms cycle); engagement follows;
  the bootstrap's sign is a property of the ODE and not of `C_total` at all.
- The CURRENT AT WHICH each phase occurs scales linearly with `C_total` and so
  moves by the full factor of ten: the charging time crosses 713 us at
  `0.101 / 0.328 / 1.010` A respectively. Only the middle of the bracket lands
  near the measured band (the pre-avalanche discharge current back-extrapolates
  to 0.34-0.47 A at the window start; 1.3 uF gives 0.328 A, 3.5 % below its
  lower edge). That near-coincidence is an observation, NOT a calibration —
  `C_total` must not be fitted to it.

**A BENCH MEASUREMENT IS INCOMING** and will replace this entry. When it
lands, the class becomes MEASURED with the instrument's own bar, the bracket
above is retired, and any result quoted against the bracket is re-read at the
measured value.

**`vessel_leak_resistance_ohm = 1.0e10` Ohm — ESTIMATED, TYPE-UNCERTAIN. THE
BRACKET IS THE CLAIM.** The bracket spans BOTH readings of the unresolved
capacitor type:

| reading | basis | `R_leak` |
|---|---|---|
| electrolytic | leakage spec `I_leak ≈ (0.01…0.03)·C[µF]·V[V]` µA, over `C_total ∈ [0.4, 4.0]` µF at the ~178 V bank scale, inverted | `2.5e7 … 1e9` Ω |
| polypropylene film | insulation-resistance class of the dielectric | up to `~1e11` Ω |

so the carried bracket is

    R_leak in [2.5e7, 1e11] Ohm,

about four decades. The shipped default `1e10` takes the **film reading**,
which is the second-look call, at its high-edge decade. `None` remains
accepted and means the idealized hard float — an explicit A/B arm.

*Honest bar: four decades, dominated by an unresolved TYPE rather than by an
unresolved value, and the LOW EDGE IS SOFT.* If the parts are electrolytic,
the spec describes a capacitor that is formed and in service; these sit
unbiased between shots and are old, and an unbiased or aged electrolytic loses
oxide, leaks WORSE than spec — sometimes by a large factor — and re-forms only
under bias. So the true `R_leak` could fall below `2.5e7` Ω, and the bracket
is not symmetric in credibility. **The bench measurement resolves the type and
the value in one step**, and this entry is provisional until it lands.

**WHY THE UNRESOLVED TYPE DOES NOT BLOCK THE MODEL.** The structural fact the
node rests on is the leak TIMESCALE against the discharge, and it survives
BOTH readings:

    tau_leak = R_leak · C_total ≈ 10 s  …  4e5 s

over the joint bracket, against a ~25 ms discharge — a separation of at least
~400x at the most pessimistic corner (`R_leak = 2.5e7` Ω, `C_total = 0.4` µF,
i.e. the aged-electrolytic edge), and vastly more at the film edge.
**Within a shot the node is therefore hard-float IN KIND whichever type these
turn out to be**, and the phase sequence (early wall-referenced / engagement /
bootstrap) is unchanged by the leak. `scripts/regime_vcm_r0b_check.py` sweeps
the `R_leak` endpoints alongside the `C_total` endpoints and reports the
in-window sensitivity as a NUMBER: the worst shift anywhere in the joint
bracket is `1.25e-3` relative, which is exactly the closed form's
`dt/(2*tau_leak)` at that corner and two decades below the factor-of-ten
`C_total` bracket the same result already carries. A shift that DID reach the
percent level would be a finding, not a rounding error. Note the converse: the
separation is a statement about the DISCHARGE window only, and it fails on any
question posed over seconds.

**Two documented model deviations, both deliberate and neither built.**

- *Polarity — conditional on the unresolved type.* IF the capacitors are
  electrolytic, they are polarized and conduct asymmetrically under reverse
  bias (roughly diode-like above ~1-2 V), and the machine's plateau
  common-mode bias is observed at EITHER sign, so the reverse branch is
  physically reachable; the shipped leak is a SYMMETRIC linear resistor in
  both directions, which is the deviation. IF they are film there is no
  polarity nuance to model at all, and the black band on one side of the
  cylinder is the conventional outer-foil marking rather than a polarity mark.
  Either way this matters only if `V_cm` scoring eventually cares about the
  negative-plateau branch; on the discharge timescale the leak moves nothing
  in either direction, which is why an asymmetric leak model is not worth its
  constants today.
- *Inter-shot memory.* `tau_leak` far exceeds the ~3 s shot period under both
  readings (and by a wide margin for film), so the capacitors cannot discharge
  the node between shots. The physical reset path is the afterglow plasma
  conductance, not the leak. Runs here are single-shot and start from
  `V_cm = 0`; a shot-to-shot study would have to model that reset explicitly.

Zero and negative values raise at construction: they are not ties.
