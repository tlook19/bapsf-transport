# Provenance of the production stance (`scripts/stances/g1atrim.toml`)

The production stance of record is the committed stance file
`scripts/stances/g1atrim.toml`, loaded by `scripts/stance_config.py`
(R1, 2026-08-20). The `PARAM_OVERRIDES` / `FLAG_OVERRIDES` dicts in
`scripts/compare_sim1d_es1.py` still exist, but they are no longer the home of
these values: every key the stance names is now POPULATED FROM the stance file
(`_STANCE = load_stance(PRODUCTION_STANCE).params`, `compare_sim1d_es1.py:101`)
rather than repeated in the dict, and what the dicts still carry on their own is
the shared configuration the stance deliberately omits — the circuit, the rate
model and the numerics package. This file records where each stance value came
from. Parameter *meanings* are in the docstrings of
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
default-side entry. **The golden fixture applies THIS FILE.** R2b recaptured it
at the stance of record — `default_config()` + this stance + `nx = 60`, minus the
per-mesh arrays, which cannot travel to the gate's coarse mesh
(`golden_baseline_provenance.md` records the re-cut). So the fixture no longer
pins a frozen operating point, and the corollary is load-bearing: **editing a
scalar in this file breaks the golden until the fixture is recaptured.** That is
deliberate, and a recapture stays a reviewed, authorized, recorded event.

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

**`V_bank` IS NO LONGER NAMED BY THE STANCE FILE (row deleted 2026-08-25 — the
23b R4 deletion, executed for this half).** It never was a stance decision:
`V_bank` is a PER-RUNG measurement owned by `run_mechanism_ladder.ES_OPERATING`
(ES1 = 177.843, exactly the number the stance file repeated), so naming it in
the stance duplicated the rung and could only go stale against it. **Campaign
scores are unaffected** — every campaign arm takes `V_bank` from its rung, which
is why deleting the row produced NO delta in the pre-flight against
`scripts/mgcr1_confirm.h5` and why the three campaign config-snapshot cases did
not move on it. The GOLDEN takes no rung, so its `V_bank` falls back to the
config default 180.0; the authorized recapture of 2026-08-25 absorbed that move
(`golden_baseline_provenance.md`). The measurement above is unchanged — what
changed is only which document owns it.

**Reference plane: all four are BANK-TO-TAP quantities.** The V_dis probes
terminate at the machine FEEDTHROUGHS (Tom, 2026-08-24), not at the
plasma-facing cathode surface, so the fit above carries the tap voltage on
both sides and is blind to the in-vacuum feed between feedthrough and cathode
(feedthrough-to-cathode conductors, graphite front panel, spring-ring contact
stack). `R_comp` is therefore the bank-to-feedthrough LOOP resistance, not
"everything external to the plasma": the feed's own `R_feed` sits outside it,
one-sided-bracketed to `R_feed` = [0, ~1.0 mOhm] by the cross-rung V_dis
transfer extraction. Values and bars above are unmoved by this — what changes
is what they MEAN.

`R_comp` and `C_bank_F` come from ONE joint fit and must move together.

`L_parasitic_H` was 6.6e-6 until the reconciliation. 6.6e-6 was the orphaned
fourth member of the same retracted free fit as the old `R_comp`/`C_bank_F`,
not a considered choice; it is still inside the plateau refit's jackknife bar
(6.7 +/- 2.5 uH) and is therefore not excluded, only unsupported.

L is inert for every scored row -- and, MEASURED on the production reference
run 2026-08-03, inert on the UNSCORED timing observables too: across the
6.6 -> 8.1 uH change t90 moved 0.00000 ms and ignition -0.0035 ms.

*(Corrected 2026-08-17 in a claims-consistency pass. This paragraph previously
read "its measurable consequences are confined to the unscored reported
fingerprints (t90 +0.05..0.11 ms, ignition +0.02..0.07 ms, both toward the
measurement)." That projection is WITHDRAWN: it extrapolated
d(t90)/d(ln L) ~ 0.24 ms from a 12-vs-20 uH artifact pair, and the
sensitivity does not hold down at 6.6-8.1 uH. The direct measurement above
postdates and supersedes it. This matters more here than in a narrative
document: this note is the authoritative home of every stance value and its
provenance, and the claims record defers to it when the two disagree, so a
retracted number left standing here would have WON that
disagreement. The adoption of 8.1 uH is a pure consistency and provenance
correction with no measurable physical consequence whatsoever.)*

The regression fixture has pinned 8.1e-6 all along, so it is bit-exact
across the change.

## Cathode emission

**`C_R = 9.30` under the `kinetic_dvm` neutral closure — FITTED (the DVM
arm's one-time ES1 drive calibration, 2026-09-01).** **PROMOTED from an arm
override to THE STANCE VALUE on 2026-09-02, in the same event that made the
kinetic closure the stance of record** (see "Neutral closure" below): the
calibration and the closure it calibrates are one decision, and a stance that
named the closure while keeping the fluid closure's 8.76 would run 5.5 % under
the drive band on every arm. `g1atrim.toml` therefore carries `C_R = 9.3` and
the golden is recaptured with it; the sentence this entry used to carry —
"an ARM OVERRIDE, not a stance change; the stance keeps 8.76 and the golden is
untouched" — is superseded, and is kept here only as the record of what was
true between 2026-09-01 and the promotion. The value, its protocol and its
freeze are UNCHANGED by the promotion. The DVM closure at α_E = 0.40 carries its own
drive calibration because the kinetic neutral inventory loads the discharge
differently (plateau 0.945 at the fluid's 8.76). Protocol (pre-registered
2026-09-01, advisor §(b), Tom-ratified): one knob (`cathode_Ts_base_K`
untouched at 1910.0), two-point secant in ln(plateau) vs ln(C_R) measured ON
the DVM, drive-band target 2963 ± 23 A only, frozen after; ES2/ES3 transfer
is the test. Points: 8.76 → 2800.1 A (0.9449); 9.30 → 2963.4 A
(**1.000006**) — the second point IS the landed point; secant
`C_R* = 9.2999` (power-law and linear identical), measured DVM slope
`d ln(plateau)/d ln C_R = 0.948` (fluid 1.069). Port scores reported
unconditionally, never entering the selection: Te 1.02, n 1.03, Isat 1.03
(mean |dev|/σ 1.32 on Isat). Five of six pre-registered predictions held;
the mid-port density statistic moved +0.009 (below the predicted +0.02…+0.06,
same sign — the density is drag-limited, not drive-limited). Artifacts:
`scripts/m1r_retrim_cr9.3_*`; the arm command of record
`scripts/m1r_retrim_cr9.3_es1.cmd`. RIDER (2026-09-01, Tom): the stance
plasma-radius profile was found mis-oriented the same day (see the profile
entry below) and its correction is registered; if the corrected profile moves
the DVM drive out of the 2963 ± 23 A band, this value is re-trimmed ONCE more
under the SAME one-knob protocol and this entry is superseded — the freeze
binds per profile of record.

**`C_R = 8.76` — FITTED (the one drive-side fit knob; re-trimmed
2026-08-25 under the `plateau_multigroup` closure).** Adopting the
multigroup plateau moves the drive, so the knob was re-trimmed ONCE, at ES1,
drive-band target only, and frozen. Protocol (pre-registered 2026-08-25 (Tom);
it supersedes a WITHDRAWN `Ts`-parameterization of the same date): a
three-point scan in `C_R`, then a log-log fit of stage-(i) plateau current
against `C_R` solved for a plateau ratio of 1.000 against the measured 2963 A
as the scorer prints, rounded to 0.01. The scan arms were `C_R` 8.4 → 2832 A
(ratio 0.956), 8.9 → 3013 A (1.017) and 9.4 → 3194 A (1.078); the fit gave
`d ln(plateau)/d ln(C_R) = 1.069481` with `R^2 = 0.99999873` and
`C_R* = 8.762467 → 8.76`, predicting 2962.11 A. The extension clause was NOT
triggered — `C_R*` lies inside the pre-registered bracket [8.4, 9.4] and the
target is bracketed from BOTH sides by scan artifacts. The confirm arm at
`C_R = 8.76` (`scripts/mgcr1_confirm.h5`) measured peak 2982 A against a
measured 2989 ± 23 A (ratio 0.997) and plateau 2962 A against 2963 A
(**ratio 1.000**). Artifact of record: `scripts/mgcr1_fit.md`. Port scores
were reported unconditionally and never entered the selection: Te mean ratio
1.11, n 0.84, Isat 0.88 (mean |dev|/sigma 1.6). A robustness cross-check
re-fitted on the fingerprint plateau means gives the same rounded trim point.
**`cathode_Ts_base_K` was NOT co-trimmed** and sat at its measured config
default 1910.0 on every arm — the flat-direction rule below stands, and only
one member of that pair may carry a calibration.

*(The 2026-08-23 `7.36` event, retained as record.)* **`C_R` was `7.36` —
FITTED, re-trimmed under the conserving ionization birth.** The predecessor `7.09`
was trimmed on 2026-08-19 with the **ionization-birth thermal leak live** —
the En sink gave up `(3/2) k Tn` per ionized atom while the ion was born at
the 300 K floor, deleting **~9.7 kW at plateau** in the source region (9250 W
bulk + 427 W beam, measured on `ph_es1.h5`, 2026-08-23). Adopting
`Ti_birth_ionization = "neutral"` returns that power to the ions and drops the
drive, so the knob was re-trimmed ONCE, drive-band target only, at ES1, and
frozen: peak/plateau ratio 0.965/0.965 at `C_R = 7.09` under the conserving
birth (`tbn_es1.h5`) -> **1.000/0.999 at `C_R = 7.36`** (`tbn2_es1.h5`, peak
2989 A against a measured 2989 +/- 23 A, plateau 2962 A against 2963 A).
Measured pair response over the trim: plateau **2859 -> 2962 A** (peak
2885 -> 2989 A) for `C_R` **7.09 -> 7.36**. The five pre-registered washout
gates (pre-registered 2026-08-23) all passed at the trimmed value: drive peak
and plateau both within +/-0.8 % of 1.000, `n` mean ratio 0.89, `Isat` mean
ratio 0.88 (>= 0.86), `Te` mean ratio 1.01 (>= 0.98). Port scores were reported
unconditionally and never entered the selection: Te 1.01, n 0.89, Isat 0.88
(1.7 sigma). `cathode_Ts_base_K` remains pinned at the measured standby (the
flat-direction rule below stands: only one member carries the calibration).

*(The 2026-08-19 `7.09` event, retained as record.)* Value chain: the 14.25
derivation below was superseded when the
L2 geometry rebaseline moved the operating point to the measured
18.415 cm aperture (the l2a4 free fit landed 7.26 there; the G1
measured-geometry arms ran 6.94); the 2026-08-19 D-phase conservation
fixes (see "Conservation fixes" below) changed the plasma load, and the
REGISTERED one-knob re-trim (drive-band target ONLY, once at ES1, frozen
and transferred; registered 2026-08-19) landed **7.09** — peak
2997 A / plateau 2964 A, ratio 1.000 against the measured band, matched
on the first log-knob interpolation (measured local exponent 0.99966;
the super-linear 1.392 figure is kinetic-era only). Port scores at that
event: n 0.91, Isat 0.84 (1.9σ), Te 0.88 at the matched drive.

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

**`T_s = 1998.15` IS named by the stance, and its deletion was DEFERRED a
third time on 2026-08-25 — now with the reason on record.** It was scheduled for
deletion alongside `V_bank` (23b R4) on the reading that both merely restate
config defaults. That reading is correct against `default_config()`, which is
why deleting `T_s` is invisible to the golden fixture (its `T_s` resolves to
1998.15 either way). **It is NOT correct on the campaign route:**
`run_m6_point.py:216` supplies `T_s` from the ES rung's `Ts_standby_K`
(ES1 = 1910.0) and THIS ROW SUPERSEDES IT, so deleting it would move `T_s`
1998.15 → 1910.0 on every campaign arm. That was MEASURED as a real second delta
in the pre-flight against `scripts/mgcr1_confirm.h5` and disappeared the moment
the row was restored. Whether the move is physically inert was left
open here on a reading the kinetic stance event (2026-09-02) DISPROVED, so the
reading is corrected rather than carried forward. Under
`cathode_warming_model = "power_balance"` the evolving surface temperature is
seeded from `cathode_Ts_base_K` rather than `T_s`, and every read of `T_s` on
the cathode path is guarded by that evolving value or by the `T_s_override_K`
route. The earlier text named ONE unguarded read — the kinetic background's —
and predicted it would go LIVE under `neutral_model = "kinetic_dvm"`. **It does
not.** That read belongs to `neutral_model = "kinetic"`, the TPMC path, whose
`_kinetic` namespace the solver builds only for that selection; under
`"kinetic_dvm"` it is `None` and the read is unreachable. The DVM's own
surface-temperature input is taken from the same evolving cathode value, with
`T_s` only as its fallback. So `T_s` is configuration-inert under the kinetic
stance too, the row is kept because deleting it would move `T_s` on every
campaign arm, and the deferred adjudication is now a bookkeeping question
(which of 1998.15 and the rung's 1910.0 the stance should name) rather than a
physics one.

It remains
one of the three RESOLVED-ACCRETION keys (`cathode_emission_profile`,
`Te_birth_ionization`, `T_s`): keys that equal their config default but are
stated explicitly in `g1atrim.toml` anyway, because `run_m6_point.py`'s own
driver defaults do not. Under the stance's
`cathode_warming_model = "power_balance"` it is ONLY the initial surface
temperature — the surface then evolves from the power balance, so `T_s` sets no
steady-state property and `cathode_Ts_base_K` is the live key below. *(Formerly
listed under "Deliberately absent" on the mistaken reading that the stance does
not name it; corrected 2026-08-23.)*

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
fitted 15.0 cm to the design-spec 18.415 cm aperture scales them by
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
(particle/fueling-side, not a power artifact). Measured and adopted
2026-08-19 (Tom); this section is the statement of record for both
conservation fixes. The production GEOMETRY is the G1a measured-machine
package (grid of record nx 268 / Lm 2117.8 / collector 7.8 + the sss
fidelity rulings; coils-ON droop-min ansatz f_end 2.221 — always quoted
as the ansatz, never "coils on" generically).

## Anode surface recycle (2026-08-21 physics batch — DISARMED 2026-09-02)

**SUPERSEDED BY THE KINETIC STANCE EVENT (2026-09-02), and retained as the
record of the fluid stance.** The three keys below are FORCED OFF under
`neutral_model = "kinetic_dvm"` — they are evolved-`M_n` momentum physics and
the DVM state owns the neutral first moment — so the stance of record now
carries `anode_neutral_jet = false`, `anode_jet_energy_convention = None` and
`neutral_mesh_accommodation = false`, stated in its
`[models.neutral_closure]` block. See "Neutral closure" below; the channel
itself is not given up, it is carried by the DVM's own anode jet
(`neutral_kinetic_dvm_anode_jet = true`), which reads its own `R_N`/`R_E`
pair at the same values. Everything from here to the end of this section
describes the FLUID arm and is the provenance of record for any artifact
produced under it.

**`anode_neutral_jet = true`, `anode_jet_energy_convention =
"total_reflected"`, `neutral_mesh_accommodation = true`** (all params) were
armed in the production stance from the 2026-08-21 physics batch until
2026-09-02. The coefficients they act on (`anode_jet_R_N = 0.63`, `anode_jet_R_E = 0.41`,
DERIVED for He -> Mo from Eckstein IPP 9/132) are surface properties and live
in the config defaults, not here; what the stance decides is that the channel
is ON and which convention its `R_E` is read in.

Class: the ARM is a stance decision; the convention is STRUCTURAL (a guard,
not a value). The jet gives the anode's collected-and-neutralized flux the
directed momentum it physically leaves with instead of rebirthing it at rest,
per collected side — gap-side −z, column-side +z, at
`v_back = sqrt(2 (R_E/R_N)(phi_a + Ti)/m)`. The accommodation sink is the
matching debit: momentum the wires intercept lands on the anode structure. The
two are armed TOGETHER because the sink was built for exactly the opposing-jet
recirculation the jet creates; arming the jet alone would leave that
recirculation artificially elastic.

**Channel is MOMENTUM-ONLY (standing M2 convention).** Neutrals carry no
energy field on this path, so the reflected atoms' kinetic energy is not
booked — an unbooked ~0.2–0.5 kW. This is disclosed as a convention, not
repaired here. The surface power ledger is IDENTICALLY unchanged: the anode
has no analogue of `cathode_jet_surface_debit`.

Pre-declared impact (registered before the capture, so the recapture's
movement can be read against it rather than rationalized after): pre-breakdown
bit-identical and breakdown shift ≈ 0 (the jet needs a solved `phi_a`); anode
momentum ~20–25 % of the cathode channel and OPPOSING it in the gap, so gap
`nn` 0 to +5 %; plateau `I`/`Isat` +0–2 % with `V_dis` slightly down; the
mid-machine trough UNMOVED — more than ~1 % movement there is a finding to
investigate, not a number to bank. Because the batch moves `S_pump` in the
same recapture, the observed movement is JOINT and neither member may be
credited with it alone.

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
**The 9010 is METER-CLASS and was carried VERBATIM through the 2026-08-21 sccm
changeover.** That changeover re-referenced `SCCM_TO_PARTICLES_PER_S` from the
0 °C chemists' standard to the flow meter's own (Sensirion SFM5500, 20 °C /
1013 mbar), so a configured sccm now MEANS what the meter reported. This value
IS a meter reading, so its digits are exactly the ones to keep — and the
changeover is what finally makes them mean what they say. The delivered
throughput it implies therefore FALLS by 6.85 %: the previously quoted
`[7.2, 8.1]e21` atoms/s restates as **`[6.7, 7.5]e21` atoms/s** at the same
9010. The `f_gp = 5200/9010 = 0.577` line in
`config_defaults_provenance.md` is a ratio of two sccm numbers taken under the
OLD convention and is unaffected as a ratio; the fitted 5200 half of it is
FITTED-FLUX-class and restates as 5582.11 meter-sccm.

The stance no longer names `S_gp_decay_target` (dropped 2026-08-21). It was
read only by the retired pulse/decay/double_erf puff waveforms and the stance
runs `"square"`, so the line was inert and its removal is
trajectory-invariant.

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

### Injection row and the shaped initial fill (2026-08-24 — `[porf-dvm-consistency]`)

**`gas_puff_profile = "orifice"`, `gas_puff_orifice_id_cm = 3.95`,
`gas_puff_orifice_length_cm = 22.0` — DERIVED, hardware-BRACKETED.** The puff's
axial placement is no longer the fluid cosine lobe. It is the tube-beamed
first-flight row derived from the CAD port station and the collimating feed
pipe behind it: emit over the pipe-exit disc at the vessel wall, weight
directions by the transparent-regime Clausing long-tube angular intensity
(Olander–Kruger parameterisation, Ashkarin et al., arXiv:2605.12212 Eqs. (17),
(22)–(25) — an analytic literature result with no fitted quantity), fly
straight, and record where each ray first reaches the plasma column. The
derivation and its full pin list live in
`cablp/solvers/_sim1d/physics/puff_orifice.py`; the key-level provenance is in
`config_defaults_provenance.md`.

**Why the values are a midpoint and not an endpoint.** The feed pipe is not in
the CAD export, so its two numbers are hardware brackets: bore ∈ [3.8, 4.1] cm
(CF35/KF40 class), length ≥ 22 cm (ONE-SIDED, the cathode-side coil stack the
flange must clear). Both push the footprint the same way, so the bracket has
two unambiguous endpoints — WIDE (4.1 cm at 22 cm) and NARROW (3.8 cm at
L → ∞). On this geometry the two endpoints move the derived foot fill by
**less than 0.1 %** (`scripts/foot_orifice_probe.txt`: mid-port column density
−3.569e-3 wide vs −3.572e-3 narrow against the same control; per-cell max
|rel| 0.3291 vs 0.3340). The stance therefore quotes the **bracket midpoint
bore, 3.95 cm, at the length bound 22.0 cm** as the value, and
**[3.8, 4.1] cm with L ≥ 22 cm remains the claim's bracket**. Nothing here is
fitted: no endpoint was chosen to move a score.

**What the row does to the placement.** At the campaign mesh the 5–95 % axial
span falls from **186.4 cm** (the superseded `cosine_pipe` envelope, throw
100 cm) to **59.81 cm** — the injection footprint is roughly three times
narrower than the deposition envelope it replaces. That difference is the
registered closure finding of `[puff-orifice]`, not an error in either row.
**Disclosed closure:** a kinetic FIRST-FLIGHT row is being read as the fluid
DEPOSITION row, so the transport the kinetic engines would apply after first
arrival is absent (see `MODEL.md`, the fueling section). Also disclosed: the
row is applied UNMASKED, so the 3.87 % of its mass that grazing rays place in
the cathode–anode gap, the plenum and the cathode cell is deposited there
rather than being redistributed onto the main-chamber roles the length-weighted
fluid shapes restrict to (`scripts/g1aporf_rowcensus.txt`). Total inflow is
conserved exactly either way.

**`nn0_profile`, `nn0_annulus_profile` — REGENERATED 2026-08-24 (DERIVED from a
model run, not measured).** The shaped 4.5 ms ballistic foot, rebuilt by
`scripts/sp3_build_nn0.py` on the equilibrated base
`scripts/g1aporf_eqbase.h5` and written to `scripts/g1aporf_foot45.npz`
(`scripts/g1aporf_foot45.cmd` is the verbatim command; `dt_foot = 4.5 ms` is
the pedestal-floor end of the registered [2.0, 4.5] ms bracket, `ballistic` the
short-reach end of the kernel bracket). This is the first provenance row this
pair has had.

**The fill it replaces was STALE, and the staleness is disclosed rather than
absorbed.** `scripts/g1afix_foot45.npz` was built 2026-08-19 and predates two
events: the CAD-span gap re-anchor (`cathode_anode_gap_cm` 50.0 → 53.25), which
moved every cell downstream of the cathode face, and the sccm changeover to the
flow meter's own 20 °C / 1013 mbar standard (`SCCM_TO_PARTICLES_PER_S`
4.477962e17 → 4.171431e17). It also predates every physics and stance event
its equilibrated base was run before — this fixture's own recapture record
lists six.

The staleness has been measured twice, at two different scopes, and the two
numbers must not be conflated:

| scope | how it was measured | result |
|---|---|---|
| the LOBE LEDGER alone, base held fixed | rebuild the fill at the current tip **on the banked file's own base** (`scripts/foot_orifice_probe.txt`) | 3.25 cm z-shift; per-cell **max 4.61 %, mean 1.28 %**; delivered inventory ×0.9315 — the sccm re-reference exactly |
| the WHOLE recipe, base rebuilt too | rebuild the base as well, still on `cosine_pipe` (`scripts/g1acos_foot45.npz`, the control in `scripts/g1aporf_foot_diff.txt`) | port 11 **−11.3 %**, mid-port (port 29) **−28.6 %**; the base's chamber-mean rises 4.919e12 → 6.958e12 cm⁻³ |

The second row is the honest size of the drift, and it is roughly **ten times
larger than the profile change itself** (port 11 +2.54 %, port 29 −0.95 % for
orifice against the cosine control). The banked file was therefore never the
fill this stance's geometry and physics imply. `scripts/g1aporf_foot_diff.txt`
reports all three side by side — banked, cosine-at-current-tip (staleness
only), and the adopted orifice fill — because reading the banked file against
the orifice fill directly would attribute the whole drift to the profile.

## Geometry

**`Rp = 18.415`, `R_cath = 18.415`** — DESIGN-SPEC HARDWARE (L2 geometry
rebaseline, 2026-08-18). The cathode is a 15.0 in x 0.25 in LaB6 disc
(R = 19.050 cm) held by a backside carbon ring against a graphite front panel
whose opening is 14.5 in — r = 18.415 cm. The ring's overlap on the disc edge
is 0.6350 cm per side, exactly the disc thickness (0.2500 in): a designed lip
fit, not a measured coincidence. The exposed aperture, not the disc, is the
emitting, collecting and conducting face, so both radii identify with it and
are equal by design rather than by the previous coincidence of one fitted
15.0. Mapping the aperture along the field to the plasma-column radius is
ASSUMED 1:1 (recorded ruling: no flux-tube expansion or compression is
modelled between the cathode face and the column).

**Honest bar: the two-spec design bracket `[18.10, 18.415]` cm.** The 14.5 in
opening is one engineer's design spec; a second engineer's CAD gives 18.10 cm
(a 14.25 in-class opening). No as-built number exists — the machine was
inaccessible — so the spread between the two specs is the systematic that
rides every aperture-sensitive claim. The live value 18.415 is the UPPER spec,
adopted as the stance value; it is not a measurement of the machine.

*Formerly labelled "MEASURED (caliper, 2026-08-17)"; corrected 2026-08-18 —
there was no caliper measurement.* Both values were 15.0 before the L2
rebaseline — a fit, not a measurement, and one that conflated the emitting
radius with the plasma-column radius. Since the R2b re-anchor (2026-08-20) the
golden is captured at the stance of record and carries 18.415.

**`neutral_baffle_positions_cm = [342.65]`, `neutral_baffle_clear_radii_cm =
[39.75]` — MEASURED (machine CAD).** The one modelled neutral baffle: a
port-7 annular ring, `TomLook-Aperature` in the CAD, whose clear aperture
throttles axial neutral flow past it. **Reduction rule: CAD-span midpoint.**
The CAD gives the ring as a SPAN in `z`, `3.401–3.452` m — it has a physical
thickness — and the value of record is that span's midpoint, `3.4265` m.
**Honest bar: the span itself, `[340.1, 345.2]` cm.** *(The `342.6` of record
was this same midpoint, stale-rounded; the 2026-08-23 ruling adopted the
midpoint rule at both CAD-span sites and `342.65` is what it gives here. The
correction is documentary at both shipped meshes: the baffle is applied at
the nearest cell FACE, and `342.6` and `342.65` snap to the SAME face at
`nx = 60` and at `nx = 268`, so no trajectory moves with it — see
`golden_baseline_provenance.md`.)*

**What the ring IS — identity confirmed by Tom, 2026-08-24.** It is a MOUNT
FOR AN IRIS LIMITER, installed for biasing experiments. It is modelled as a
neutral baffle on PHYSICAL-OBSTRUCTION grounds: what the model needs from it
is the clear area it leaves at that `z`, and the ring presents that area
whether or not an iris is mounted in it. Its intended purpose is immaterial
to the transport role, and no claim rests on it being a baffle by design.

**`39.75` cm clear radius — MEASURED (machine CAD), no reduction applied.**
The ring's inner radius. It must leave the local plasma channel fully open
and lie inside the local vessel radius, and it does at both: the column is at
`Rp = 18.415` cm there and the bore is 50.0 cm.

**`plasma_radius_profile_cm` — MEASURED (machine-state field record), adopted
2026-09-01 (Tom) and RE-REGISTERED the same day.** The per-cell flux-tube
radius under `prescribed_area_geometry`, one entry per cell of the stance's
280-cell mesh. It is built by `scripts/build_msi_field_profile.py` from the
`MSI/Magnetic field` group the ES1 raw shot files carry: 1024 axial field
samples at a 2.273 cm pitch, recorded at the first and last shot of every run.
Each recorded profile is divided by its OWN plateau level (the median over
z = 300–1500 cm), and the build uses the MEAN normalized shape `B_hat(z)` over
the retained runs. Column-anchored flux conservation then gives
`r(z) = Rp * sqrt(1 / B_hat(z))`, held EXACTLY flat at `Rp` upstream of the
first sustained departure of `B_hat` from 1 (measured at z = 1748.714 cm,
`B_hat` = 0.9792; tolerance 0.02 over a ≥20 cm run searched beyond 1500 cm),
the ratio applied beyond it, capped at `sqrt(0.95) * R_m(z)` — the declared
annulus regularization, unchanged from the census build. 231 of the 280 cells
are at exactly `Rp = 18.415` cm; the cap binds in the terminal 20 cells and the
collector sits on it at 74.2706 cm. **Disclosed:** where the bore steps 50.0 →
76.2 cm the cap RELEASES, and the plasma area takes a +63.4 % step in one cell
— cell 259 clamped at `sqrt(0.95) * 50.0` = 48.7340 cm (7461.28 cm²), cell 260
free at 62.3022 cm (12194.31 cm²), below that cell's own 74.2706 cm cap — so
the step is the cap letting go at the bore staircase, not a feature of the
measured field; the validators accept it.

**THE REGISTRATION — MEASURED (instrument convention) + MEASURED (machine
CAD).** `z_MSI` is NOT the model's axial coordinate. The MSI record is written
in the machine's port-referenced frame (`bapsflib.lapd`): `z = 0` at PORT 53,
the most northern regular port, with `+z` pointing SOUTH toward the main
cathode — `portnum_to_z = 31.95*(53 − port)`. The model's `z` runs the other
way, from the cathode face outward. The two frames are REVERSED, so the
registration is a reflection:

    z_model = C − z_MSI,   C = 1814.67 cm

`C` is the port-53 station in model coordinates, read off the CAD port ladder:
port 2 at 182.67 cm from the cathode face and exactly 53 regular stations at a
32.00 cm pitch. Corroborated at two independent stations — the stance's neutral
baffle at 342.65 cm lands on port 7.00 (7.01 on the nominal map), an OCTAGONAL
RING station (the octa rings sit at ports 7, 13, 19, 24, 30, 35, 41 and 47,
Tom-confirmed), and the anode plane sits at CAD 53.2–53.4 cm against the
model's 53.25 cm.

**The port pitch is the CAD's 32.00 cm, not the 31.95 cm nominal — RULED (Tom,
2026-09-01).** The nominal map `182.5 + 31.95*(port − 2)` puts port 53 at
1811.95 cm; the CAD ladder puts it at 1814.67 cm. Both values are on the
record; the difference is 2.72 cm, well inside one 7.488 cm mesh cell, and the
builder checks that both ladders put every reported port on the same side of
the departure. Nothing in the emitted array turns on the choice.

**END COILS OFF — MEASURED (machine hardware record).** The end-pair magnet
supply is set by hand and its channel is auto-recorded in every shot file, and
it reads 0 A (−0.12 … −0.10 A, the recorder's offset) in all 64 recorded shots.
There was no end mirror in this data set: the far end simply falls off. This is
a machine fact from the record, not an inference from the profile shape.

**The measured level is not used, only the shape.** The plateau level varied
~3 % across the 32 files with the main-supply setting, which is exactly what
the ratio divides out. Two runs (32, 34) ran the lowest main-supply currents
and are candidates for exclusion; the adjudication is a leave-one-out
comparison of normalized SHAPE on the FAR END (`z_model ≥ 1500` cm — the span
the flux ratio is actually read on) against the largest deviation any other
file shows there. RE-ADJUDICATED under the corrected registration: run 32
deviates 0.0251 against the others' 0.0205 and is EXCLUDED; run 34 deviates
0.0122, inside the others' spread, and is RETAINED. The mean shape is built
from 62 shots over 31 files, and the exclusion moves it by at most 0.00095. The
adjudication is measured and printed by the builder, not assumed.

**Honest bar: what is pinned, and the one residual.** The registration is no
longer an assumption — it is the instrument's documented convention closed onto
the CAD ladder, corroborated at the two independent stations above — so the
coordinate question this entry used to carry is CLOSED. Two independent checks agree with it: the measured fall-off
tracks the census `off` case (end pair unpowered, which is the state the 0 A
channel records) to a few cm all the way down — same-rule departure +5.91 cm,
and the `B_hat` = 0.90 / 0.50 / 0.10 crossings at −2.12 / +0.37 / +4.45 cm —
with the two profiles differing by at most 4.05 cm over the whole mesh; and the
flux-tube ratio the emitted array implies between p50 and p41 is 1.0000 against
the measured 0.9905 ± 0.0114, a 0.83σ agreement, which the builder gates on.

**The p50/p41 width ratio 0.9905 ± 0.0114 — MEASURED.** Its instrument is the
geometric-mean face-reconciled effective-radius estimator on the ES1 radial
scans, and the ± 0.0114 is that estimator's sample ⊕ shot uncertainty. Carried
SEPARATELY is the face systematic: the two faces bracket the ratio at
0.987–1.022, which is not folded into the sigma above and which takes the
originating measurement from 5σ statistical to roughly 3σ. The builder's
acceptance gate is a ±1σ POINT test on the constant —
`scripts/build_msi_field_profile.py` refuses an emitted profile whose implied
p50/p41 area ratio differs from `MEASURED_P50_P41_FLUX_RATIO` by more than
`MEASURED_P50_P41_FLUX_RATIO_SIGMA` — so the σ figures quoted around it are
distances on that pair, and the gate does turn on the point value.

The residual that remains is the SOFT DEPARTURE. The flat hold ends on a 2 %
tolerance, at z = 1748.71 cm, which is 30.0 cm downstream of p50 (CAD 1718.67
cm). The mean shape crosses 1 % below plateau at z = 1719.04 cm — within 0.4 cm
of p50 — so p50 sits at the very edge of the flat column on a softer rule than
the adopted one, and the +30.0 cm of hold beyond it is a rule choice rather
than a measurement. What it is worth: applying the ratio at p50 instead of
holding it flat would give `r` = 1.0050 × `Rp` there and an implied p50/p41
area ratio of 1.0085, which is 1.58σ from the measured 0.9905 ± 0.0114 against
the flat hold's 0.83σ — so the residual runs in the direction that favours the
hold, and no claim here turns on it. It is disclosed because the next station
out, p51 at 1750.67 cm, falls on the other side of the departure.

**The source-side structure is REPORTED, NOT APPLIED.** In the corrected frame
the cathode side carries the source-coil structure: a peak `B_hat` = 1.0667 at
z = 23.49 cm and a dip `B_hat` = 0.9052 at z = 96.23 cm. The flat hold covers
all of it, so none of it reaches the emitted profile — the same z-threshold
rule that keeps the cathode, plenum, gap, puff and anode-flanking cells at
exactly the scalar `Rp` the cathode and anode-transparency sites read. Whether
the source region should carry per-cell structure is a SEPARATE question and is
not decided here.

**What this changes against the mis-oriented build of the same day.** 39 of the
280 entries move, indices 222–263; every cell at or below 221 was already at
`Rp` and is byte-equal. `machine_radius_profile_cm` is untouched. The earlier
build assumed `z_MSI == z_model` and was therefore mirror-imaged end-for-end,
which put each end of the machine onto the other. Its readings, recorded here
as SUPERSEDED: a 1.050× `Rp` flare at p50 (+10.3 % in area) — that was the
cathode-side dip reflected onto the far column, and p50 is in fact flat at
18.415 cm like p11–p41; a THROAT at 17.832 cm at z = 1791.8 cm — that was the
cathode source-coil peak reflected onto the far column, and the corrected
profile never goes below `Rp` anywhere, as an end-coils-off machine cannot
compress the tube; and a "~156 cm coil-location disagreement" between the MSI
end-pair peak and the CAD census end-pair centroid — that was an artifact of
the reflected frame and is RETIRED, not carried forward. Its shot exclusion
(both 32 and 34) was measured on `z_MSI ≥ 1500` cm, which is the CATHODE region
in model coordinates, and is superseded by the re-adjudication above.

*(SUPERSEDED 2026-09-01: the CAD-census droop_min profile.* Until the MSI
adoption the shipped array was the `droop_min` case of
`scripts/g1_build_profiles.py` — a traced flux surface through a
finite-element re-solve of the drawn coil set, flat to 1855 cm and flared on
the traced ratio beyond it. That build is RETAINED, unchanged, and is now the
independent CROSS-CHECK; it also still owns `machine_radius_profile_cm`, which
neither the adoption nor the re-registration touches. Note that `droop_min` is
NOT the case the machine was in — the end pair read 0 A — so the meaningful
cross-check is its `off` case, which is the one the corrected registration
agrees with.)*

**`machine_radius_profile_cm` — MEASURED (machine CAD), unchanged.** The
vessel bore staircase, still built by `scripts/g1_build_profiles.py`
(40.0 cm source chamber, 50.0 cm main shell to the 19.65 m step, 76.2 cm far
source chamber, with the cathode-box cells carrying the annulus-area-equivalent
radius that reproduces the measured clear areas 1350.1 / 1847.6 cm²). The
2026-09-01 adoption changed the plasma profile only; this array is byte-equal
across it.

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

**`source_region_length_cm = 103.25`, `source_region_dz_cm = 10.0`** (with
`source_fixed_grid = True`) — ASSUMED, interim geometry. *(All three FOLDED to
config defaults at R2a, 2026-08-20; no longer stance keys.)* The 50 cm column in
front of the anode is meshed at exactly 10 cm regardless of `nx`, so refining
`nx` refines only the far column and no longer moves the source cells or the
puff cell underneath the source terms. **The region END is not an independent
number: it is the anode face plus that 50 cm span**, so it moved
`100.0 -> 103.25` with the measured `cathode_anode_gap_cm` (2026-08-23 ruling)
and would move again with it. The span and the source cell count are what is
ASSUMED here; the offset is MEASURED and lives with the gap in
`config_defaults_provenance.md`. All three must travel together: the
geometry is presence-gated both ways and raises if either parameter is set
without the flag. `nx_gap` is deliberately not pinned — it is already the config
default, so it never appears in the delta.

## Neutral closure (2026-09-02 kinetic stance event)

**The stance of record adopted the transient discrete-velocity neutral model.**
`neutral_model` moves `"moment"` -> `"kinetic_dvm"`: the neutral field is
carried as a distribution `f(z, v_z, v_perp)` on the column and the annulus,
advanced on its own clock, instead of by the two-moment fluid. The selection
was made on the completed ES1 program on the measured field profile; the
drive-side consequence is the `C_R` promotion recorded under "Cathode emission"
above, which is the ONE knob that moved with it.

**What may be claimed is the CLOSURE FAMILY, not a prediction.** The fluid
two-moment closure is the other arm of the same bracket and remains reachable
(`neutral_model = "moment"`), so a result must state which closure it ran.
Adopting one as the stance of record does not retire the other, and no result
may read the adoption as evidence that the kinetic closure is correct.

### The selection and the thirteen keys it forces

| key | stance value | class | honest bar |
|---|---|---|---|
| `neutral_model` | `"kinetic_dvm"` | CLOSURE SELECTION | bracket-is-the-claim; the `"moment"` arm is the other member |
| `neutral_momentum` (flags) | `false` | FORCED | no freedom — the DVM state IS the neutral first moment |
| `neutral_energy` (flags) | `false` | FORCED | no freedom — the frictional book needs a fluid wind |
| `neutral_hot_internal_wall` (flags) | `false` | FORCED | presence gate on the hot channel `neutral_energy` creates |
| `neutral_hot_birth_drift` (flags) | `false` | FORCED | the same gate |
| `neutral_momentum_radial` | `"uniform"` | FORCED | closes the radial profile of the evolved wind |
| `neutral_knudsen_temperature` | `"frozen"` | FORCED | `"local"` needs an evolved per-cell `Tn` |
| `cathode_neutral_jet` | `false` | FORCED | fluid `M_n` momentum physics |
| `cathode_jet_surface_debit` | `false` | FORCED | reads the fluid jet's `R_E` |
| `cathode_jet_energy_convention` | `"legacy"` | FORCED | `"total_reflected"` rescales the fluid jet's launch energy |
| `cathode_jet_hot_carrier` | `false` | FORCED | needs the fluid jet and an `En` field |
| `anode_neutral_jet` | `false` | FORCED | fluid `M_n` momentum physics |
| `neutral_mesh_accommodation` | `false` | FORCED | fluid `M_n` momentum physics |
| `anode_jet_energy_convention` | `None` | FORCED | reads an anode jet that is off |

**FORCED is not a provenance class and is not a choice.** Every row below the
selection is the value `core/model_families.py` REQUIRES under
`neutral_model = "kinetic_dvm"`, measured by construction — arm the selection,
read the refusal, clear the key it names — and the solver refuses any other
value at construction. They carry no bar of their own because there is nothing
to be uncertain about: they are the selection restated key by key. Eleven of
the fourteen rows equal their config default and would resolve to these values
in silence; they are written into the stance's `[models.neutral_closure]` block
because a declaration block is an inventory of the whole decision. The three
that are NOT defaults are `neutral_momentum`, `neutral_energy` and
`neutral_hot_internal_wall`, all shipped ON.

**Consequence for the fluid jets' surface pairs.** `cathode_jet_R_N = 0.34`,
`cathode_jet_R_E = 0.18`, `anode_jet_R_N = 0.63` and `anode_jet_R_E = 0.41`
are no longer named by the stance file. They are inert with the fluid jets off,
they are unchanged at their config defaults, and their provenance is unmoved —
`config_defaults_provenance.md` is their home, and the DVM's own jets read the
same numbers through their own keys.

### The DVM's own controls

| key | stance value | class | honest bar |
|---|---|---|---|
| `neutral_kinetic_dvm_accommodation` | `0.40` | MEASURED (literature-boxed) | bracket `[0.35, 0.46]` |
| `neutral_kinetic_dvm_wall_reflection` | `"diffuse_elastic"` | CLOSURE SELECTION | TMAC bracket `[alpha_E, 1]`; the two shipped values span it |
| `neutral_kinetic_dvm_nvz` / `_nvp` | `64` / `24` | ASSUMED (numerical) | discrete-equilibrium offset, below |
| `neutral_kinetic_dvm_cadence_s` | `3.125e-6` s | ASSUMED (numerical) | first order by construction; below |
| `neutral_kinetic_dvm_cathode_jet` | `true` | CHANNEL ARM | the `false` arm is thermal-only recycle |
| `neutral_kinetic_dvm_anode_jet` | `true` | CHANNEL ARM | the `false` arm is thermal-only rebirth |
| `neutral_jet_arm_current_A` | `50.0` A | ASSUMED | 3.5 % above the measured 48.3 A onset maximum |
| `neutral_jet_disarm_current_A` | `25.0` A | ASSUMED | anti-chatter 2:1 band, not a second threshold |
| `neutral_kinetic_dvm_baffles` (flags) | `true` | STRUCTURAL ARM | adds no coefficient — the clear radius is measured CAD |
| `neutral_two_zone` (flags) | `true` | PREREQUISITE | equals the config default; the closure requires it |

**`neutral_kinetic_dvm_accommodation = 0.40` and
`neutral_kinetic_dvm_wall_reflection = "diffuse_elastic"` are CONFIG DEFAULTS,
restated by the stance as class-1 declarations and NOT re-derived here.**
`config_defaults_provenance.md` is their one authoritative home: it carries the
`[0.35, 0.46]` box (the Sandia parallel-plate programme plus the 304 SS
re-confirmation), the disclosure that the non-accommodated 60 % share is what
makes the reflection selector live at all, and the TMAC bracket the two
spectrum values span. The stance names them because the closure IS this pair of
choices and a stance that ran them by silence would not read as having made
them; it adds no evidence of its own.

**`neutral_kinetic_dvm_nvz = 64`, `neutral_kinetic_dvm_nvp = 24` — ASSUMED,
numerical.** A refinement of the shipped `48 x 12`, adopted with the closure.
Honest bar, taken from the defaults note's own measurement on the shipped grid:
the shared velocity axis is stretched to hold the ~10 eV charge-exchange tail,
so a 300 K gas occupies few nodes and the operator's discrete equilibrium sits
away from the continuum Maxwellian — about `2.4e-2` in both the density split
and the temperature at `48 x 12`, with `96 x 32` taking the temperature offset
to `8.4e-3` while the density split converges to `~2.1e-2`. `64 x 24` lies
between those two measured points and its own offsets are NOT separately
recorded here; `scripts/verify_sim1d_k2_dvm.py` (gate L4) is the instrument
that measures them and is the pointer of record. `nvz` must be EVEN — an odd
count places a node at exactly `v_z = 0`, which neither transports nor mirrors.

**`neutral_kinetic_dvm_cadence_s = 3.125e-6` s — ASSUMED, numerical, and no
longer the shipped placeholder.** The neutral clock's tick. The package default
`2.5e-5` s is explicitly PROVISIONAL and its own defaults entry says no result
may present it as accuracy-chosen; this stance value comes instead from the DVM
program's B0c cadence-and-grid convergence ladder, whose harness is
`scripts/verify_sim1d_b0c_cadence.py` and whose registered structure that file
states in full. What the ladder can and cannot establish, stated honestly:
between ticks the plasma consumes a ZERO-ORDER HOLD of the DVM, a Lie-type
splitting whose global error is `O(dt_n)` — FIRST order BY DESIGN, so nothing
higher is claimable — and the registered question is whether the coupling
behaves as designed and at which cadence the truncation error falls under the
registered 1 % bar. Particle conservation is cadence-independent by
construction (the counted handshake), so the conservation rows there are a
consistency monitor and never the convergence signal. Honest bar: an eightfold
refinement of a provisional placeholder, selected against that 1 % bar rather
than derived; a result that turns on the exact tick is a result about the
cadence and owes the ladder.

**`neutral_kinetic_dvm_cathode_jet = true`, `neutral_kinetic_dvm_anode_jet =
true` — CHANNEL ARMS.** The kinetic closure's form of the directed-recycle
channels the fluid `cathode_neutral_jet` / `anode_neutral_jet` carried and that
the block above must turn off. Each splits the COUNTED recycle at its electrode
into an energetic backscatter share, born as a directed volume birth carrying
`(R_E/R_N)(phi + Ti)` per atom, and a thermal remainder; the cathode share's
energy is debited from the cathode surface's own power balance in the same
accepted step, as a named `backscatter` ledger row, so it is created once. The
`R_N`/`R_E` pairs they read are config defaults and are not named by the
stance. Honest bar: these are ARMS of a closure family, not calibrated values —
the `false` arm (wholly thermal recycle) is the other member, and a result must
say which it ran.

**`neutral_jet_arm_current_A = 50.0`, `neutral_jet_disarm_current_A = 25.0` —
ASSUMED, and NOT re-derived here.** These are the registered arm values that
`config_defaults_provenance.md` already records against the inert `0.0` / `0.0`
shipped pair; what changes on 2026-09-02 is only that the stance now names them
instead of each arm setting them explicitly on its command line. That note's
statement of the bar stands unchanged: 50 A sits 3.5 % above the measured
48.3 A breakdown-onset maximum so that ordinary step-to-step variation cannot
arm the jet, 25 A makes the band a 2:1 anti-chatter ratio rather than a second
physical threshold, neither number is itself measured, and a result that turns
on the exact placement of either is a result about the convention. The A/B
against `arm = 0` is what separates the two.

**`neutral_kinetic_dvm_baffles = true` (flags) — STRUCTURAL ARM.** The measured
annular baffles already named in `[input_dict]` act on the DVM annulus as well
as on the fluid neutral field: the annulus flux crossing each baffle face is
intercepted in the blocked fraction of the face area and re-emitted at the wall
temperature in the cell it came from, particle-conserving, in the anode mesh's
channel form. The column is untouched, because a baffle's bore is at least the
local plasma radius. It adds NO coefficient — the clear radius is the
geometry's measured CAD value, whose provenance is the geometry section below —
so there is nothing here to calibrate.

**`neutral_two_zone = true` (flags) — PREREQUISITE, equal to the config
default.** The kinetic closure carries column AND annulus and REQUIRES this
flag; the solver refuses `kinetic_dvm` without it rather than resolving it,
because turning it on silently would arm physics the caller did not ask for.
The stance names it as a class-1 declaration so that the prerequisite is stated
rather than inherited, and so that the campaign driver's `--two-zone` flag is
redundant rather than load-bearing. It carries no provenance of its own.


## Transport and closures

**`atomic_rate_model = "adas"`** — published inputs, not tuned. The ADAS
cooling coefficients are radiation-only and are applied at unit scale: the
`b_Qei` / `b_Qen` / `b_Qcx` scale factors this note previously recorded at 1
were removed from the configuration surface on 2026-08-28, so unit scaling is
now structural rather than a setting.

**`b_beam_excitation = 1.4`** — ASSUMED. 1.0 books the He 2^1P channel alone and
the extra 0.4 approximates the rest of the singlet manifold. It is inert under
`beam_deposition_model = "csda"`, which uses the measured manifold knob-free.

**`heat_flux_limiter_f = 0.45`** — **BOXED (literature), NOT FITTED. Bracket of
record [0.32, 1.5].** The free-streaming cap on the parallel electron heat
flux. It combines harmonically with the Braginskii flux at
`heat_flux_limiter_exponent = 1`, which is already the config default.

**Convention (required whenever this number is quoted or compared):**
`q_sat = f * n * Te * v_the` with `v_the = sqrt(Te/m_e)`. Fundamenski 2005
(`alpha p v_t`, `v_t = sqrt(T/m)`) and Malone, McCrory & Morse 1975 eq. (1)
use this convention exactly — conversion factor 1.000. Cowie & McKee use
`sqrt(2kT/pi m)` and convert in by `sqrt(2/pi) = 0.7979`.

**Why 0.45 and not something else.** It is the unique literature value that is
simultaneously an FP/PiC-matched ELECTRON coefficient in our exact harmonic
form for a sheath-terminated field-aligned channel, AND inside the derived
free-streaming ceiling fork governing the regime where the limiter acts.
Kinetically-matched in-regime values span [0.45, 1.5]; the derived ceiling fork
spans [0.32, 0.80]; 0.45 is the only actual kinetic computation in the
intersection. **Value-bearing locator (sight-verified 2026-08-23 from the
banked PDF): Fundamenski 2005, PPCF 47 R163 — SS2.6 p. R181 + figure 6
p. R179 ("alpha_chi_e ~ 0.45 for both PiC-sh and LC-BA", K_Te-independent
over 0.01 < K_Te < 1); harmonic form eq. (10a) p. R174; v_t = sqrt(T/m)
p. R166; conversion x1.000. Disclosures that travel with any use: (i) the
0.45 literal is Fundamenski's OWN harmonic-form fit to the Cohen et al.
1994 (Contrib. Plasma Phys. 34, 198; NOT independently read) PiC-sh/LC-BA
sheath-terminated kinetic data — the primary's tabulated multiplier in his
Table 2 (p. R178) is 0.2, a fit-form/convention gap open until Cohen is
fetched; (ii) the uniqueness statement above holds only with its
qualifiers (point-valued FP/PiC computation, harmonic form,
sheath-terminated channel) — Table 2 lists other kinetic values (0.6
class) inside the [0.32, 0.80] fork.** Bracket edges: LOWER `0.319` = the half-Maxwellian one-sided
energy flux with the Spitzer-Harm zero-current factor `epsilon ~ 0.40`
(Cowie & McKee 1977 eq. 7) converted into our convention; UPPER `1.5` =
Fundamenski 2005's recommendation given kinetic boundary conditions.

**What may be claimed: the BRACKET, not the central value.** This coefficient
is still a closure-family bracket for claim purposes — boxed by literature is
not measured by us.

**It was NOT chosen because it scored best, and must not be written that way.**
The scored `f` family (2026-08-21) is FLAT above `f ~ 0.3`, which is exactly
the evidence that the data exerted no pull on the choice. Note also that TIMING
is not flat across the wider family, so an "insensitive to `f`" statement has to
name the observable it applies to.

*(Fold history, which is a separate event from the value: split at R2a,
2026-08-20 — the FLAG `electron_heat_flux_limit` folded to a config default of
`True` while the VALUE stayed stance-side — and rejoined at R2b the same day at
the then-value `0.1`, so the stance file names neither key. The value moved
`0.1 -> 0.45` on 2026-08-21 with its own authorized recapture. See
`config_defaults_provenance.md`.)*

**`heating_anomalous_transport = "plateau_multigroup"`** — **DERIVED**
(adopted 2026-08-25; predecessor `"tail_walk"`). Where the CSDA ray's
anomalous (quasilinear) heating lands. The quasilinear plateau is not one
energy, and this value carries the SPECTRUM instead of a line: in the flux
frame the relaxed distribution is flat over the resonant band, so `dP/dE` goes
as `E` from the plateau edge `E_1` up to the beam energy `E_b = e*phi_c`, and
the bank splits into its two heirs — a wave/bulk share `(E_b - E_1)/2E_b`
banked locally and a streaming share `(E_b + E_1)/2E_b` split into `N`
equal-power groups with `E^2`-uniform edges, each walked on the same Coulomb
machinery `"tail_walk"` used. **Zero fitted parameters**: the shares, edges and
weights all follow from the flat plateau, and `E_1` is a state-dependent
bisection solve against the launch cell's own 1D-reduced Maxwellian, not a
dial. That is the class argument — the two single-line predecessors
(`"local"`, `"tail_walk"`) are this closure's two heirs taken one at a time,
which is why the tail-energy dial, the fixed rung and the keying selector are
INERT under it and are REFUSED at construction rather than silently ignored.
The derivation is the advisor's two-heirs argument (2026-08-25).

Adoption verdict (ruled 2026-08-25 (Tom)): the closure was adopted on
**3 of 4 bands** plus the toll discriminator, and the drive was restored by the
`C_R` re-trim above rather than by any parameter of the closure itself.
ENERGY-ONLY in the same sense as `"tail_walk"` — ionization events, the
particle rows and the circuit currents are unchanged — and it inherits that
value's tail-ionization, cathode-boundary and end-ledger conventions unchanged,
which is why `heating_anomalous_tail_ionization = "on"` rides it untouched.
**What may be claimed is the closure family, not a prediction**: a result must
state which value it used. Not supported under `coverage_closure`, which raises.

**Run-cost consequence, disclosed:** the golden fixture's 4,000-step digest
horizon costs ~8.8× more than under `"tail_walk"` and the full fixture runs
94,044 steps against 70,408, because the adaptive `dt` falls. See
`golden_baseline_provenance.md`.

**`anode_sheath_full_debit = true`** (flags) — ARMED 2026-08-25; the ARM is the
stance decision, the booking itself is a CORRECTION rather than a calibration.
The plasma electron store is debited `(2 Te + phi_a)` per electron the anode
collects — the sheath-edge energy flux of the truncated Maxwellian whose zeroth
moment the sheath solve already closes on — instead of the plasma-thermal
`2 Te` alone, and the anode mesh's Bohm collection rows are re-cut to their
sheath-edge values in the same arming. The circuit/load ledger is IDENTICALLY
untouched: the `phi_a` those electrons pay is field energy the loop and the
anode ions already book. The electron-ATTRACTING regime (`phi_a <= 0`) keeps
the unarmed thermal-only booking, because there the bank pays the fall, and
that branch is COUNTED rather than silent (`anode_attracting_steps`). It
completes the thermal-only electrode routing, which became unconditional when
the legacy volumetric absorber and its `characteristic_boundary` selector were
RETIRED 2026-08-31 (Tom); the construction-time `ValueError` that used to
refuse the unarmed-routing combination went with them, the combination no
longer being constructible. Adoption gates: pre-registered and passed at
adoption, 2026-08-25 (Tom).

**`beam_deposition_in_heat_substep = true`** (flags) — ARMED 2026-08-25; the
class-1 fix for the staged Te bias. The beam's electron-energy deposition row
leaves the explicit operator A and is applied by the implicit heat substep B,
held constant over each substep and solved together with the tridiagonal
conduction operator. Nothing else about the beam moves: the ionization births,
the ionization cost and the excitation radiation are reaction-channel terms and
stay in A. **Honest bar:** it changes the A/B commutator, so `NUMERICS.md`'s
split-order table is stale under it until re-measured, and
`scripts/verify_sim1d_order.py` CANNOT certify it — that harness measures the
split step in a deliberately cathode-free regime where the beam deposition row
is identically zero and this flag therefore changes nothing. Gate record:
armed and gated 2026-08-25 (Tom).

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

## Afterglow dt cost (2026-08-26 stance event — shipped config defaults)

Two timestep-controller keys flipped from off to armed as one adoption. They
are **config defaults, not keys of `g1atrim.toml`** — the stance file is
untouched by this event — and they are recorded here because flipping them was
a stance change and a golden re-anchor in the same event (`golden_baseline_
provenance.md` carries the recapture). Registration and gates:
`AFTERGLOW_DT_COST_REGISTRATION_2026-08-26`; adjudication:
`AFTERGLOW_DT_COST_A3_ADVISORY_2026-08-26` (both in the local docs repo).

**`surface_loss_floor_exempt_exit_rtol = 0.1`** (params, was `0.0`) — the
outer re-admission threshold of the hysteresis band on the `surface_loss`
floor-exempt drain bound; entry stays at `SURFACE_LOSS_FLOOR_EXEMPT_RTOL`
= 1e-3.

**`dt_growth_recovery_patience = 4`** (params, was `0`) — arms the built
accelerated dt-growth re-approach, which consequently makes
`dt_growth_recovery_factor = 4.0` live at its own default.

**Class: DERIVED, both.** They are numerics knobs, not physics: an
A/B-selected pair, chosen against gates pre-registered before any run and not
tuned against data. The claim under test was COST, not correctness — the
adopted-stance afterglow spent 34.1 % of its steps inside a 2 ms window
(7.7 % of sim time), owned by knife-edge floor-exempt re-admission amplified
by the ×1.25 re-approach. The adopted arm is the pair (both on); the band-only
arm was measured and REJECTED on its own gate (A1 1.796× against a 2× bar).

**Honest bar (the corrected two-prong A3 bar, physics-advisor 2026-08-26).**
The registration's A3 bar — "no increase above measurement noise" — is
ill-posed for deterministic runs, where measurement noise is exactly zero, and
was superseded on the record rather than reinterpreted. The bar the floor
honesty is actually held to:

1. **Rate invariance.** Floor injection per unit window time must be
   dt-invariant to within the O(dt) crossing-layer correction. Measured across
   a 2.7× change in step count: 1147.8 → 1181.8 erg/ms (+3.0 %), while
   injection per STEP tracked the mean-dt ratio to 2–3 % (0.1031 → 0.2860 erg,
   2.775 against a mean-dt ratio of 2.695). Total injection ~invariant with
   per-step ∝ dt is the fingerprint of the pinned-drain-refund model, and it is
   what a laundering-free fix looks like. A rise that scaled with dt in the
   TOTAL, or any binding of the conduction floor site, would have failed.
2. **Materiality.** Injection as a fraction of final column thermal must not
   change at the quoted precision: 0.0661 % → 0.0678 % in-window,
   0.2035 % → 0.2052 % whole-run. The 68 erg delta is 2.0e-5 of column
   thermal, ~175× smaller than the 0.35 % column-energy reshuffle the
   timestep change itself causes, and two orders below anything the ES1
   comparison resolves.

**Quoting convention adopted with the flip: floor injection is quoted as
energy per window as a fraction of column thermal, never as clip counts.**
This A/B is the proof — clip counts fell 3.5× while the physical integral was
invariant, so a count-based statement flips sign under a step-size change and
would misinform. Figures of record: 0.068 % window / 0.21 % run.

**Resolution bracket the band buys, and it is the price.** An exempted cell is
not re-admitted to the drain bound until its margin exceeds 10 % of its floor
energy, so `Te` in `[Te_floor, 1.1·Te_floor]` = **[0.1, 0.11] eV** is
drain-unthrottled by design. That interval sits far inside the already
semi-quantitative sub-1-eV afterglow regime, so it adds no new figure caveat,
but it IS the meaning of the number and any near-floor afterglow `Te` read
carries it. The density channel is never exempted at any width, and the floor
itself — not this bound — holds those cells.

**Two consequences that are not physics but must not be discovered later.**
Continuation bit-identity is no longer guaranteed: the exemption latch is run
state and is not carried, so a resumed run starts un-exempt
(`_sim1d/RESTART.md`). And recovering the historical bound is now a two-key
operation — clearing `surface_loss_floor_exempt` while the band sits at its
default raises at construction, by design, with the remedy in the message.

Memo: `AFTERGLOW_DT_COST_A3_ADVISORY_2026-08-26` (docs repo). The A/B arm
numbers above are the runner's BASE-vs-FIX2 transcripts as quoted there.

## Deliberately absent

- `beam_product_transport` — `"local"` is both the stance and the config
  default; the non-default arm must travel with the run it scored.
- Run-cost settings (`tau_afterglow`, `max_steps_action`, `density_dt_fraction`)
  — they buy runtime, not physics, and belong on the command line of the run
  that wants them.
