# Thesis notes: claims, caveats, and open items

Working notes for writing up `cablp` / sim1d. Every number here was measured in
this repo and names the script that produced it, so each claim can be re-checked
rather than trusted.

Last updated 2026-07-18.

---

## 1. The numerical scheme

### What is actually true

The sim1d operator-split step is second-order **only** with all three of:

| setting | value for 2nd order | default |
|---|---|---|
| `implicit_heat_scheme` | `crank_nicolson` or `tr_bdf2` | `backward_euler` |
| `operator_splitting` | `strang` | `lie` |
| `heat_picard_iterations` | >= 1 | `0` |

These are three *independent* first-order error terms — the substep
discretization, the Lie splitting error `dt*[A,B]`, and the conductivity frozen
at `t^n`. Each caps the step on its own, so turning on one or two changes
nothing. Measured with `scripts/verify_sim1d_order.py` at 62 cells,
`t_end = 1e-6 s`:

| picard | splitting | `backward_euler` | `shifted` | `crank_nicolson` | `tr_bdf2` |
|---|---|---|---|---|---|
| 0 | lie | 0.97 | 1.01 | 1.01 | 1.02 |
| 4 | lie | 0.97 | 1.00 | 1.01 | 1.02 |
| 0 | strang | 0.98 | 0.98 | 1.04 | 0.98 |
| 4 | strang | 0.99 | 0.96 | **1.99** | **2.00** |

The notebook runs `crank_nicolson` + `strang` + `heat_picard_iterations = 2`,
costing ~1.45x wall clock (~1.29x at `picard = 1`, which already reaches 1.99 —
the second iteration buys fixed-point margin at large dt, not order).

### A defensible sentence

> The plasma equations are advanced with a second-order operator-split scheme:
> Strang splitting between an explicit SSPRK2 step over the non-heat terms and
> an implicit Crank–Nicolson heat-conduction substep, with the Braginskii
> conductivity Picard-iterated to its midpoint value. The observed temporal
> order was verified at 1.99 by fixed-timestep Richardson refinement, with
> backward Euler held at 0.99 in the same test as a negative control.

The negative control is the part that makes this a *verification* rather than an
assertion. Backward Euler cannot be second order at any dt, so its refusal to
move is what rules out a harness that reports 2.0 whenever features are enabled.
Say it explicitly; it is cheap and it pre-empts the obvious challenge.

### The caveat that has to be stated

**A production discharge does not exhibit second order.** The verification runs
in a deliberately clean regime that a real run is not:

- Floors bind on ~37–43% of cell-visits (`Te0 = Te_floor = 0.1 eV`, so the
  domain *starts* on the floor). Floors are non-smooth projections; local order
  collapses wherever one binds.
- Phase transitions are threshold-triggered (`phase_transition_mode = "current"`),
  which makes the RHS genuinely discontinuous — order is undefined across them.
- The harness also disables the cathode, whose solve caches a continuation guess
  across steps.

So the honest framing is that the *scheme* is second-order and the *production
path* degrades to first order wherever the limiters engage. This is normal —
every plasma transport code has it. Stating it yourself is much stronger than
having a committee member find it, and it costs one sentence.

### If asked "why Crank–Nicolson?"

CN is the conventional choice and is genuinely second-order, but it is **not
L-stable**: its amplification factor tends to −1 as `dt*lambda -> -inf`, so
stiff modes ring at undamped amplitude rather than decaying. This matters here
because the conduction is stiff — measured `|z| ~ 865` in the floor-adjacent
cells and `~1363` in the column at production dt.

`tr_bdf2` is second-order **and** L-stable and beats CN on every axis measured:

| | ringing amplitude* | linear-problem error | conduction floor clips |
|---|---|---|---|
| `crank_nicolson` | 4.93e-01 | 1.10e-04 | 1 / 20,328 |
| `tr_bdf2` | **8.11e-03** | **5.54e-05** | **0 / 19,892** |

\* stiffest grid mode, one step at `dt = 1e-5`.

The useful framing, if this comes up: **TR-BDF2's first stage *is* the
trapezoidal rule, which is Crank–Nicolson under the PDE community's name for
it** (verified bit-for-bit in this repo). Choosing TR-BDF2 would not mean
abandoning CN — it means following the CN stage with a BDF2 stage that
annihilates the ringing CN leaves behind. Being able to say *why* the choice is
contested is a stronger position than "CN because it's standard".

CN's ringing was measured and is harmless **on this configuration**: 6 clips in
121,776 solves, injecting 5.3e-10 of the column thermal energy, confined to the
plasma-launch transient (`scripts/audit_sim1d_floor_activation.py`). That is a
measurement, not a guarantee — it would need re-checking if the grid, dt, or
floors change.

---

## 2. What is measured vs what is tuned

This distinction matters more than the values, and belongs in any table of
parameters or sensitivity study.

**Measured directly on the device** — not free parameters:

- `V_bank = 180 V`
- `R_comp = 0.010 Ohm`

**Hand-tuned proxies** — adjusted until the model produced experiment-like
performance, because neither can be measured directly:

- `T_s = 1975 K` (cathode surface temperature)
- `S_gp` (gas puff rate)

Presenting these in one undifferentiated table invites the reader to treat the
tuned pair as observations. They are fitted quantities and should be varied in
any sensitivity study; the measured pair should not.

### The fudge factors: ~~two~~ three different kinds of wrong

Both historically sat at `0.5` in the notebook and looked identical. They fail
differently, and conflating them in the write-up would be a mistake.

**`b_Qei` — an uncertain coefficient.** Confirmed by the ADAS comparison
below: the He II fit really is a factor ~2 high at 8–20 eV. 0.5 was
*plausibly* right, and now has measured Te-structure.

**`b_Qen` — a bookkeeping overlap, not an uncertainty.** The ADAS comparison
(below) shows the IAEA He I "electron cooling rate" *already contains the
ionization-potential loss*, which the model books again as `I_ion * S_ion`.
`b_Qen = 0.5` was partially compensating a double count. The consistent
radiation-only value is ~0.26, nearly Te-flat.

**`b_ion_neutral_drag` — a missing field.** See §3. No constant is right,
because the thing it stands in for varies by ~10x across a discharge. This is a
model problem wearing a measurement problem's clothes; tuning cannot fix the
functional form. (A per-cell entrainment closure, `ion_neutral_drag_model =
"slip"`, now exists as a middle option; unvalidated as yet.)

### Why the model now runs ADAS rates (`atomic_rate_model = "adas"`, 2026-07-18)

`scripts/compare_rates_adas.py` compares the historical fits against the
OPEN-ADAS GCR '96 effective coefficients (packaged in `cablp/vars/adas`,
provenance in its README). The swap was not a cooling correction — it touched
**every bulk atomic rate in the model**, and the largest error was in the
particle source, not the energy sink. The reasons, in decreasing order of
impact:

1. **Ionization was undercounted by a lot.** The model's direct ground-state
   He rate is **0.08–0.67x the ADAS effective SCD over 2–12 eV** — 0.16–0.35
   at the 3–5 eV the column actually runs, i.e. the model was ionizing at a
   *third* of the physical rate where it matters, converging to 0.98 only at
   100 eV. The missing channel is stepwise ionization via metastables, which
   finite density turns on and a coronal ground-state rate cannot see. This
   was the leading candidate for the ~40% density deficit vs ES1, and
   switching rates (no tuning) halved that deficit (port ratio 0.61 → 0.80).
   It is *known atomic physics*, not a radial-loss proxy, and it is citable.
2. **The IAEA He I cooling fit includes the ionization cost.** Near-coronal
   `PLT(1) + 24.587*SCD(1)` reproduces the fit within ~9–25% over 5–100 eV;
   radiation alone is 2.3–4.4x smaller. With `ionization_energy_cost` on
   (default), running `b_Qen = 1` double-counts that channel.
3. **The He II fit is ~1.9–2x high at 8–20 eV** vs ADAS PLT(2), drifting to
   3.5x at 100 eV (shape ~ `(Te/12 eV)^-0.3` above 12 eV). The fit's
   recombination term is ~1000x *smaller* than ADAS PRB — but both are
   dynamically negligible (cooling time ~0.1 s at 10^13 cm^-3 against ms
   decays), so `icool_recomb = False` stays defensible either way.
4. **Recombination becomes one consistent coefficient.** ACD carries
   radiative + dielectronic + three-body recombination at the actual density;
   the historical `alpha_r + ne*alpha_3` pair agrees with it within ~20% at
   0.5–5 eV but over-predicts 2–5.5x at 0.2 eV, exactly where the afterglow
   tail lives.
5. **The coefficients are (ne, Te) surfaces, not Te-only curves.** The
   coronal-era fits structurally cannot express the density dependence that
   the metastable populations impose at 10^12–10^13 cm^-3; no scalar (or even
   `b(Te)`) correction on them can.
6. **Provenance.** Every rate in adas mode traces to a maintained, citable
   database instead of coefficient arrays with no source comment.

Consequence: `atomic_rate_model = "adas"` selects SCD/ACD particle rates and
PLT/PRB radiation-only cooling — internally consistent with the separate
ionization-cost term, because that split is ADAS's own bookkeeping. `"janev"`
remains the default for historical reproducibility (the golden baseline is
janev and still bit-exact); the ES1 benchmark config runs `"adas"`. Under
`"adas"`, `b_rec_3b` is inert (ACD already contains three-body) and the
`b_Q*` factors are genuine O(1) sensitivity knobs rather than corrections.

Two caveats that must ride along: below ~2 eV both cooling channels are
threshold-killed, so **afterglow decay rates cannot calibrate `b_Qei`/`b_Qen`**
— the afterglow constrains e–i exchange, conduction, surface losses, and the
drag/thermalization closures instead; and the ADAS grid bottoms out at 0.2 eV
(clamped below), while the model's 3-body fit over-predicts ACD by 2–5.5x
*at* 0.2 eV, so the deep-afterglow recombination carries that uncertainty.

### The sentence that matters most

With four free parameters set to 0.5, **agreement with experiment is not
evidence the model is right.** Any figure showing the model "close to" data
while these are set needs to say what was tuned. The version that survives
*"what did you tune?"* is:

> Rate coefficients were scaled by 0.5, within the factor-of-2 uncertainty the
> fits carry at these temperatures.

not

> The model reproduces the measured profiles.

---

## 3. Known model limitations

### No neutral momentum equation

The state is `(n, nn, M, Ee, Ei)` — `nn` is a neutral *density* with no momentum
field. So ion–neutral drag is computed as `-m_i*nu*n*(u_i - u_n)` with
`u_n == 0` hardcoded by omission, and `b_ion_neutral_drag = 0.5` is asserting
`u_n ~ 0.5*u_i` everywhere, for all time.

Two consequences worth stating:

1. **Momentum is not conserved between species.** Drag is a sink on `M` with no
   receiver — it vanishes rather than entering the neutrals. This implicitly
   assumes instant transfer to the wall.
2. **The slip is density-dependent.** A crude balance gives
   `u_n/u_i ~ sigma_in * v_ti * n_i * tau_wall` (the neutral density cancels),
   which runs from ~0.1 at `n_i = 1e11` to full entrainment at `n_i = 1e13`. The
   correct factor therefore sweeps an order of magnitude across a discharge; a
   constant is the one form it certainly does not have.

Rough timescales (He, 300 K) split the verdict on assumption 1. **Radially it is
defensible**: a neutral reaches the wall in ~0.28 ms against a 20 ms discharge
(~72 transits, free-molecular — the neutral–neutral mfp is ~100 cm at
`nn = 1e13`, well past the 35 cm to the wall), so the momentum genuinely does
reach the wall. **Axially it is not**: transit over `Lm = 2000 cm` is ~15.9 ms,
comparable to the discharge, so axial bulk flow does not relax within a shot —
and axial is the only direction the model has.

Unlike the rate factors, this has a concrete in-model fix: add `M_n` to the
state (5 fields to 6), reusing the existing conservative-flux machinery. Drag
becomes `-m_i*nu*n*(u_i - u_n)` with momentum conserved between species. Worth
scoping before publication.

**Measured (2026-07-18): the closure is load-bearing at leading order.** The
`"slip"` entrainment closure — the *local steady state* of the missing `M_n`
equation (ion momentum input balanced against free-molecular wall loss) —
predicts slip ~0.1 in the dense column, i.e. ~5x less drag than the
calibrated 0.5. Swapping it in (ES1 benchmark, nx = 120, all other physics
fixed) moved port density 0.76 → 0.60, Te 1.37 → 1.81, and tilted the axial
density profile hard downstream (0.54 near cathode → 0.93 at the far port,
against flat data). **Defensible axial drag physics under-confines the
plasma.** Since a full `M_n` field reproduces this closure in steady state,
adding it would not rescue agreement (its extras are neutral-wind advection
and end recycling); the constant 0.5 is therefore *calibrated compensation*
for a channel the model lacks — the neutral wind's back-reaction or the
out-of-scope radial physics. Any density agreement obtained with the
constant must be presented as such, not as validated axial confinement
physics. This is the sharpest evidence yet on gate #2.

### The cathode beam: ground-state ionization is fine; excitation radiation is missing

The natural follow-up to the ADAS swap is whether the primary beam suffers the
same undercount — it too uses only the ground-state channel
(`beam_cross = He_EII_cross_lkup(phi_c)` sets both the ionization profile and
the Beer–Lambert deposition of `P_prim`). The two halves split:

- **Ionization: no material undercount.** The thermal-rate error came from
  sub-threshold Maxwellian electrons needing metastable stepping stones. A
  60–180 eV primary sits at the *peak* of the direct cross section, where the
  GCR enhancement vanishes (our rate agrees with ADAS to ~2% at 100 eV).
  Beam-produced metastables are also small: triplet excitation is
  exchange-driven and collapses above ~50 eV.
- **Excitation radiation: real and absent.** The beam excites singlets at
  ~0.26–0.34 of its ionization event rate (1¹S→2¹P alone, from the in-repo
  `He_EIE_cross_DA(b_11s_21p)`; 2¹S/3¹P add roughly another 30–50%). Each
  such collision radiates ~21 eV that the model instead deposits as plasma
  heat. Net: **~10–20% of the beam power is radiated as He I lines, not
  delivered to the plasma**, and the true inelastic energy-deposition length
  is ~25–35% shorter than the ionization-only Beer–Lambert profile. This
  biases the source region, where the bulk of the heat loss occurs.

**Is Beer–Lambert still the right deposition model?** Mostly, with one
regime-dependent caveat. The profile length `1/l_b = 1/l_bi + 1/l_bn` mixes
two different objects: `l_bi` (Coulomb drag on the plasma, a genuinely
continuous slowing process for which an exponential deposition profile is a
fair proxy) and `l_bn` (an inelastic *event* MFP). At main-discharge column
conditions (`n ~ 5e12`, `Te ~ 8 eV`, `nn ~ 2e13`) the Coulomb term dominates
~10:1 (`l_bi ~ 1 m` vs `l_bn ~ 10 m`), so the profile is Coulomb-shaped and
Beer–Lambert is fine. **At breakdown the ordering flips** (`n ~ 1e10`:
`l_bn ~ 5 m` vs `l_bi ~ 600 m`): attenuation is event-dominated, and the
single-event-absorption assumption — a primary "dies" at its first inelastic
collision — undercounts the events a real primary produces. A 150 eV primary
survives roughly `phi_c / <dE per event> ~ 4` inelastic collisions before
falling below threshold, so **beam-driven ionization during breakdown is
undercounted by a factor ~3–5** (the model books `l_b/l_bn <= 1` ionizations
per primary; reality is several). The main discharge is insensitive (thermal
ionization dominates there); the exposure is breakdown timing and the
early-current trace, which are deliberately not benchmark targets. The
right-sized upgrade, if breakdown ever becomes a claim, is a CSDA
slowing-down profile: integrate `dE/dx = nn*(sigma_ion(E)*<dE_ion> +
sigma_exc(E)*E_exc) + Coulomb` along z with the in-repo cross sections,
depositing events per cm until `E` crosses the thresholds — deterministic,
no new data, replaces the absorption weights. Open item #10.

**Implemented 2026-07-18** as `b_beam_excitation` (default 0 = historical
beam, golden bit-exact): the 2^1P cross section joins the beam's inelastic
channels, shortening the Beer–Lambert deposition length, and each excitation
event radiates `beam_excitation_energy_eV` (21.2 eV) via the new
`beam_excitation_radiation` term — energy that previously heated the plasma
now leaves as light. The channels split one absorbed flux, so radiated
events / ionizations = `sigma_exc/sigma_ion` cell by cell (asserted in
`smoke_sim1d.py`). The ES1 benchmark runs `b_beam_excitation = 1.4` (2^1P
plus ~40% for the rest of the singlet manifold — that 0.4 is an estimate,
not a measurement, and is the parameter's main soft spot). Still absent, and
deliberately: triplet/metastable production (suppressed at beam energies)
and a true slowing-down treatment (primaries remain single-event absorbed).

### Resolved source/end boundary (`resolved_boundaries`, default OFF)

Branch `1D_source` builds a spatially resolved machine end behind a master switch:
a neutral-only plenum behind the cathode, an optional annular obstruction, a
resolved cathode-anode gap, and the anode as a partial obstruction. The cathode
surface is the origin (`z = 0`) with the anode at `z = 50 cm`; cathode and anode
are *faces*, not cells. **It is off by default and no published number should come
from it yet**, but it changes enough that the difference is worth stating.

**Legacy is provably untouched.** A golden trajectory captured before any refactor
(`scripts/baseline_sim1d.py --capture`) is reproduced **bit-for-bit** — `max_rel =
0` — at every step of the work, with the switch off. Time-integration order is
also unaffected: 2.00 (TR-BDF2), 1.99 (CN), 1.00 (BE control), floors inert.

> **⚠ SUPERSEDED — the table and sweep below were produced with the historical
> `molecular_flow` neutral transport, which is not a consistent discretization of
> diffusion (see "Neutral transport" below). Correcting it moves resolved peak
> `Te` from 51.6 eV to 23.9 eV, i.e. essentially onto the legacy value.
> Regenerated under `knudsen` on 2026-07-18 — see the table after this one.**

**The resolved model gives a materially different discharge**, as plan §10
predicted. Against the legacy reference on the same production config:

| | legacy | resolved | delta |
|---|---|---|---|
| peak `Te` | 24.5 eV | 51.6 eV | **2.1x** |
| final thermal | 1.914e6 erg | 1.672e6 erg | -13% |
| discharge end | 2.791e-2 s | 2.917e-2 s | +4.5% |

The factor ~2 in peak `Te` is the headline: resolving the cathode-anode gap puts
the ohmic deposition and the sheath losses where they belong instead of smearing
them over a 100 cm lump. This is a *change of model*, not a bug fix on either
side, so it cannot be presented as a correction to previously published numbers.

**Sensitivity** (`scripts/sweep_sim1d_resolved.py --sensitivity`), relative to the
resolved default, ranked by effect on final thermal energy:

| knob | change | thermal | peak `Te` |
|---|---|---|---|
| `eta` 0.358 -> 0.6 (opaque anode) | **-49%** | 51.6 -> 16.7 eV |
| pump elbow `C = 2000 L/s` | **-42%** | 51.6 -> 27.5 eV |
| `S_pump_L` doubled | +21% | 51.6 -> 97.6 eV |
| `Lcs = Rcs = 25` (annular duct) | -8% | 51.6 -> 59.8 eV |
| `Rsup = 10` (support rods) | -0.6% | 51.6 -> 50.4 eV |

Anode opacity and effective pump speed dominate; support rods are negligible, as
expected from a volume-only blockage. **`eta` and the pump path are therefore new
tuning handles of the same order as the existing `b_Q*` factors** — which is a
caution, not a feature: a resolved model with two extra strong knobs is easier to
fit to data and correspondingly weaker as evidence.

**Regenerated under `knudsen` (2026-07-18,
`scripts/sweep_sim1d_resolved.py --sensitivity`, janev rates, nx = 60).** The
ranking survives; the magnitudes shift:

| knob | change | thermal | peak `Te` |
|---|---|---|---|
| `eta` 0.358 -> 0.6 (opaque anode) | **-56%** | 23.9 -> 15.0 eV |
| pump elbow `C = 2000 L/s` | -22% | 23.9 -> 21.8 eV |
| `S_pump_L` doubled | +15% | 23.9 -> 24.2 eV |
| `Lcs = Rcs = 25` (annular duct) | -0.6% | 23.9 -> **33.6 eV** |
| `Rsup = 10` (support rods) | -0.6% | 23.9 -> 24.7 eV |

`eta` still dominates; the pump elbow's leverage halves once neutral transport
is mesh-consistent; and the annular duct now moves *peak `Te`* by +40% at
essentially zero thermal change — a shape knob, not an inventory knob. (Peak
`Te` carries the unresolved column-mesh caveat below.) `eta = 0` fails with
the documented singular-circuit error, as designed.

**Mesh dependence is real and not yet eliminated.** Refining the gap
(`--convergence`, `nx_gap` = 5/10/20, i.e. 10/5/2.5 cm cells) leaves a residual
spread of 1.3% (discharge end), 1.5% (peak `n`), 5.0% (thermal) and 7.1% (peak
`Te`). A presheath correction — applying only the fraction of the `exp(-1/2)`
sheath-edge drop that a cell actually spans, with the depth computed as
`c_s / nu_in` — roughly halves the near-wall sensitivity (peak `Te` spread 15.9%
-> 7.1%, peak `n` 3.2% -> 1.5%) but does *not* improve the global integrals and
marginally worsens them. **Any resolved number should carry a few-percent mesh
uncertainty at the `nx_gap = 5` default.**

### Neutral transport is a consistency problem, not just a tuning one

The historical axial neutral exchange (`neutral_exchange_model =
"molecular_flow"`, still the default) applies the Clausing **duct** formula to
every cell face. That is the wrong object for exchange *inside* a continuous
tube: the implied axial diffusivity is `D = 0.25*v_th*P(dz)*dz`, which **tends to
zero as the mesh is refined**. It is not a consistent discretization of diffusion
— refining the grid removes neutral transport rather than converging to it. The
form is only correct for `dz >> Rm`, and with `Rm = 50` cm the model ran at 19% of
the physical free-molecular value at 30.8 cm cells, 7% at 10 cm.

This matters because the redesign's central claim is about the puff-to-pump
back-path (§1), and that path is governed entirely by this transport. Its
timescale straddles the 20 ms discharge and was being set by the mesh: 14 ms at
30.8 cm cells, 38 ms at 10 cm (back-path effectively closed), against 2.7 ms
physical.

`neutral_exchange_model = "knudsen"` replaces it with Fickian transport at the
Knudsen diffusivity `D = (2/3)*v_th*R`, so `C = D*A/dz`. This is exactly
mesh-independent (verified identical at nx = 60/185/370) and reproduces the
textbook long-tube conductance `(2*pi/3)*v_th*R^3/L`. Thin apertures — the anode
mesh — keep an orifice conductance in series. `molecular_flow` remains the default
so existing results stay reproducible; **resolved runs should set `knudsen`.**

**Resolved results are not mesh-converged even so.** Under `knudsen`, going from
`nx = 60` to `nx = 185` (30.8 -> 10 cm column cells) moves final thermal energy by
+7.5% and `final_time` by -3.5%, but **peak `Te` by +73%** (23.9 -> 41.5 eV). Peak
quantities converge worst, but this is not a usable number: only volume-integrated
quantities are worth quoting from resolved runs at present, and then with a ~10%
bar. Converging the column is an open item.

**`eta = 0` is not a usable limit with `cathode_coupling` on.** The anode sheath
relation is singular at zero anode current — an anode that collects nothing cannot
close the circuit. It is a valid limit for the anode's neutral and heat throttles
only. The solver now says so explicitly rather than failing on an internal `-inf`.

---

## 4. Open before publication

| # | item | status |
|---|---|---|
| 1 | ~~Decay-rate study → is a constant rate factor defensible, or must it be `Te`-dependent?~~ Answered for `b_Qei`/`b_Qen` by the ADAS comparison (§2, §5) — value *and* shape now measured, and the double count identified. Residual: the afterglow decay rates constrain the *other* terms (e–i exchange, conduction, surface loss, drag closures) and currently misfit in axial structure (model τ flat ~2 ms vs measured 3.7 → 0.9 ms along z; `compare_sim1d_es1.py` stage iii) | **re-scoped** |
| 2 | Neutral momentum (`M_n`) — or an explicit, defended justification for the sink closure. The `"slip"` entrainment closure now exists as a middle option; unvalidated | **gate** |
| 3 | Does scheme choice change the physics at all? | **unmeasured** |
| 4 | Resolved boundary: ~~is the ~2x peak `Te` vs legacy physical~~ — resolved as a transport-model artifact; peak `Te` is 23.9 eV vs legacy 24.5 under `knudsen` | **closed (see §3)** |
| 5 | Resolved boundary: converge the gap, or state the few-percent mesh uncertainty explicitly | **gate for resolved results** |
| 6 | ~~Regenerate the resolved sensitivity sweep under `knudsen`~~ — done 2026-07-18, table in §3 | **closed** |
| 7 | Resolved column is not mesh-converged: peak `Te` moves 73% between nx = 60 and 185. **But the ES1 benchmark metrics are**: port-sampled Te/n, peak current, and decay τ move ≤3% between nx = 120 and 185 (≤14% from 60), so tuning proceeds at nx = 120 while *peak* quantities stay unquotable | **bounded; peak quantities still gated** |
| 8 | ~~`sigma_in_cm2 = 5e-15` is a constant~~ — `sigma_in_model = "cx_derived"` now builds the momentum-transfer rate from the in-repo CX table (`sigma_mt ~ 2*sigma_cx` + Langevin floor), consistent with the CX energy channel by construction. The constant crossed the true curve at ~0.5 eV: too small ~1.5-1.8x in the afterglow, too large ~1.3x in the warm column | **closed** |
| 9 | ~~Beam excitation radiation~~ — implemented as `b_beam_excitation` (§3); the residual soft spot is the ~1.4 singlet-manifold factor, which is an estimate rather than a measured manifold sum | **closed; factor is an estimate** |
| 10 | Beam single-event absorption undercounts breakdown-phase beam ionization ~3–5x (§3); harmless for main-discharge targets, matters only if breakdown timing becomes a claim. Upgrade path: CSDA slowing-down deposition | **open, low priority** |

Item 3 is the cheapest and is still open. Every scheme comparison so far has
been uncontrolled: adaptive stepping gives each run a different step sequence,
and the one spread observed (~0.3% in final thermal energy) was **non-monotone
in theta**, which is the signature of trajectory divergence rather than a scheme
effect. A controlled fixed-dt comparison of front profiles would settle it.

A null result there is a perfectly good outcome and worth writing:

> Scheme choice changes the solution by <X% at production timestep; the error
> budget is dominated by the Braginskii closure and the rate-fit uncertainty.

That is a stronger statement than silence, and a committee member may well ask.

---

## 5. Verification available to cite

| claim | evidence | script |
|---|---|---|
| split step is 2nd order | 1.99 / 2.00, with BE at 0.99 as control | `verify_sim1d_order.py` |
| conduction substep is 2nd order | 2.00 (CN), 2.02 (TR-BDF2) on the linear problem | ad hoc; see commit `08eaaed` |
| frozen kappa is what caps order | same schemes: 2.00 → 1.05 when kappa is frozen | commit `08eaaed` |
| conduction conserves energy | drift ~1e-16 relative | commit `08eaaed` |
| ringing matches theory | decay ratio 0.673 vs predicted 0.667 (shifted), 0.991 vs 1.000 (CN) | commit `08eaaed` |
| conduction matrix is correct | 20,268 solves, zero maximum-principle violations under BE | `audit_sim1d_floor_activation.py` |
| floors are safe under CN | 5.3e-10 of thermal energy injected | `audit_sim1d_floor_activation.py` |
| refactors preserved behaviour | backward Euler bit-for-bit at dt = 1e-9, 1e-7, 1e-5 | commits `50b0903`, `08eaaed`, `12bc6ba` |
| boundary redesign preserved legacy | golden trajectory bit-for-bit (`max_rel = 0`) with `resolved_boundaries` off, at every milestone M1-M5 | `baseline_sim1d.py --verify` |
| resolved gap mesh sensitivity | 1.3% / 1.5% / 5.0% / 7.1% spread over `nx_gap` = 5/10/20 | `sweep_sim1d_resolved.py --convergence` |
| resolved knob sensitivity | `eta` -49%, pump elbow -42%, `Rsup` -0.6% on final thermal (⚠ under the superseded transport model) | `sweep_sim1d_resolved.py --sensitivity` |
| Knudsen transport is mesh-independent | implied `D` identical at nx = 60/185/370, equals `(2/3)v_th*R` | `smoke_sim1d.py` |
| IAEA He I cooling fit contains the ionization cost | near-coronal `PLT+I*SCD` matches within ~9–25% over 5–100 eV; radiation alone 2.3–4.4x smaller | `compare_rates_adas.py` |
| He II cooling fit is ~2x high at 8–20 eV | IAEA/PLT(2) = 1.86–1.95 there; 3.5 at 100 eV | `compare_rates_adas.py` |
| direct ionization rate misses the stepwise channel | 0.16–0.35x ADAS SCD at 3–5 eV, 0.98 at 100 eV | `compare_rates_adas.py` |
| ES1 benchmark metrics are mesh-usable at nx = 120 | port Te/n, peak current, decay τ move ≤3% for nx 120→185 (knudsen) | `compare_sim1d_es1.py --nx {60,120,185}` |
| resolved knob sensitivity under `knudsen` | `eta` -56%, elbow -22%, duct -0.6% thermal but +40% peak `Te` | `sweep_sim1d_resolved.py --sensitivity` |
| ADAS rates halve the density deficit untouched by tuning | port `n` ratio 0.61 → 0.80, `Te` 1.04 → 1.32, peak current 0.863 → 0.919 (nx = 120) | `compare_sim1d_es1.py` |
| beam excitation channels split one absorbed flux | radiated events / ionizations = `sigma_exc/sigma_ion` per cell; deposition length strictly shorter; off = bit-exact golden | `smoke_sim1d.py`, `baseline_sim1d.py --verify` |
| CX-derived `nu_in` is what it claims | equals `nn*(2*<sigma v>_cx + k_L)` exactly; constant crosses it at ~0.5 eV; `k_L(He) ~ 7.5e-10 cm^3/s` | `smoke_sim1d.py` |
| slip closure under-confines: drag closure is load-bearing | ES1 nx=120 decomposition: slip alone moves n 0.76 → 0.60, Te 1.37 → 1.81; cx_derived + decoupled thermalization alone move n by −0.04 and decay by +0.09 | `compare_sim1d_es1.py` (+`run_model` extra overrides) |

Note what the order harness does **not** cover: the cathode (excluded — its
continuation cache makes runs history-dependent), the floors (excluded by
construction — it asserts they stay inert), and phase transitions (excluded). It
verifies the scheme, not the production path. Worth saying if it becomes a
figure.

---

## 6. Reproducing the numbers

```bash
conda activate fenicsx-env
cd cablp

# order table
python scripts/verify_sim1d_order.py --picard 4 --splitting strang

# floor activation on the production config
python scripts/audit_sim1d_floor_activation.py --scheme crank_nicolson

# still to run: controlled fixed-dt front comparison (item 3 above)
```

`cablp/cablp/solvers/_sim1d/NUMERICS.md` carries the method details: the theta
map, TR-BDF2's `gamma = 2 - sqrt(2)` and why both its stages share one operator,
the M-matrix argument for backward Euler's monotonicity, and the measured order
tables.
