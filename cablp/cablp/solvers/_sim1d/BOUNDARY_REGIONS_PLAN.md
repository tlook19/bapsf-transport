# Resolved Source Boundary: Plenum, Obstructions, and Split Cathode/Anode Coupling

Design plan for making the sim1d source and end regions physically resolved in
1D. Status: **design, not yet implemented.** This supersedes the lumped 0D
source/end cell model described in [`MODEL.md`](MODEL.md) §"Reductions".

## 1. Motivation

The current model represents each machine end as a single lumped 0D cell (100 cm
by default), with gas puff, pumping, cathode drive, and surface loss all piled
into cell `[0]` / `[-1]`. That collapses four things that are physically distinct
in space:

- **Localized neutral injection.** The puff enters at a specific axial location
  (just in front of the anode), not smeared over a 100 cm lump.
- **The axial neutral pressure gradient.** Gas puffed in front of the anode feeds
  the column forward *and* leaks backward through the anode mesh and the cathode
  obstruction to the pump. That back-path sets how much puffed gas fuels the
  column versus gets pumped — a gradient a 0D lump cannot represent. **This is the
  central physics payoff.**
- **Distinct cathode vs anode plasma.** The cathode and anode sit ~50 cm apart in
  plasma of different density and temperature; the sheath solver should sample
  each locally (see §7).
- **The region behind the cathode.** The machine extends behind the cathode into a
  neutral-only plenum where the pumps live. There is no plasma there.

### Core reframing: the plasma domain is a subset of the neutral domain

Every requirement below is one statement: **neutrals live on a longer grid than
the plasma.** Today they are forced onto the same cells (same `z`-grid, differing
only by `Rm` vs `Rp`). The redesign lets the neutral domain extend behind the
cathode (the plenum) while the plasma domain is bounded, inside it, by reflecting
faces at the cathode and collector.

## 2. Physical picture

Single-cathode layout (asymmetric — see §3 for twin/mirror):

```
        pump (radial ports)
         ⇡  elbow: NOT modeled — folded into effective pump speed
    ┌──────────┐  ║  ┌─────────┬─────────┬──────┬──── ... ────┬───────────┐
    │  plenum  │  ║  │ cathode │  anode  │ puff │   column    │ collector │
    │ neutral- │  ║  │  cell   │  cell   │ cell │   cells     │   cell    │
    │  only    │  ║  │         │         │      │             │           │
    └──────────┘  ║  └─────────┴─────────┴──────┴──── ... ────┴───────────┘
    z=0        cathode         ↑ anode mesh                             z=Lm
   (closed   face:            (neutral throttle,                     (plasma
    back      ·plasma wall     plasma-open)                           end-loss;
    wall)     ·ion neutralize                                         closed
    pump→     ·e⁻ sheath-filtered                                     neutral
    here      ·annular obstruction                                    end)
              (Rcs..Rm, over Lcs)

  neutral flow:  plenum ◄─obstruction─ cathode ◄─anode mesh─ PUFF ─► column ─► (closed end)
                    │                                          │
                  pump                                     (source)
```

The plenum is topologically identical to today's end cell — a cell with a closed
outer face. What makes it a *plenum* is three attachments: a plasma-reflecting
face on its column side, a volumetric pump sink, and the annular-obstructed face
to the cathode cell.

## 3. Geometry schema

### Typed cells

`cell_role` exists today but is informational only ([`physics/cathode.py`](physics/cathode.py)
reads it as a string). This makes it **load-bearing**. Roles and what each anchors:

| role | anchors |
|---|---|
| `plenum` | pump sink; reduced area for support structure; **no plasma** |
| `cathode` | ion neutralization (end-loss), sheath electron-energy loss, thermionic emission, ohmic deposition |
| `anode` | electron collection (`P_anode_e`), beam energization |
| `puff` | gas-puff source (in front of the anode) |
| `column` | bulk plasma + neutrals |
| `collector` | plasma end-loss; closed neutral end |

### Typed faces — three *independent* properties

The anode mesh is the proof they must be independent: it is **plasma-open** (beam
and flow pass through), **neutral-throttled**, and **heat-throttled** — three
different fractions on the same face.

- `plasma_open` (bool): `False` at the true machine ends *and* at the cathode
  face → the plasma domain is bounded inside the neutral domain. Generalizes
  today's end-only reflecting faces ([`physics/flux.py:66`](physics/flux.py)).
- `neutral_conductance`: either **default Clausing** from the local area +
  hydraulic radius, or a **prescribed aperture** (obstruction, mesh).
- `heat_transmission` ∈ [0,1]: the fraction of parallel (electron *and* ion)
  conductive flux `κ∥∇T` that crosses the face, over the plasma cross-section.
  `1` on normal interior faces; `0` at every plasma wall (cathode, collector,
  true ends — heat leaves there via sheath/collection sinks, not conduction
  across); `(1−eta)` at the anode (open holes conduct, solid mesh blocks).

### Neutral geometry generalization (the keystone)

Today the neutral geometry derives everything from a single `Rm_cm`
(`neutral_area = πRm²`, and the Clausing `R` is that same `Rm`). Obstructions
break the one-radius assumption. **Carry, per neutral cell and face, a
cross-sectional `area` and a `hydraulic_radius` that can differ.** Defaults
reproduce current behavior exactly (`area = πRm²`, `hydraulic_radius = Rm`).

Three real features all reduce to different inputs to this one schema:

| feature | area | hydraulic radius | notes |
|---|---|---|---|
| full-bore face/cell | `πRm²` | `Rm` | recovers current model |
| cathode↔plenum obstruction | `π(Rm²−Rcs²)` | `Rm−Rcs` | annular duct over `Lcs` |
| anode mesh | `(1−eta)·πRm²` | (aperture) | plasma-open |
| plenum with support rods | `π(Rm²−Rsup²)` | `Rm` | volume only; `Rsup` default 0 |

### Concrete inputs (defaults discussed)

- `Rm = 50 cm` (machine), `Rp = 18 cm` (plasma column).
- Plenum: full `Rm = 50` extension of the machine tube, closed back wall.
- Cathode structure obstruction: inner `Rcs = 25`, outer `Rm = 50`, length `Lcs`.
  Open area `π(50²−25²) ≈ 5890 cm²` (**75% open** — blocking the central 25 cm
  removes only a quarter of the area). Hydraulic diameter `2(Rm−Rcs) = 50 cm`.
- Support rods: effective blockage radius `Rsup` (summed rod cross-sections as one
  disk), reduces plenum volume only; **distributed thin structure, not a duct**,
  so no hydraulic-radius/transmission effect. Config knob, default `Rsup = 0`.
- Anode mesh: `eta` is the anode **opacity** (solid fraction), so `1−eta` is the
  transparency. The transparency *fraction* `(1−eta)` sets what passes the mesh,
  but the relevant *cross-section differs by what crosses*: the **neutral
  aperture** spans the machine cross-section `πRm²` (gas fills the tube), while
  the **beam** spans only the cathode cross-section — `πRp²`, since cathode =
  plasma cross-section for now. **Ion/electron collection** scales with opacity,
  doubled for both faces (`2·eta`), over the plasma cross-section `πRp²` (see §7).

### Config change

Replace the symmetric `source_length_cm + Lz + end_length_cm` tiling
([`core/geometry.py:59`](core/geometry.py)) with a **per-end sequence of typed
segments**, asymmetric by end:

- **Single cathode:** `[plenum, cathode, anode, …column…, collector]`.
- **Twin/mirror cathode:** symmetric — `[plenum, cathode, anode, …, anode, cathode, plenum]`.

This is the single biggest structural decision; the role arrays and face arrays
are built from it.

## 4. Neutral conductances

Generalized Clausing, same structure as [`physics/neutrals.py:18`](physics/neutrals.py):

```
C = 0.25 · v_th · A · P,      P = 1 / (1 + 3L/(4·D_h))
```

- **Full-bore face:** `A = πRm²`, `D_h = 2Rm` → recovers current coefficients.
- **Cathode↔plenum obstruction:** `A ≈ 5890 cm²`, `D_h = 50 cm`, over `Lcs`.
  `P ≈ 1/(1 + 3·Lcs/200)` (≈0.73 at `Lcs=25`; combined with 75% area ≈0.55 of an
  unobstructed face). **Face-vs-cell decision** (open, see §11): a lumped face
  conductance if `Lcs` is small vs the ~30 cm cell size; a real `Lcs`-long
  annular cell if `Lcs` is appreciable and holds enough gas to matter to the
  pump throughput.
- **Anode mesh:** the neutral aperture is transparency × machine cross-section
  `(1−eta)·πRm²`; the face is plasma-open. The **beam** crosses over the
  cathode/plasma cross-section `πRp²` (not `Rm`), transmitted by `(1−eta)` with
  the intercepted `eta` fraction captured at the anode (the existing
  `1 − eta·beam_bypass_fraction` factor). The anode also *collects and
  neutralizes* ion saturation current on both faces (`2·eta`) over `πRp²` — a
  bilateral plasma sink + neutral source, see §7.
- **Pump:** volumetric sink on the plenum cell — `pump_rate(lps, vol)` already
  takes a speed on a chosen volume. The unmodeled elbow folds into an **effective
  pump speed** `1/S_eff = 1/S_pump + 1/C_elbow` (series conductance). Radial ports
  justify the volumetric (non-directional) treatment. The plenum back wall is a
  closed neutral end — inherits existing external-face behavior (Clausing lives
  only on interior faces, [`physics/neutrals.py:37`](physics/neutrals.py)).

## 5. Plasma domain boundary

- The plasma domain is bounded by **reflecting faces at the cathode (interior) and
  collector**. Generalize the current end-only reflecting logic to a
  `plasma_open` face array; reflecting faces get `face_M = p`, zero advective
  particle/thermal flux (as [`physics/flux.py:66`](physics/flux.py) does today for
  the ends).
- Plenum cells are **plasma-dead by construction**: reflecting face on the column
  side + no beam + bulk ionization negligible at floor `n`. Plasma there sits at
  the floor.
- **Do not** try to suppress plenum plasma by setting `Rp → 0`. A near-zero plasma
  volume blows up the flux divergence (`_flux_divergence` divides by
  `plasma_volume_cm3`). The explicit reflecting face is the safe mechanism.
- Confirm the timestep / floor-clip / step-rejection machinery tolerates inert
  plenum plasma cells (they should never bind, but verify with
  `audit_sim1d_floor_activation.py`).

### Anode: a partial interior obstruction, not a wall

The cathode and collector faces are full walls (§5); the anode is a *partial*
obstruction *inside* the plasma flow. Two distinct fluxes reach the mesh:

- **Thermal (Bohm) saturation flux**, both faces — the `2·eta·I_i_a` collection /
  neutralization already defined (§7): circuit current + bilateral neutral source.
- **Directed bulk flow** `n·u` crossing the plane — the solid fraction `eta`
  intercepts it.

**These are the same particles when the flow is near-sonic.** Approaching a loss
region `u → c_s`, so the directed interception `eta·n·u` ≈ the Bohm collection
`eta·n·c_s·exp(-0.5)`. A transparency-reduced advective transmission and the
collection sink are therefore *two descriptions of one effect* — applying **both**
double-counts the removed plasma.

**Recommended model:** represent the anode's entire plasma effect as the
**bilateral collection/neutralization sink**, removing mass, momentum, *and*
thermal energy together (exactly as `surface_neutralization` does at a wall:
`M = −m_i·u·loss_rate`, `Ee,Ei = −1.5 T·loss_rate`), at rate `2·eta·I_i_a`, at the
anode cell(s). **Keep the plasma advective face open** — do *not* additionally
shrink it by `(1−eta)`. When sonic, the sink alone reproduces the correct net
transmission `≈ (1−eta)·n·c_s`.

- **Momentum is a sink, not heating.** The intercepted plasma's directed momentum
  is absorbed by the solid, grounded anode structure — lost from the plasma, not
  thermalized into it. Opposite of ion–neutral drag (where the neutral recoils and
  stays, so half the drift energy heats the ions). The `M`/energy removal in the
  sink handles this consistently.
- **⚠ The equivalence is near-sonic only.** If the flow is strongly *subsonic* at
  the anode, the Bohm sink (∝ `c_s`) over-removes relative to the directed
  interception (∝ `u`). Revisit if that regime matters — see §11.

**Parallel heat conduction across the mesh** is a *separate* transport channel and
is throttled independently. The solid fraction `eta` of the field lines terminates
on the mesh, so the anode face transmits only `(1−eta)` of the parallel electron
and ion conductive flux `κ∥∇T` (over the plasma cross-section). This does *not*
double-count the collection sink — they act on complementary areas: heat to the
**solid** fraction is carried off by the collected particles (sink + sheath
`P_anode_e`, already counted), heat across the **open** fraction conducts to the
other side. Implement via the per-face `heat_transmission` factor (§3): `(1−eta)`
at the anode, and `0` at the new cathode reflecting face — a wall must not conduct
plasma heat into the plenum, yet the conduction operator runs on every interior
face ([conduction.py:130](physics/conduction.py)), so the wall faces must be zeroed
explicitly.

## 6. Cathode surface asymmetry

The cathode "rejects electrons below a certain energy but neutralizes ions like
end loss." Most of this asymmetry is **already in the model**, just colocated and
needing relocation:

- **Ion / particle loss** = ambipolar Bohm flux at the sound speed → the existing
  `surface_neutralization` term ([`physics/sources.py:78`](physics/sources.py)),
  which also produces the neutralized neutrals. Relocate from `[0]` to the cathode
  cell.
- **Electron energy loss** = sheath-filtered → already `P_cathode_e` from the
  cathode solve ([`physics/cathode.py:251`](physics/cathode.py)); the sheath drop
  `phi_c` *is* "rejects electrons below a certain energy." Relocate to the cathode
  cell.

**Modeling choice (open, §11):** keep ion loss as a *volumetric* term in the
cathode cell (recommended — already encodes the asymmetry correctly, reflecting
face stays a wall for the bulk flow, lower risk) versus promoting it to a genuine
Bohm-flux face (cleaner, but then it replaces the surface term and must be
reconciled against the collector-end loss model).

## 7. Cathode solver rewiring — distinct anode/cathode sampling

**Current behavior** ([`funcs/_cathode_solver.py`](../../funcs/_cathode_solver.py)):
`solve()` takes a *single* `PlasmaState` and computes *both* sheaths from it.
`solve_beam_system` samples the cathode cell only:

```
I_i   = A_c · e · n_e · C_s(T_e) · exp(-0.5)     # cathode cell (n_e, T_e)
I_i_a = 2 · eta · I_i                            # anode current tied to cathode plasma
J_i   = I_i   · R_p / T_e                        # single R_p, single T_e
J_i_a = I_i_a · R_p / T_e
```

The anode ion current is just the cathode current scaled by `2·eta` (`eta` =
anode **opacity**, the `2` because both mesh faces collect) — so both sheaths run
off the cathode-cell plasma.

**Target:** sample the cathode cell *and* the anode cell, so **each sheath sees a
different `I_i`**:

```
I_i   =        A_c   · e · n_cath  · C_s(Te_cath)  · exp(-0.5)   # cathode, cathode area A_c
I_i_a = 2·eta · πRp² · e · n_anode · C_s(Te_anode) · exp(-0.5)   # anode, both faces, plasma cross-section
```

The `2·eta` stays (opacity × both faces) but the base ion flux now comes from the
**anode-cell** plasma, not the cathode's. **Three cross-sections apply to the same
anode**, and only the neutral one uses `Rm`:

| interaction | fraction | cross-section |
|---|---|---|
| neutral aperture (Clausing) | `(1−eta)` | machine `πRm²` — gas fills the tube |
| primary beam transmission | `(1−eta)` | cathode = plasma `πRp²` for now |
| ion/electron collection | `2·eta` | plasma `πRp²` — collects only where plasma is |

The anode is physically larger than the plasma, but it *collects* only over the
plasma cross-section; the *beam* also spans only the cathode cross-section. So
`Rm` is a neutrals-only quantity here. (Cathode and plasma cross-sections coincide
today; flag if `R_cath ≠ Rp` later.)

- `P_cathode_e` lands at the cathode cell, `P_anode_e` at the anode cell —
  retiring the TODO at [`physics/cathode.py:258`](physics/cathode.py).
- **The anode is a bilateral plasma sink + neutral source.** The collected
  `2·eta·I_i_a` is neutralized on *both* mesh faces, so the neutral gas is **split
  across the two cells flanking the anode** (cathode-gap side and column/puff
  side) — not dumped on one side. This matters because the mesh *throttles*
  cross-anode neutral flow: gas must be born on the correct side, since a neutral
  born on the column side cannot easily diffuse back through the mesh, and vice
  versa. Today `_cathode_particle_loss_rate` returns `(1 + 2·eta)·I_i/qe` and
  lumps the cathode (`1`) and both-faces anode (`2·eta`) neutralization together
  at cell `[0]` ([`physics/cathode.py`](physics/cathode.py)); resolved, they
  separate — cathode → cathode cell (one face), anode → both anode-flanking cells.
- **⚠ Asymmetric anode sheath — flag to investigate (do not resolve here).**
  Because the gap-facing and column-facing sides of the anode see *different*
  plasma (density, temperature, flow), each face collects a different
  ion/electron flux — even though it is one physical object. The sheath potential
  drop from the plasma potential may then differ between the two faces, whereas
  the current solver computes a single `phi_a`. A resolved anode may need a
  per-face sheath solve. Parked in §11.
- **Graded implementation:**
  - **(a) now** — feed the anode-cell `(n, Te)` into the anode-sheath block and
    compute a distinct `I_i_a`; captures the stated concern.
  - **(b) later** — replace the single `R_p`-from-`L_cath` plasma resistance with a
    path integral of resistivity over the resolved cathode–anode gap cells (`T_e`
    also varies along it).
- `solve()` / `solve_beam_system` signature: pass the `(Te, ne)` profile plus
  explicit `cathode_index` and `anode_index` (or two `PlasmaState`s).

## 8. Term relocations (anchor via roles, not `[0]`/`[-1]`)

| term | today | resolved anchor |
|---|---|---|
| gas puff | cell `[0]` | `puff` cell (in front of anode) |
| pump | cell `[0]` / `[-1]` | `plenum` cell |
| surface neutralization | `[0]`, `[-1]` | `cathode` cell, `collector` cell |
| cathode ion neutralization (`1·I_i`) + e⁻ loss | `[0]` | `cathode` cell (one face) |
| anode electron collection | (in `[0]`) | `anode` cell |
| anode ion neutralization (`2·eta·I_i_a`) | lumped in `[0]` | **both** `anode`-flanking cells (bilateral) |
| beam Beer-Lambert deposition | from `[0]` | already per-cell; works once cells exist |
| ohmic deposition | `[0]` | `cathode` cell |

## 9. Already handled — no work needed

- The finite-volume core is **non-uniform-grid aware** (`center_distance_cm`,
  per-cell `length_cm` in flux/conduction/timestep). No discretization rewrite.
  The only uniform-grid assumption is the `dz_cm` convenience accessor
  ([`core/geometry.py:31`](core/geometry.py)), used for reporting only.
- Beam Beer-Lambert absorption is already per-cell
  ([`physics/cathode.py:290`](physics/cathode.py)).
- `pump_rate` / `puff_rate` already act volumetrically on a chosen cell.
- External faces are already closed for both plasma and neutrals.

## 10. Validation and thesis impact

- Rerun `scripts/smoke_sim1d.py` (required after any `_sim1d/` change) and
  `scripts/verify_sim1d_order.py` (time integration is untouched but confirm).
- **Steady state will move.** Rerouting neutral fueling through the puff→column /
  puff→pump back-path and splitting anode/cathode conditions changes the column
  neutral profile and ionization balance. Per
  [`manuscript/THESIS_NOTES.md`](../../../../manuscript/THESIS_NOTES.md) this is a
  "changes what the model claims" change, coupled to the already-provisional
  `b_Q*` and `b_ion_neutral_drag` factors. **Review-not-publication until
  re-validated;** build deliberately, not mid-calibration. Update THESIS_NOTES
  when this lands.
- New sensitivity knobs to sweep: `Lcs`, `Rcs`, `Rsup`, anode transparency,
  effective pump speed.

## 11. Open decisions

1. **`Lcs` in cell-size terms** → obstruction as a face (thin) or a cell
   (appreciable). Decides the neutral-geometry schema's granularity.
2. **Cathode solver (a) vs (b)** — distinct `I_i`/`Te` per side now; gap
   resistivity path-integral later.
3. **Cathode ion loss** — volumetric term (recommended) vs Bohm-flux face.
4. **End default** — plain collector (single cathode) with the mirror/twin
   symmetric plenum as an opt-in.
5. **Anode as face or cell** (analogous to `Lcs`, decision 1). The sheath sampling
   and `P_anode_e` want a cell that reads local `(n, Te)`; the mesh throttle and
   the bilateral neutralization split want a *face* between a gap-side and a
   column-side cell. Decide whether the anode is one cell (with the neutral source
   split to its two neighbors) or a face pair — this sets how the `2·eta·I_i_a`
   neutral gas is apportioned across the throttling mesh.
6. **⚠ Asymmetric anode sheath — investigate.** With each anode face seeing a
   different ion/electron flux (§7), the sheath drop from the plasma potential may
   be asymmetric between the gap-facing and column-facing sides. The solver
   currently produces a single `phi_a`. Determine whether a per-face anode sheath
   is needed and, if so, how it feeds `P_anode_e`, the anode collection, and the
   circuit current balance. Coupled to decision 5.
7. **Anode plasma obstruction in the subsonic regime** (§5). The recommended
   collection-sink-only model equals a transparency-reduced advective face *only*
   near sonic flow. If the anode sits where the flow is strongly subsonic, decide
   whether to add an explicit `(1−eta)` advective-transmission reduction with the
   collection sink adjusted to the directed flux to avoid double-counting.

## 12. Suggested sequencing

Reversibility (§13) constrains every step: build the resolved path behind a
default-off master switch and prove it collapses to today's behavior.

0. **Golden baseline** — capture today's `smoke_sim1d.py` trajectory and a
   representative `run_sim1d.py` output and commit them *before* any refactor.
   Cannot be reconstructed later; it is what makes "turn it off" verifiable (§13).
1. **Geometry schema** — typed segments, `area` + `hydraulic_radius` per neutral
   cell/face, `cell_role` and face-property arrays, behind the `resolved_boundaries`
   master switch (default off → legacy lump). The keystone; everything hangs off it.
2. **Neutral conductances** from the new schema (generalized Clausing; prescribed
   apertures for obstruction and mesh; effective pump speed).
3. **Plasma reflecting-face generalization** + confirm inert plenum plasma.
4. **Term relocations** via roles (§8).
5. **Cathode solver split sampling** (§7, level (a)).
6. **Validation** — smoke + order tests, the §13 legacy-equivalence assertions,
   sensitivity sweep, re-baseline, update THESIS_NOTES.

## 13. Reversibility and kill switches

The `b_*` gates and the THESIS_NOTES caveats already mark this as a
review-not-publication change that may not pan out. It must therefore be **cleanly
and completely reversible to today's lumped behavior**, designed in from step 0.
This redesign is a strict *superset* of the current model (today = no plenum,
`eta=0`, no obstruction, single sample point), which is what makes exact
reversibility achievable.

### Master switch — default off
- A single flag `resolved_boundaries` (default `False`) selects geometry mode. Off
  → the current symmetric source/domain/end lump ([`core/geometry.py`](core/geometry.py));
  on → the typed-segment machine (§3). Everything downstream keys off geometry, so
  this one switch reverts the whole design. The default stays off until
  legacy-equivalence is proven and trusted.

### Legacy config is a valid config — one code path, not two
- Prefer **rewriting the operators to be role-/face-driven** over branching on the
  flag internally. Legacy geometry has roles `source/domain/end`, all faces fully
  open or today's walls (no throttles: `heat_transmission=1`, default Clausing, no
  apertures), terms anchored at `[0]`/`[-1]`. Fed that, the rewritten operators must
  reproduce today's behavior — so "off" is the same code on legacy inputs, not a
  dead parallel path that rots.
- The cathode solver's two-point sampling (§7) collapses to `I_i_a = 2·eta·I_i`
  when the anode sample cell *is* the cathode cell — no separate legacy branch.

### Per-feature knobs for bisection — prefer continuous over boolean
Follow the `b_*` scale-factor convention so each effect dials to its legacy limit
*continuously*, not just on/off — far easier to bisect a bad result:

| feature | legacy limit |
|---|---|
| annular obstruction (`Lcs`, `Rcs`) | `Rcs=0` / `Lcs=0` → full aperture |
| anode mesh (`eta`) | `eta=0` → transparent, no collection, `heat_transmission=1` |
| support rods (`Rsup`) | `0` |
| pump relocation | pump back on the source cell |
| plenum | zero-length / absent |
| split anode/cathode sampling | sample the same cell |

A fully-legacy-limit *resolved* config (single cell per end, `eta=0`, no
obstruction) is thus a second, independent route to "off."

### The guarantee — a golden regression test
None of the above is trustworthy without a test that *proves* equivalence:
- Capture today's `smoke_sim1d.py` trajectory and a representative `run_sim1d.py`
  output as a **golden baseline**, committed before the refactor (§12 step 0).
- Assert that both (a) master-switch-off and (b) the degenerate legacy-limit
  resolved config reproduce the baseline to tight tolerance.
- This makes "turn it all off" a guarantee rather than a hope, and guards against
  silent drift as the new code evolves.

### VCS-level
- Already on a branch (`sim1d-patch1`); keep §12 steps as **granular commits** so
  any single step reverts independently and the whole series can be dropped without
  touching `_sim3` or unrelated code.
