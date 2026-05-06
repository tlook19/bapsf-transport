# Cathode Cell Refactor — Changes to `cablp` and Required Updates to `bapsf_app`

## Summary of change

Cathode cells are now **fixed-length 100 cm cells** that are separate from and additional to
the user-configured `cells` parameter.  Previously `cells` was the total cell count and all
cells had uniform length `Lp / cells`.  After the refactor:

- `cells` in the config/params dict = **number of interior cells** (always rounded up to the
  nearest odd integer).
- Total cells in the simulation = `cells + 2` — **always**, regardless of `TwinCathode`.
  Both ends have a fixed 100 cm boundary cell so the pump geometry is always symmetric.
- Cell layout:
  - Index 0: primary cathode cell, length 100 cm
  - Indices 1 … cells: interior cells, each length `(Lp − 200) / cells`
  - Index −1: right boundary cell, length 100 cm (twin cathode when `TwinCathode=True`,
    otherwise an inert end cell with only the right vacuum pump)

### Concrete example

| Config | TwinCathode | Total cells | `L_plasma` array (cm) |
|--------|-------------|-------------|------------------------|
| Lp=1800, cells=3 | False | 5 | [100, 533.3, 533.3, 533.3, 100] |
| Lp=1800, cells=3 | True  | 5 | [100, 533.3, 533.3, 533.3, 100] |

Geometry is identical in both modes. The only difference is that the right end cell receives
gas puffing (`S_gp[-1]`) and the cathode solver only when `TwinCathode=True`.

## New attributes on `LAPDSim`

| Attribute | Type | Description |
|-----------|------|-------------|
| `_n_interior` | int | Number of interior cells (user `cells` value, always odd) |
| `_n_cathode_cells` | int | Always 2 |
| `_cathode_cell_len` | float | 100.0 cm (fixed) |
| `_cells` | int | Total cells = `_n_interior + 2` |
| `_L_plasma` | ndarray (cells,) | **Non-uniform** cell lengths |
| `_L_cell` | ndarray (cells,) | **Non-uniform** machine cell lengths |

`_max_cells` (padding bound for HDF5 output) and `_min_cells` (adaptive-mesh floor) are also
adjusted: both refer to interior cells internally; `_max_cells` stored as
`max_interior + 2`.

## Invariants that still hold

- Gas puff (`S_gp`) is in index 0; twin gas puff is in index −1 **only when `TwinCathode=True`**.
- Both pumps are always at index 0 (left) and index −1 (right) — symmetric in all modes.
- The cathode solver reads/writes index −1 only when `TwinCathode=True`.
- Adaptive mesh resizing only moves interior cells; the two end cells stay fixed at 100 cm.
- `cells_at_time` in HDF5 results stores **total** cell count at each timestep.

---

## Required changes in `bapsf_app`

### 1. `stats.py` — `cell_centers()`

**Current signature:**
```python
def cell_centers(n_cells, L_plasma, convention="sim"):
    sim_z = np.array([(i + 0.5) * L_plasma / n_cells for i in range(n_cells)])
```
This assumes uniform cell lengths and breaks with the new non-uniform layout.

**Fix:** Accept an optional array of cell lengths.  When the array is provided, compute
cumulative-sum positions instead of the uniform formula.

```python
def cell_centers(n_cells, L_plasma, convention="sim", cell_lengths=None):
    """
    cell_lengths : array-like, shape (n_cells,), optional
        Actual per-cell lengths [cm].  When provided, L_plasma is ignored.
        When omitted, falls back to the old uniform formula (backwards-compat).
    """
    if cell_lengths is not None:
        cl = np.asarray(cell_lengths)
        sim_z = np.cumsum(cl) - cl / 2  # cell centers from cumulative lengths
    else:
        sim_z = np.array([(i + 0.5) * L_plasma / n_cells for i in range(n_cells)])
    return sim_z if convention == "sim" else L_plasma - sim_z
```

The cumulative-sum formula gives the correct center position even when the first cell
is only 100 cm and the rest are ~566 cm.

### 2. `plot.py` — pass cell lengths to `cell_centers()`

**Current code (around line 103):**
```python
L_plasma = params.get("Lp", params.get("L_plasma", 1800))
z_pos = cell_centers(n_cells, L_plasma, convention=z_convention)
```

If the loaded results include per-cell length information, pass it through.  The safest
approach is to store `L_plasma` as an array in the HDF5 run group (see §4 below) and load it:

```python
cell_lengths = results.get("L_plasma_cells", None)  # array or None
z_pos = cell_centers(n_cells, L_plasma, convention=z_convention, cell_lengths=cell_lengths)
```

If `L_plasma_cells` is not yet stored, you can reconstruct it from params:

```python
def _cell_lengths_from_params(params, n_cells):
    Lp = params.get("Lp", 1800)
    cath_len = 100.0
    n_int = n_cells - 2            # both ends are always 100 cm boundary cells
    int_len = (Lp - 2 * cath_len) / n_int
    lens = np.full(n_cells, int_len)
    lens[0] = cath_len
    lens[-1] = cath_len
    return lens
```

### 3. `plot.py` — uniform cell volume assumption

**Current code (around line 226):**
```python
L_plasma = params.get("Lp", 1800)
cell_vol = math.pi * Rp**2 * (L_plasma / n_cells)  # cm³ per cell — WRONG
```

This is used for normalising energy-balance terms.  With non-uniform cells each cell has its
own volume:

```python
cell_lengths = _cell_lengths_from_params(params, n_cells)  # helper from §2
cell_vols = np.pi * Rp**2 * cell_lengths  # shape (n_cells,)
```

Any place that multiplies a per-cell quantity by `cell_vol` (a scalar) needs to multiply by
`cell_vols[i]` instead, or sum `quantity * cell_vols` across cells.

### 4. `sweep.py` — `_apply_equilibrated_nn()`

**Current code:**
```python
n_cells = int(params.get("cells", 3))       # was total; now is interior count
active_nn = nn_eq[:n_cells]                 # slices to wrong length
if n_cells > 2:
    nn0_eq = float(active_nn[1:-1].mean())  # meant to average interior cells
```

After the refactor `params["cells"]` is the number of **interior** cells.  Total cells =
`n_cells + n_cathode_cells`.  The `nn_eq` result array has `total_cells` valid entries
followed by NaN padding.

**Fix:**
```python
n_interior = int(params.get("cells", 3))
n_total = n_interior + 2          # both ends always have a 100 cm boundary cell
active_nn = nn_eq[:n_total]       # all valid cells (interior + 2 end cells)
interior_nn = active_nn[1:-1]     # strip both end cells to get interior slice
nn0_eq = float(interior_nn.mean())
```

### 5. `app.py` — UI label for `cells`

**Current (around line 286–295):**
```python
"cells": {
    "label": "Number of cells", "unit": "", "default": 3,
},
"max_cells": {
    "label": "Max cells (adaptive mesh)", "unit": "", "default": 18,
},
"min_cells": {
    "label": "Min cells (adaptive mesh)", "unit": "", "default": 3,
},
```

Update labels to make it clear these refer to **interior** cells:

```python
"cells": {
    "label": "Interior cells (cathode cells added automatically)",
    "unit": "", "default": 3,
},
"max_cells": {
    "label": "Max interior cells (adaptive mesh)", "unit": "", "default": 18,
},
"min_cells": {
    "label": "Min interior cells (adaptive mesh)", "unit": "", "default": 3,
},
```

Also note: the simulator silently rounds even values up to the next odd integer.  If the UI
validates or displays the effective value, it should apply the same rounding:
`n = v if v % 2 == 1 else v + 1`.

### 6. `plot.py` — plot title cell count

**Current (around line 63, 71):**
```python
cells = params.get("cells", "?")
...
return f"... {cells} cells ..."
```

`cells` now means interior cells.  To avoid confusion show total cells alongside, or
clarify the label:

```python
n_int = params.get("cells", "?")
n_total = n_int + 2 if isinstance(n_int, int) else "?"
label = f"{n_int}+2 cells"   # e.g. "3+2 cells" (always 2 boundary cells)
```

---

## What does NOT change

- The HDF5 schema: result arrays are still padded to `max_cells` with NaN; `cells_at_time`
  still records the **total** active cell count at each step.
- The cathode solver interface, cathode result fields, or any cathode-physics logic.
- Sweep parameter keys (`cells`, `max_cells`, `min_cells`) — same names, narrowed meaning.
- All other params/flags keys.
