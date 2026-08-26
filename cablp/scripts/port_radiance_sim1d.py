#!/usr/bin/env python3
"""Synthetic radiated-power (line-radiation) diagnostic at one LAPD port.

READ-ONLY over a saved sim1d HDF5 artifact.  Nothing is fitted, nothing is
written back to the artifact, and the solver is never constructed: every
number here is a re-evaluation of the run's OWN rate closure on the run's own
saved state, so the instrument cannot report a different physics than the run
carried.

WHAT IS COMPUTED.  Two line-radiation channels, both taken straight from the
OPEN-ADAS adf11 PLT coefficients that ``atomic_rate_model = "adas"`` runs on
(``cablp/funcs/_adas.he_rates``), evaluated at the saved cell state:

    e-i (He II) line power   eps_ei = PLT2(ne, Te) * ne * ne
    e-n (He I)  line power   eps_en = PLT1(ne, Te) * ne * nn

with PLT in eV cm^3 s^-1 and the result converted to W cm^-3.  This mirrors
``physics/energy.py::electron_cooling_rhs_terms`` term for term -- the same
fused ``he_rates`` lookup, the same ``max(n, ne_floor)`` guard on the lookup
density, the same RAW ``n`` in the density product, the same saved ``Te`` --
so the e-i channel reproduces the artifact's ``electron_ion_cooling`` ledger
row and the e-n channel its ``electron_neutral_cooling`` row.  That identity
is checked and printed rather than assumed (see SELF-CONSISTENCY below).

Two closure caveats, printed in every header because they are the only ways
the two can legitimately differ:

  * the solver scales its cooling rows by ``b_Qei`` / ``b_Qen`` (and by the
    optional ``b_Q*_Te_exp`` shape); this instrument reports the UNSCALED
    radiated power, because a cooling fudge factor is not an emissivity.  At
    the standing ``b = 1`` rate-channel policy the two coincide exactly.
  * with ``icool_recomb`` on, the solver folds the PRB recombination +
    bremsstrahlung coefficient into its ``electron_ion_cooling`` row; the
    PLT2 channel here is line power only and excludes it.

CHORD QUANTITIES AND THEIR ASSUMPTION.  The model is 1D: it carries one
(ne, nn, Te) per axial cell and no radial profile at all.  A chord integral
therefore requires an explicit radial assumption, and the honest one for a
1D model is the flattest: **emissivity is taken radially UNIFORM across the
plasma disc of radius Rp(z), and zero outside it.**  This is a construction,
not a measurement, and it is stated in the output header of every product
this script writes.  Under it, for a radial line of sight through the column
axis at that z,

    chord length      L_chord = 2 * Rp(z)                      [cm]
    surface brightness    B   = eps * L_chord                  [W cm^-2]
    per-length power   dP/dz  = eps * A_p(z)                   [W cm^-1]

with A_p(z) = plasma_volume_cm3 / length_cm, the cross-section the solver
actually books the plasma row on (identically pi*Rp^2 on the shipped
geometry).  B is what a radially-viewing, absolutely-calibrated,
wide-open-band radiometer would read on the axial chord; dP/dz is the
column's radiated power per unit length and integrates over z to the total
radiated power without any radial assumption entering.

WHAT THIS IS NOT (thesis item 56).  The e-i channel is >= 99.9 % EUV -- the
30.4 nm class 1s-2p resonance line dominates PLT2 -- so this instrument
reports RADIATED POWER, not the signal a glass-windowed visible spectrometer
or a filtered photodiode would see.  The band split of record is
``scripts/pec_band_fractions.md``: the glass-transmissible (> 350 nm) share of
the adf11 line power is 1.4e-05 to 8.5e-04 for He II and 0.052 to 0.120 for
He I over its report grid at Te >= 2 eV.  Never quote a number from here as a
visible-light prediction without folding that band fraction in.

PORT -> z MAPPING.  Mirrors ``compare_sim1d_es1.py``: the probe ports and
their axial positions are read from the same committed overlay
(``scripts/data/es1_sim1d_overlay.npz``, fields ``port`` / ``z_cm``, consumed
at ``compare_sim1d_es1.py:608-609``) and the cell is selected by nearest cell
centre exactly as at ``compare_sim1d_es1.py:669``.  The five overlay ports lie
on a single exact linear law (pitch 31.95 cm, the LAPD port spacing), which is
verified against every anchor at load and then used to place ports the overlay
does not carry -- port 27 among them, at z = 981.25 cm, matching the in-repo
value at ``scripts/fab_choke.py:47``.

WINDOWS.  The two registered windows of ``power_ledger_sim1d.py`` (DRIVE
plateau 15.25-19.75 ms, AFTERGLOW 20.5-24.5 ms, both RUN-CLOCK) are reused
unchanged so a number here can be read against that ledger's rows.

SELF-CONSISTENCY.  When the artifact carries the
``electron_energy_terms_W_cm3`` group, both channels are compared against
their ledger rows -- drive-window means and the worst per-frame relative
deviation -- and the comparison is written into the markdown and json
products.  An artifact without the group is reported as such rather than
silently skipped.

    port_radiance_sim1d.py [--h5 RUN.h5] [--port 27] [--output-stem STEM]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cablp.funcs._adas import he_rates  # noqa: E402
from cablp.vars._cons import qe_SI  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent

#: Default artifact: the mgcr1 confirmation run this instrument was built on.
DEFAULT_H5 = SCRIPT_DIR / "mgcr1_confirm.h5"

#: Port -> z anchors, read from the same overlay compare_sim1d_es1.py reads.
PORT_ANCHOR_OVERLAY = SCRIPT_DIR / "data" / "es1_sim1d_overlay.npz"

#: In-repo band-split note cited by the item-56 context line.
BAND_SPLIT_NOTE = "scripts/pec_band_fractions.md"

#: Registered windows, run-clock ms; identical to power_ledger_sim1d.py.
DRIVE_WINDOW_MS = (15.25, 19.75)
AFTERGLOW_WINDOW_MS = (20.5, 24.5)

#: Ledger row each channel must reproduce, keyed by channel.
LEDGER_ROW = {"plt2": "electron_ion_cooling", "plt1": "electron_neutral_cooling"}

CHANNEL_LABEL = {
    "plt2": "e-i line (He II, PLT2 . ne^2)",
    "plt1": "e-n line (He I, PLT1 . ne.nn)",
    "total": "total line (PLT1 + PLT2)",
}

QUANTITY_UNIT = {
    "emissivity": "W cm^-3",
    "brightness": "W cm^-2",
    "dPdz": "W cm^-1",
}


class ArtifactRefused(SystemExit):
    """Raised with an explanatory message when the artifact cannot be read."""


# --- port -> z ------------------------------------------------------------


def port_axial_law(overlay_path=PORT_ANCHOR_OVERLAY):
    """Return ``(z0_cm, pitch_cm, anchors)`` for the linear port -> z law.

    ``anchors`` is the committed (port, z_cm) table the overlay carries, i.e.
    exactly the mapping ``compare_sim1d_es1.py`` scores against.  The law is
    ``z(port) = z0 + pitch * port``, fitted through the extreme anchors and
    then verified against every anchor; a table that is not exactly linear
    raises rather than being silently least-squares-averaged, because the
    whole point of extrapolating to an unscored port is that the pitch is a
    hardware fact.
    """
    if not Path(overlay_path).exists():
        raise ArtifactRefused(
            f"port anchor overlay not found: {overlay_path}\n"
            "This instrument mirrors compare_sim1d_es1.py's port -> z mapping "
            "and will not invent one."
        )
    with np.load(overlay_path, allow_pickle=True) as ov:
        ports = np.asarray(ov["port"], dtype=int)
        z_cm = np.asarray(ov["z_cm"], dtype=float)
    order = np.argsort(ports)
    ports, z_cm = ports[order], z_cm[order]
    if ports.size < 2:
        raise ArtifactRefused(
            f"{overlay_path}: fewer than two port anchors; no pitch is defined"
        )
    pitch = (z_cm[-1] - z_cm[0]) / (ports[-1] - ports[0])
    z0 = z_cm[0] - pitch * ports[0]
    residual = float(np.max(np.abs(z0 + pitch * ports - z_cm)))
    if residual > 1.0e-6:
        raise ArtifactRefused(
            f"{overlay_path}: port/z_cm anchors are not collinear "
            f"(worst residual {residual:.3e} cm); a single port pitch cannot "
            "place a port the overlay does not carry"
        )
    return float(z0), float(pitch), dict(zip(ports.tolist(), z_cm.tolist()))


def port_to_z_cm(port, law):
    """Axial position [cm] of a port under the linear law."""
    z0, pitch, _ = law
    return z0 + pitch * int(port)


def resolvable_ports(law, z_lo, z_hi):
    """Ports (>= 1) whose mapped z lands inside ``[z_lo, z_hi]``.

    The bound is the artifact's own plasma-active axial span, so this reports
    "ports this run can answer for", not a machine port census.
    """
    z0, pitch, _ = law
    lo = int(np.ceil((z_lo - z0) / pitch))
    hi = int(np.floor((z_hi - z0) / pitch))
    return list(range(max(lo, 1), hi + 1))


# --- artifact -------------------------------------------------------------


def require_adas(params):
    """Return the run's ADAS closure settings, or refuse the artifact.

    A ``janev``-configured run has no PLT tables behind it at all: its cooling
    terms are the IAEA fit expressions, and the He I fit even folds the
    ionization potential into the "cooling" it reports.  Evaluating ADAS PLT
    against such a run would report a radiance the run never had, so this
    refuses rather than substituting.
    """
    model = str(params.get("atomic_rate_model", "<absent>"))
    if model != "adas":
        raise ArtifactRefused(
            f"atomic_rate_model = {model!r}: this instrument reports the "
            "OPEN-ADAS PLT line-radiation channels and is only meaningful for "
            "an artifact whose own rate closure IS adas.  Refusing rather "
            "than substituting ADAS coefficients into a run that never used "
            "them (a janev run's cooling terms are IAEA fits, and its He I "
            "fit includes the ionization-potential loss, so the numbers are "
            "not the same quantity)."
        )
    gas = str(params.get("gas_type", "He"))
    if gas != "He":
        raise ArtifactRefused(
            f"gas_type = {gas!r}: the PLT tables in this repo are He-only"
        )
    return {
        "atomic_rate_model": model,
        "adas_low_te_extension": bool(
            params.get("adas_low_te_extension", False)
        ),
        "ne_floor": float(params.get("ne_floor", 0.0)),
        "b_Qei": float(params.get("b_Qei", 1.0)),
        "b_Qen": float(params.get("b_Qen", 1.0)),
        "b_Qei_Te_exp": float(params.get("b_Qei_Te_exp", 0.0)),
        "b_Qen_Te_exp": float(params.get("b_Qen_Te_exp", 0.0)),
    }


def locate_cell(f, port, law):
    """Return the geometry record for the cell serving ``port``."""
    z = f["geometry/z_cm"][:]
    active = np.flatnonzero(f["geometry/plasma_active"][:])
    if active.size == 0:
        raise ArtifactRefused("artifact has no plasma-active cells")
    z_lo, z_hi = float(z[active[0]]), float(z[active[-1]])
    available = resolvable_ports(law, z_lo, z_hi)
    if int(port) not in available:
        anchors = ", ".join(str(p) for p in sorted(law[2]))
        raise ArtifactRefused(
            f"port {port} does not map into this artifact's plasma-active "
            f"axial span ({z_lo:.2f} to {z_hi:.2f} cm).\n"
            f"Resolvable ports on this run: {available[0]}-{available[-1]} "
            f"(z = {port_to_z_cm(available[0], law):.2f} to "
            f"{port_to_z_cm(available[-1], law):.2f} cm, pitch "
            f"{law[1]:.2f} cm).\n"
            f"Measured probe ports carried by the overlay: {anchors}."
        )
    z_want = port_to_z_cm(port, law)
    iz = int(np.argmin(np.abs(z - z_want)))
    role = f["geometry/cell_role"][iz]
    Rp = float(f["geometry/Rp_cm"][iz])
    length = float(f["geometry/length_cm"][iz])
    volume = float(f["geometry/plasma_volume_cm3"][iz])
    if not (Rp > 0.0 and length > 0.0):
        raise ArtifactRefused(
            f"port {port} maps to cell {iz} (z = {float(z[iz]):.2f} cm), "
            "which carries no plasma cross-section; no chord is defined there"
        )
    return {
        "port": int(port),
        "z_want_cm": float(z_want),
        "cell": iz,
        "z_cell_cm": float(z[iz]),
        "cell_role": role.decode() if isinstance(role, bytes) else str(role),
        "Rp_cm": Rp,
        "chord_length_cm": 2.0 * Rp,
        "area_cm2": volume / length,
        "length_cm": length,
        "plasma_active_span_cm": [z_lo, z_hi],
        "resolvable_ports": available,
    }


def port_traces(f, iz):
    """Sliced per-frame state at one cell: ``(t_ms, n, nn, Te)``."""
    return (
        f["time"][:] * 1.0e3,
        f["n"][:, iz],
        f["nn"][:, iz],
        f["Te"][:, iz],
    )


def emissivities(n, nn, Te, closure):
    """Return ``{channel: emissivity [W cm^-3]}`` for the two PLT channels.

    Mirrors ``electron_cooling_rhs_terms``: the fused lookup is evaluated at
    ``max(n, ne_floor)`` while the density product uses the raw ``n``.
    """
    n_lookup = np.maximum(n, closure["ne_floor"])
    coeff = he_rates(
        n_lookup,
        Te,
        ("plt1", "plt2"),
        low_te_extension=closure["adas_low_te_extension"],
    )
    return {
        "plt2": coeff["plt2"] * n * n * qe_SI,
        "plt1": coeff["plt1"] * n * nn * qe_SI,
    }


# --- reduction ------------------------------------------------------------


def window_mean(t_ms, y, window):
    """Frame mean of ``y`` over ``[lo, hi]`` ms, or None if no frame lands."""
    mask = (t_ms >= window[0]) & (t_ms <= window[1])
    if not np.any(mask):
        return None
    return float(np.mean(y[mask]))


def trace_stats(t_ms, y):
    """Window means, peak and peak time for one trace."""
    k = int(np.argmax(y))
    return {
        "drive_mean": window_mean(t_ms, y, DRIVE_WINDOW_MS),
        "afterglow_mean": window_mean(t_ms, y, AFTERGLOW_WINDOW_MS),
        "peak": float(y[k]),
        "peak_time_ms": float(t_ms[k]),
    }


def ledger_crosscheck(f, iz, t_ms, eps):
    """Compare each reconstructed channel against its own ledger row.

    The ledger rows are electron energy SINKS (negative W cm^-3), so the
    comparison is against their negation.  Returns one record per channel, or
    a single unavailable record naming what is missing.
    """
    group = "electron_energy_terms_W_cm3"
    if group not in f:
        return {
            "available": False,
            "reason": f"artifact carries no {group!r} group",
        }
    out = {"available": True}
    for channel, row in LEDGER_ROW.items():
        key = f"{group}/{row}"
        if key not in f:
            out[channel] = {"available": False, "reason": f"no {row!r} row"}
            continue
        ledger = -f[key][:, iz]
        mine = eps[channel]
        scale = np.maximum(np.abs(ledger), np.abs(mine))
        good = scale > 0.0
        rel = np.zeros_like(scale)
        rel[good] = np.abs(mine[good] - ledger[good]) / scale[good]
        k = int(np.argmax(rel))
        drive_mine = window_mean(t_ms, mine, DRIVE_WINDOW_MS)
        drive_ledger = window_mean(t_ms, ledger, DRIVE_WINDOW_MS)
        out[channel] = {
            "available": True,
            "ledger_row": row,
            "drive_mean_reconstructed_W_cm3": drive_mine,
            "drive_mean_ledger_W_cm3": drive_ledger,
            "drive_mean_rel_dev": (
                None
                if not drive_ledger
                else float(drive_mine / drive_ledger - 1.0)
            ),
            "max_frame_rel_dev": float(rel[k]),
            "max_frame_rel_dev_time_ms": float(t_ms[k]),
        }
    return out


def build_report(h5_path, port):
    """Assemble every number this script reports, from the artifact alone."""
    law = port_axial_law()
    with h5py.File(h5_path, "r") as f:
        params = json.loads(f.attrs.get("params_json", "{}"))
        flags = json.loads(f.attrs.get("flags_json", "{}"))
        closure = require_adas(params)
        closure["icool_recomb"] = bool(flags.get("icool_recomb", False))
        geom = locate_cell(f, port, law)
        t_ms, n, nn, Te = port_traces(f, geom["cell"])
        eps = emissivities(n, nn, Te, closure)
        cross = ledger_crosscheck(f, geom["cell"], t_ms, eps)
        run = {
            "steps": int(f.attrs["steps"]),
            "run_status": str(f.attrs.get("run_status", "")),
            "final_time_ms": float(f.attrs["final_time"]) * 1.0e3,
            "t_breakdown_trigger_ms": float(
                f.attrs.get("t_breakdown_trigger", np.nan)
            )
            * 1.0e3,
            "saves": int(t_ms.size),
            "compiled_kernels": bool(f.attrs.get("compiled_kernels", False)),
        }

    eps["total"] = eps["plt2"] + eps["plt1"]
    traces = {"emissivity": eps}
    traces["brightness"] = {
        c: v * geom["chord_length_cm"] for c, v in eps.items()
    }
    traces["dPdz"] = {c: v * geom["area_cm2"] for c, v in eps.items()}

    stats = {
        q: {c: trace_stats(t_ms, v) for c, v in chans.items()}
        for q, chans in traces.items()
    }
    return {
        "h5": str(h5_path),
        "run": run,
        "port_law": {
            "z0_cm": law[0],
            "pitch_cm": law[1],
            "overlay": str(PORT_ANCHOR_OVERLAY),
            "anchors": {str(k): v for k, v in law[2].items()},
        },
        "geometry": geom,
        "closure": closure,
        "windows_ms": {
            "drive": list(DRIVE_WINDOW_MS),
            "afterglow": list(AFTERGLOW_WINDOW_MS),
        },
        "stats": stats,
        "ledger_crosscheck": cross,
        "state": {
            "time_ms": t_ms,
            "n_cm3": n,
            "nn_cm3": nn,
            "Te_eV": Te,
        },
        "traces": traces,
    }


# --- products -------------------------------------------------------------


def _fmt(value):
    return "n/a" if value is None else f"{value:.6e}"


def markdown_report(rep):
    """The small md table plus the header that makes it quotable."""
    g, c, r = rep["geometry"], rep["closure"], rep["run"]
    L = []
    L.append(
        f"# Port {g['port']} synthetic radiated power -- "
        f"{Path(rep['h5']).name}"
    )
    L.append("")
    L.append(
        "**Item-56 context.** The e-i channel is >= 99.9 % EUV (the 30.4 nm "
        "class resonance line dominates PLT2): this instrument reports "
        "RADIATED POWER, not what a glass-windowed spectrometer would see. "
        f"For the band split see `{BAND_SPLIT_NOTE}` (glass-transmissible "
        "> 350 nm share of the adf11 line power: 1.4e-05 to 8.5e-04 for "
        "He II, 0.052 to 0.120 for He I, at Te >= 2 eV)."
    )
    L.append("")
    L.append(
        "**Chord assumption.** The model is 1D and carries no radial profile. "
        "Emissivity is taken RADIALLY UNIFORM across the plasma disc of "
        "radius Rp(z) and zero outside it; the chord is a radial line of "
        "sight through the column axis, so L_chord = 2 Rp. This is a stated "
        "construction, not a measurement. dP/dz = emissivity x plasma "
        "cross-section carries no radial assumption."
    )
    L.append("")
    L.append("## Placement")
    L.append("")
    L.append(f"* artifact `{rep['h5']}`")
    L.append(
        f"* run: {r['steps']} steps, {r['saves']} saves, "
        f"{r['final_time_ms']:.4f} ms, status `{r['run_status']}`, "
        f"breakdown trigger {r['t_breakdown_trigger_ms']:.4f} ms"
    )
    L.append(
        f"* port {g['port']} -> z_want {g['z_want_cm']:.2f} cm "
        f"(law z = {rep['port_law']['z0_cm']:.4f} + "
        f"{rep['port_law']['pitch_cm']:.4f} x port, anchored on "
        f"`{Path(rep['port_law']['overlay']).name}`)"
    )
    L.append(
        f"* cell {g['cell']} at z {g['z_cell_cm']:.4f} cm, role "
        f"`{g['cell_role']}`, Rp {g['Rp_cm']:.4f} cm"
    )
    L.append(
        f"* chord length {g['chord_length_cm']:.4f} cm, plasma cross-section "
        f"{g['area_cm2']:.4f} cm^2, cell length {g['length_cm']:.4f} cm"
    )
    L.append(
        f"* closure: atomic_rate_model `{c['atomic_rate_model']}`, "
        f"adas_low_te_extension {c['adas_low_te_extension']}, "
        f"ne_floor {c['ne_floor']:.3e} cm^-3, icool_recomb {c['icool_recomb']}"
    )
    L.append(
        f"* the artifact's cooling scalars are b_Qei {c['b_Qei']:g} / b_Qen "
        f"{c['b_Qen']:g} (Te exponents {c['b_Qei_Te_exp']:g} / "
        f"{c['b_Qen_Te_exp']:g}); this instrument reports the UNSCALED "
        "radiated power and does not apply them."
    )
    L.append(
        f"* windows (run clock): drive {rep['windows_ms']['drive'][0]}-"
        f"{rep['windows_ms']['drive'][1]} ms, afterglow "
        f"{rep['windows_ms']['afterglow'][0]}-"
        f"{rep['windows_ms']['afterglow'][1]} ms "
        "(the registered windows of `power_ledger_sim1d.py`)"
    )
    L.append("")
    L.append("## Channel table")
    L.append("")
    L.append(
        "| quantity | unit | channel | drive mean | afterglow mean | peak "
        "| t_peak [ms] |"
    )
    L.append("|---|---|---|---|---|---|---|")
    for q in ("emissivity", "brightness", "dPdz"):
        for ch in ("plt2", "plt1", "total"):
            s = rep["stats"][q][ch]
            L.append(
                f"| {q} | {QUANTITY_UNIT[q]} | {CHANNEL_LABEL[ch]} | "
                f"{_fmt(s['drive_mean'])} | {_fmt(s['afterglow_mean'])} | "
                f"{_fmt(s['peak'])} | {s['peak_time_ms']:.4f} |"
            )
    L.append("")
    L.append("## Self-consistency against the artifact's own term ledger")
    L.append("")
    cross = rep["ledger_crosscheck"]
    if not cross.get("available"):
        L.append(f"Not available: {cross.get('reason')}.")
    else:
        L.append(
            "Reconstructed emissivity vs the `electron_energy_terms_W_cm3` "
            "row the run itself saved (rows are electron sinks; compared "
            "negated)."
        )
        L.append("")
        L.append(
            "| channel | ledger row | drive mean recon [W cm^-3] | "
            "drive mean ledger [W cm^-3] | rel dev | worst frame rel dev "
            "| at [ms] |"
        )
        L.append("|---|---|---|---|---|---|---|")
        for ch in ("plt2", "plt1"):
            e = cross.get(ch, {})
            if not e.get("available"):
                L.append(
                    f"| {CHANNEL_LABEL[ch]} | -- | -- | -- | -- | -- | "
                    f"{e.get('reason', 'not present')} |"
                )
                continue
            L.append(
                f"| {CHANNEL_LABEL[ch]} | `{e['ledger_row']}` | "
                f"{_fmt(e['drive_mean_reconstructed_W_cm3'])} | "
                f"{_fmt(e['drive_mean_ledger_W_cm3'])} | "
                f"{_fmt(e['drive_mean_rel_dev'])} | "
                f"{e['max_frame_rel_dev']:.3e} | "
                f"{e['max_frame_rel_dev_time_ms']:.4f} |"
            )
    L.append("")
    return "\n".join(L)


def json_report(rep):
    """Full machine-readable product, traces included."""
    out = {
        k: rep[k]
        for k in (
            "h5",
            "run",
            "port_law",
            "closure",
            "windows_ms",
            "stats",
            "ledger_crosscheck",
        )
    }
    out["geometry"] = dict(rep["geometry"])
    out["units"] = dict(QUANTITY_UNIT)
    out["channel_label"] = dict(CHANNEL_LABEL)
    out["chord_assumption"] = (
        "emissivity radially uniform across the plasma disc of radius Rp(z), "
        "zero outside; chord is a radial line of sight through the column "
        "axis, so L_chord = 2 Rp"
    )
    out["band_note"] = (
        "e-i channel is >= 99.9 % EUV (30.4 nm class); this reports power, "
        f"not a glass-windowed spectrometer signal. Band split: "
        f"{BAND_SPLIT_NOTE}"
    )
    out["state"] = {k: v.tolist() for k, v in rep["state"].items()}
    out["traces"] = {
        q: {c: v.tolist() for c, v in chans.items()}
        for q, chans in rep["traces"].items()
    }
    return out


CHANNEL_STYLE = {
    "plt2": ("tab:red", "-"),
    "plt1": ("tab:blue", "-"),
    "total": ("0.25", "--"),
}

#: Decades shown below the panel peak.  The PLT coefficients fall off
#: exponentially once Te drops below the excitation thresholds, so an
#: unbounded log axis spans ~50 decades of physically dead afterglow and
#: compresses the whole discharge into the top of the frame.  Traces are
#: clipped to this floor for drawing only; the tabulated numbers are not.
PLOT_DECADES = 8.0


def write_plot(rep, path, dpi=180):
    """Two-panel per-channel time trace: chord brightness, then emissivity."""
    g = rep["geometry"]
    t = rep["state"]["time_ms"]
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    for ax, q in zip(axes, ("brightness", "emissivity")):
        peak = max(
            float(np.max(rep["traces"][q][ch])) for ch in CHANNEL_STYLE
        )
        floor = peak * 10.0**-PLOT_DECADES
        for ch in ("plt2", "plt1", "total"):
            color, ls = CHANNEL_STYLE[ch]
            ax.plot(
                t,
                np.maximum(rep["traces"][q][ch], floor),
                color=color,
                ls=ls,
                lw=1.3,
                label=CHANNEL_LABEL[ch],
            )
        for window, face in (
            (rep["windows_ms"]["drive"], "tab:orange"),
            (rep["windows_ms"]["afterglow"], "tab:green"),
        ):
            ax.axvspan(window[0], window[1], color=face, alpha=0.10, lw=0)
        ax.set_yscale("log")
        ax.set_ylim(floor, peak * 3.0)
        ax.grid(True, which="major", alpha=0.25)

    axes[0].set_ylabel(
        f"chord surface brightness [{QUANTITY_UNIT['brightness']}]"
    )
    axes[0].set_title(
        f"Port {g['port']} synthetic radiated power -- "
        f"{Path(rep['h5']).name}\n"
        f"cell {g['cell']}, z = {g['z_cell_cm']:.2f} cm, chord "
        f"{g['chord_length_cm']:.2f} cm (uniform-disc assumption); "
        "e-i channel is EUV, not visible",
        fontsize=10,
    )
    axes[0].legend(loc="lower left", fontsize=8, framealpha=0.9)

    axes[1].set_ylabel(f"emissivity [{QUANTITY_UNIT['emissivity']}]")
    axes[1].set_xlabel("time [ms, run clock]")
    twin = axes[1].twinx()
    twin.set_yscale("log")
    lo, hi = axes[1].get_ylim()
    twin.set_ylim(lo * g["area_cm2"], hi * g["area_cm2"])
    twin.set_ylabel(
        f"dP/dz [{QUANTITY_UNIT['dPdz']}]  "
        f"(= emissivity x {g['area_cm2']:.1f} cm^2)"
    )
    axes[1].annotate(
        f"traces clipped at {PLOT_DECADES:.0f} decades below the panel peak "
        "(drawing only; the tables are not clipped)",
        xy=(0.005, 0.02),
        xycoords="axes fraction",
        fontsize=7.5,
        color="0.35",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# --- CLI ------------------------------------------------------------------


def _parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--h5", type=Path, default=DEFAULT_H5)
    parser.add_argument("--port", type=int, default=27)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=None,
        help="output path stem; default "
        "<artifact dir>/port_radiance_<artifact stem>_p<port>",
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if not args.h5.exists():
        raise ArtifactRefused(f"artifact not found: {args.h5}")
    rep = build_report(args.h5, args.port)
    stem = args.output_stem
    if stem is None:
        stem = args.h5.resolve().parent / (
            f"port_radiance_{args.h5.stem}_p{args.port}"
        )
    stem.parent.mkdir(parents=True, exist_ok=True)

    png = stem.with_suffix(".png")
    md = stem.with_suffix(".md")
    js = stem.with_suffix(".json")
    write_plot(rep, png, dpi=args.dpi)
    md.write_text(markdown_report(rep))
    js.write_text(json.dumps(json_report(rep), indent=2))

    print(markdown_report(rep))
    print(f"wrote {png}")
    print(f"wrote {md}")
    print(f"wrote {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
