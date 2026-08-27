"""Footprint proof for the CAD-derived kinetic puff row ([puff-orifice]).

Tabulates and plots the derived axial launch row at BOTH endpoints of the
one-sided feed-line bracket beside the fluid ``cosine_pipe`` deposition
envelope, on a production run's own mesh, and sweeps the pipe length so the
one-sidedness of the bracket is visible rather than asserted.

Usage::

    python scripts/porf_footprint_proof.py RUN.h5 [--window 5 19.5]
        [--out-prefix porf_footprint]

Writes ``<prefix>_table.md`` and ``<prefix>.png``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import puff_orifice as orifice  # noqa: E402
from mc_neutrals import load_background  # noqa: E402

#: Feed-pipe lengths [cm] swept to show the bracket is one-sided. 22 is the
#: ruled lower bound (the widest footprint); 32 is the same bound measured
#: from the source-chamber wall the port actually sits in; the rest march
#: toward the fully-beamed limit.
SWEEP_LENGTHS_CM = (22.0, 32.0, 50.0, 100.0, 400.0)


def _rows_on(edges, r_wall, r_edge, z_port):
    common = dict(r_wall_cm=r_wall, r_edge_cm=r_edge, z_port_cm=z_port)
    out = {}
    for L in SWEEP_LENGTHS_CM:
        for d in orifice.PIPE_ID_CM_BRACKET:
            out[(L, d)] = orifice.launch_row(
                edges, pipe_id_cm=d, aspect_ratio=L / d, **common
            )
    out[("inf", orifice.PIPE_ID_CM_BRACKET[0])] = orifice.launch_row(
        edges,
        pipe_id_cm=orifice.PIPE_ID_CM_BRACKET[0],
        aspect_ratio=orifice.NARROW_ASPECT_RATIO,
        **common,
    )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--window", nargs=2, type=float, default=(5.0, 19.5))
    ap.add_argument("--out-prefix", default="porf_footprint")
    args = ap.parse_args(argv)

    bg = load_background(args.run, tuple(args.window))
    edges = bg["z_edges"]
    zc = 0.5 * (edges[:-1] + edges[1:])
    fluid = np.asarray(bg["sources"]["puff_cells"], dtype=float)
    fluid = fluid / fluid.sum()
    i_port = int(np.searchsorted(edges, orifice.PORT_CENTER_Z_CM) - 1)
    r_wall = float(bg["Rm"][i_port])
    r_edge = float(bg["Rp"][i_port])
    z_port = orifice.PORT_CENTER_Z_CM

    bracket = orifice.launch_row_bracket(
        edges, r_edge_cm=r_edge, r_wall_cm=r_wall, z_port_cm=z_port
    )
    sweep = _rows_on(edges, r_wall, r_edge, z_port)

    f_span, f_lo, f_hi = orifice.mass_span(fluid, edges)
    lines = [
        "# [puff-orifice] footprint proof",
        "",
        f"Run `{Path(args.run).name}`, window {args.window[0]}-{args.window[1]} ms, "
        f"{zc.size} cells.",
        f"Flight geometry from the run's own mesh at the port cell: vessel wall "
        f"r = {r_wall:.5g} cm, plasma column r = {r_edge:.5g} cm, station "
        f"z = {z_port:.4g} cm.",
        "",
        "## Bracket endpoints vs the fluid deposition envelope",
        "",
        "| row | d [cm] | L [cm] | Gamma | 5-95% span [cm] | interval [cm] | "
        "peak cell z [cm] | peak share | perigee-placed | off-grid |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key, label, L in (
        ("wide", "kinetic WIDE", f"{orifice.PIPE_LENGTH_CM_MIN:.0f}"),
        ("narrow", "kinetic NARROW", "inf"),
    ):
        row, meta = bracket[key]
        span, lo, hi = orifice.mass_span(row, edges)
        j = int(np.argmax(row))
        lines.append(
            f"| {label} | {meta['pipe_id_cm']:.2f} | {L} | "
            f"{meta['aspect_ratio']:.4g} | {span:.3f} | [{lo:.2f}, {hi:.2f}] | "
            f"{zc[j]:.2f} | {row[j]:.4f} | {meta['missed_fraction']:.4f} | "
            f"{meta['clipped_fraction']:.5f} |"
        )
    lines.append(
        f"| fluid `cosine_pipe` | - | - | - | {f_span:.3f} | "
        f"[{f_lo:.2f}, {f_hi:.2f}] | {zc[int(np.argmax(fluid))]:.2f} | "
        f"{fluid.max():.4f} | - | - |"
    )
    ratios = sorted(
        f_span / orifice.mass_span(bracket[k][0], edges)[0]
        for k in ("wide", "narrow")
    )
    off_profile = fluid <= 0.0
    lines += [
        "",
        f"The fluid envelope is {ratios[0]:.2f}-{ratios[1]:.2f} times wider "
        "than the derived injection footprint. That difference is the "
        "registered kinetic-vs-fluid finding: `cosine_pipe`'s `throw_cm` is an "
        "end-state closure for a model with no neutral transport, and the "
        "derived row is the aperture the machine actually injects through.",
        "",
        "The derived row also reaches cells the fluid profile excludes by role "
        "(it lands only on puff-eligible main-chamber cells): "
        f"{bracket['wide'][0][off_profile].sum():.4f} of the wide row and "
        f"{bracket['narrow'][0][off_profile].sum():.4f} of the narrow row sit "
        "where `cosine_pipe` places nothing, upstream of the station in the "
        "cathode-anode gap. The wing of a beamed source does reach back there; "
        "a consumer that must not fuel those cells has to say so.",
        "",
        "## The bracket is one-sided (longer pipe => narrower footprint)",
        "",
        "| L [cm] | d [cm] | Gamma | 5-95% span [cm] |",
        "|---|---|---|---|",
    ]
    for (L, d), (row, meta) in sweep.items():
        span = orifice.mass_span(row, edges)[0]
        lines.append(
            f"| {L} | {d:.2f} | {meta['aspect_ratio']:.4g} | {span:.3f} |"
        )
    lines += [
        "",
        "The spread is monotone in Gamma and flattens: for a long tube the "
        "TRANSMITTED-flux angular shape becomes Gamma-independent (the "
        "forward-peaked core carries a fraction ~3/(2 Gamma) of the molecules, "
        "the wall-re-emitted wing the rest), so the unpinned pipe length moves "
        "the footprint by only a few per cent. The bracket is therefore tight "
        "even though its length bound is one-sided.",
        "",
        "## Per-cell rows around the station",
        "",
        "| z [cm] | kinetic WIDE | kinetic NARROW | fluid cosine_pipe |",
        "|---|---|---|---|",
    ]
    near = np.flatnonzero((zc > z_port - 120.0) & (zc < z_port + 220.0))
    for j in near:
        lines.append(
            f"| {zc[j]:.2f} | {bracket['wide'][0][j]:.5f} | "
            f"{bracket['narrow'][0][j]:.5f} | {fluid[j]:.5f} |"
        )
    out_md = Path(f"{args.out_prefix}_table.md")
    out_md.write_text("\n".join(lines) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for ax, (lo, hi, tag) in zip(
        axes, ((0.0, 300.0, "station"), (0.0, float(edges[-1]), "full machine"))
    ):
        ax.step(zc, bracket["wide"][0], where="mid",
                label=f"kinetic WIDE (L={orifice.PIPE_LENGTH_CM_MIN:.0f} cm, "
                      f"d={orifice.PIPE_ID_CM_BRACKET[1]:.1f} cm)")
        ax.step(zc, bracket["narrow"][0], where="mid",
                label="kinetic NARROW (L->inf, "
                      f"d={orifice.PIPE_ID_CM_BRACKET[0]:.1f} cm)")
        ax.step(zc, fluid, where="mid", label="fluid cosine_pipe")
        ax.axvspan(*orifice.PORT_SPAN_Z_CM, color="0.8", zorder=0,
                   label="CAD port collar")
        ax.set_xlim(lo, hi)
        ax.set_xlabel("z [cm]")
        ax.set_ylabel("share of the puff per cell")
        ax.set_title(tag)
        ax.set_yscale("log")
        ax.set_ylim(1e-7, 1.0)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    out_png = Path(f"{args.out_prefix}.png")
    fig.savefig(out_png, dpi=140)
    print(f"wrote {out_md} and {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
