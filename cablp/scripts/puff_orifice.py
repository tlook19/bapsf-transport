"""Re-export of :mod:`cablp.solvers._sim1d.physics.puff_orifice`.

The tube-beamed launch row moved INTO the package when the fluid solver gained
``gas_puff_profile = "orifice"``: the row is now derived in one place and read
by the solver's own puff sites as well as by the kinetic instruments here, so
the two cannot desync. This module is the scripts-side name those instruments
already import (``mc_neutrals.py``, ``porf_footprint_proof.py``, and
``kn2zone.py`` through ``mc_neutrals``); it adds nothing and re-exports the
package module's public surface unchanged.

Read the package module for the geometry, the angular law, the hardware
bracket and the disclosure list. Nothing is documented twice here.
"""

from cablp.solvers._sim1d.physics.puff_orifice import (  # noqa: F401
    NARROW_ASPECT_RATIO,
    PIPE_ID_CM_BRACKET,
    PIPE_LENGTH_CM_MIN,
    PORT_BORE_DIAMETER_CM,
    PORT_CENTER_Z_CM,
    PORT_SPAN_Z_CM,
    QUADRATURE,
    WALL_RADIUS_CM,
    clausing_intensity,
    describe,
    launch_row,
    launch_row_bracket,
    launch_row_for_grid,
    mass_span,
)

__all__ = [
    "NARROW_ASPECT_RATIO",
    "PIPE_ID_CM_BRACKET",
    "PIPE_LENGTH_CM_MIN",
    "PORT_BORE_DIAMETER_CM",
    "PORT_CENTER_Z_CM",
    "PORT_SPAN_Z_CM",
    "QUADRATURE",
    "WALL_RADIUS_CM",
    "clausing_intensity",
    "describe",
    "launch_row",
    "launch_row_bracket",
    "launch_row_for_grid",
    "mass_span",
]
