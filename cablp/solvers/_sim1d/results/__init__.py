"""Result IO and analysis helpers for sim1d.

The names in ``__all__`` are the phase-3 qualified-capture sub-API, the one
group this package re-exports. Its other members -- ``io``, ``health``,
``restart`` -- are imported by their own module path, and the five of their
symbols that carry the package surface (``save_result_hdf5``,
``load_result_hdf5``, ``summarize_result``, and the restart-state pair) are
published on ``cablp.solvers._sim1d``.
"""

from .phase3_capture import (
    PHASE3_CAPTURE_PROFILE,
    blank_crosswalk_inventory,
    load_qualified_capture,
    reserve_run_id,
    validate_run_id,
    write_qualified_capture,
)

__all__ = [
    "PHASE3_CAPTURE_PROFILE",
    "blank_crosswalk_inventory",
    "load_qualified_capture",
    "reserve_run_id",
    "validate_run_id",
    "write_qualified_capture",
]
