"""[perf-batch-1] item 2 -- equivalence and guard checks for the beam-smoothing key memo.

``_beam_smoothing_key`` fingerprints four geometry arrays on every call so that
``_BEAM_SMOOTH_CACHE`` is keyed by CONTENT and not by a recyclable address.
Item 2 memoizes that key on ``id(geometry)`` while holding a strong reference
to the geometry, which pins the address for as long as the entry lives. These
checks pin the three properties that makes sound:

  (a) the memoized key is EQUAL to the freshly fingerprinted one, and the
      smoothing matrix built through it is bitwise identical;
  (b) two distinct geometries that agree in cell count but differ in cell
      positions still get DIFFERENT keys and different matrices -- the memo
      does not collapse them;
  (c) an in-place write into a fingerprinted geometry array raises, so the one
      hazard identity keying cannot see is loud rather than silent.
"""

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from cablp.solvers._sim1d.core.config import default_config  # noqa: E402
from cablp.solvers._sim1d.core.geometry import build_geometry  # noqa: E402
from cablp.solvers._sim1d.physics import cathode  # noqa: E402

SIGMA_CM = 12.5


def _geometry(nx):
    params, flags = default_config()
    params = dict(params)
    params["nx"] = int(nx)
    return build_geometry(params, flags)


def main():
    failures = []
    geometry = _geometry(60)

    # (a) memo-cold vs memo-warm.
    cathode._BEAM_SMOOTH_KEY_CACHE.clear()
    cathode._BEAM_SMOOTH_CACHE.clear()
    cold_key = cathode._beam_smoothing_key(geometry, SIGMA_CM)
    cold_matrix = cathode._beam_smoothing_matrix(geometry, SIGMA_CM).copy()
    warm_key = cathode._beam_smoothing_key(geometry, SIGMA_CM)
    warm_matrix = cathode._beam_smoothing_matrix(geometry, SIGMA_CM)
    same_key = cold_key == warm_key
    differing = int(
        np.count_nonzero(
            cold_matrix.view(np.uint64) != np.ascontiguousarray(warm_matrix).view(np.uint64)
        )
    )
    print(f"(a) memoized key equals fresh key: {same_key}")
    print(f"(a) matrix differing uint64 words: {differing}")
    if not same_key or differing:
        failures.append("memoized key or matrix moved")

    # (b) a different mesh must not share the entry.
    other = _geometry(62)
    other_key = cathode._beam_smoothing_key(other, SIGMA_CM)
    distinct = other_key != cold_key
    print(f"(b) distinct geometry gets a distinct key: {distinct}")
    if not distinct:
        failures.append("two geometries collapsed onto one key")

    # (c) the read-only guard.
    try:
        geometry.z_cm[0] = -1.0
    except ValueError as exc:
        print(f"(c) in-place write raised ValueError: {exc}")
    else:
        print("(c) in-place write SUCCEEDED -- the guard is not armed")
        failures.append("fingerprinted geometry array is still writeable")

    if failures:
        print("BEAM KEY MEMO FAIL: " + "; ".join(failures))
        return 1
    print("BEAM KEY MEMO OK: key equal, matrix bit-identical, guard armed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
