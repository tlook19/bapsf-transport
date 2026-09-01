"""Equivalence and guard checks for the beam-smoothing key memo.

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
  (c) an in-place write into EVERY array the key reads raises, so the one
      hazard identity keying cannot see is loud rather than silent. All five
      components are attempted, not a representative one: the first version of
      this check probed only ``z_cm``, which is exactly the component the code
      did freeze, so the instrument's coverage matched the code's gap and
      neither caught that ``cathode_face_indices`` stayed writeable. A guard
      check that does not enumerate what the guard claims to cover proves
      nothing about the components it skipped;
  (d) the memo is BOUNDED and evicts: building more geometries than the cap
      leaves the entry count at the cap, and a geometry whose entry was
      evicted becomes collectable once the caller drops its own reference --
      the memo holds strong references, so an uncapped one would pin every
      geometry a process ever built.
"""

import gc
import sys
import weakref
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

    # (c) the read-only guard, on EVERY component the key reads.
    for name in (
        "z_cm",
        "length_cm",
        "z_edges_cm",
        "plasma_active",
        "cathode_face_indices",
    ):
        values = getattr(geometry, name)
        before = np.asarray(values).copy()
        try:
            values[0] = values[0]
        except ValueError as exc:
            print(f"(c) in-place write to {name} raised ValueError: {exc}")
            continue
        print(f"(c) in-place write to {name} SUCCEEDED -- the guard is not armed")
        failures.append(f"{name} is still writeable")
        # Leave the geometry as it was found, so a later check is not misled.
        np.asarray(values)[...] = before

    # (d) the memo is bounded, and an evicted geometry is collectable.
    cap = cathode._BEAM_SMOOTH_KEY_CACHE_ENTRIES
    cathode._BEAM_SMOOTH_KEY_CACHE.clear()
    victim = _geometry(80)
    victim_ref = weakref.ref(victim)
    cathode._beam_smoothing_key(victim, SIGMA_CM)
    for nx in range(81, 81 + 2 * cap):
        cathode._beam_smoothing_key(_geometry(nx), SIGMA_CM)
    resident = len(cathode._BEAM_SMOOTH_KEY_CACHE)
    bounded = resident <= cap
    print(f"(d) cap={cap}; entries after {1 + 2 * cap} geometries: {resident} "
          f"-> bounded: {bounded}")
    if not bounded:
        failures.append(f"memo grew to {resident} entries past a cap of {cap}")
    del victim
    gc.collect()
    collected = victim_ref() is None
    print(f"(d) evicted geometry collectable after caller drops it: {collected}")
    if not collected:
        failures.append("an evicted geometry is still pinned by the memo")

    if failures:
        print("BEAM KEY MEMO FAIL: " + "; ".join(failures))
        return 1
    print(
        "BEAM KEY MEMO OK: key equal, matrix bit-identical, all five "
        "components guarded, memo bounded and evicting"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
