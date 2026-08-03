"""Opt-in selection of compiled scalar kernels, with loud provenance.

Decision D3/D4 (Tom, 2026-08-02), extended the same day to the Tier A cathode
unit. The compiled path answers whether a Cython transcription of the cathode
scalar kernels -- and of the sheath root find built on them -- reproduces the
golden bit-exactly. It is therefore off by default and stays off unless a
caller says otherwise, in as few words as possible:

    CABLP_COMPILED_KERNELS=1 python scripts/baseline_sim1d.py --verify

Why an environment variable rather than a config key: the binding happens once,
at module import, so that the hot path is a plain function object with no
per-call branch -- and it must therefore be decided before any ``LAPDSim1D`` is
constructed. A config key would be read too late, and threading it through
``input_dict`` would make the kernel choice look like a physics parameter,
which it is not. It is a build/deployment fact about the process, and it is
recorded in every artifact (``compiled_kernels`` in the result metadata) so no
run is anonymous about which arithmetic produced it.

Three states, no silent ones:

* unset / ``0`` / ``false`` / ``no`` / ``off`` / empty -- pure Python. The
  default, and byte-for-byte the historical behaviour.
* ``1`` / ``true`` / ``yes`` / ``on`` -- compiled. If the extension is not
  importable this raises at import time: opting in and quietly getting the
  pure path would mean a benchmark or a bit-exactness verdict that describes
  code the process never ran.
* anything else -- ``ValueError``. A typo'd opt-in must not read as "off".
"""

import importlib
import os

ENV_VAR = "CABLP_COMPILED_KERNELS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"", "0", "false", "no", "off"})

_MODULE = "cablp.funcs._cathode_kernels_cy"

#: Provenance string recorded in artifact metadata when nothing is compiled.
PURE_PROVENANCE = "pure"


def compiled_kernels_requested():
    """Return True when the environment opts in; raise on an unreadable value."""
    raw = os.environ.get(ENV_VAR)
    if raw is None:
        return False
    value = raw.strip().lower()
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{ENV_VAR}={raw!r} is not a recognised on/off value. Use one of "
        f"{sorted(_TRUTHY)} to opt in to the compiled kernels or one of "
        f"{sorted(_FALSY)} (or leave it unset) for the default pure-Python "
        "path. It is not treated as 'off', because a typo'd opt-in would "
        "silently benchmark or verify the wrong code."
    )


def _load():
    """Import the compiled kernel module, or return None when not opted in."""
    if not compiled_kernels_requested():
        return None
    try:
        return importlib.import_module(_MODULE)
    except ImportError as error:
        raise RuntimeError(
            f"{ENV_VAR} opts in to the compiled cathode kernels but "
            f"{_MODULE} could not be imported ({error}). Build it with "
            "`python build_ext.py --inplace` from the cablp/ directory (needs "
            "Cython and a C compiler), or unset the variable to run the "
            "default pure-Python path. This is deliberately NOT a fallback: "
            "silently running pure Python here would mean a timing or "
            "bit-exactness result that describes code this process never ran."
        ) from error


#: The compiled kernel module, or ``None`` on the default pure path. Bound
#: once at import so the hot path never branches.
COMPILED_KERNELS = _load()

#: Short provenance string for artifact metadata: ``"pure"`` or the compiled
#: module's own ``KERNEL_ID``. An artifact with NO ``compiled_kernels`` entry
#: predates this selector and was produced on the pure path.
PROVENANCE = (
    PURE_PROVENANCE
    if COMPILED_KERNELS is None
    else str(getattr(COMPILED_KERNELS, "KERNEL_ID", _MODULE))
)
