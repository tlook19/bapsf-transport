# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Compiled scalar kernels for the cathode sheath solver (SPIKE, D4).

Scope
-----
ONE kernel: ``j_eth_crit``, a faithful transcription of
``_cathode_solver._j_eth_crit``. The cathode solve is ~31 % of a production
run and is call-overhead bound (2026-07-30 profile), so the question this
module exists to answer is narrow and empirical:

    does a libc.math transcription with UNCHANGED operation order reproduce
    the golden bit-exactly?

CPython's ``math.exp``/``math.sqrt`` are thin wrappers over the platform
libm -- the SAME libm this module links against -- so the transcription has
a real chance of being bit-identical rather than merely close. It is not
guaranteed: the compiler may contract ``1.0 + 2.0*psi`` into an FMA, which
rounds once instead of twice. The build therefore passes
``-ffp-contract=off`` (see ``build_ext.py``); that is a flag that DISABLES a
transform, which is the opposite of fastmath.

Faithfulness rules for anything added here
------------------------------------------
* identical operation order and identical parenthesisation to the Python;
* ``pow(psi, 3.0)`` where the Python writes ``psi**3`` -- CPython's
  ``float.__pow__`` calls libm ``pow()``, and ``psi*psi*psi`` is a DIFFERENT
  computation that can differ in the last bit;
* no ``-ffast-math``, no reassociation, no ``--fast-math``-adjacent pragmas;
* every physical constant checked against the Python module's value at bind
  time (``check_constants``), so a divergent literal is a loud failure and
  never a quiet change of physics.

Nothing RUNS this module's kernel unless ``CABLP_COMPILED_KERNELS`` opts in;
see ``cablp.funcs._kernels``. (``smoke_sim1d`` imports it directly regardless,
to compare it against the pure kernel -- comparing is not running.)
"""

from libc.math cimport exp, sqrt, pow

# Proton-to-electron mass ratio, in the same two literals and the same order
# as ``_cathode_solver``. DUPLICATED DELIBERATELY: importing the value from
# ``_cathode_solver`` would be a circular import (that module is what binds
# this one). ``check_constants`` below asserts the two agree bit-for-bit at
# bind time, so the duplication cannot drift silently.
cdef double _ME_CGS = 9.1093837015e-28   # Electron mass [g]
cdef double _MP_CGS = 1.67262192369e-24  # Proton mass [g]
cdef double _PEMR = _MP_CGS / _ME_CGS

# Reported in artifact provenance so a run names the kernel that produced it.
KERNEL_ID = "cython/_cathode_kernels_cy/j_eth_crit"


def pemr():
    """Return this module's proton-to-electron mass ratio."""
    return _PEMR


def check_constants(double pemr_py):
    """Raise unless the Python module's ``_pemr`` is bit-identical to ours.

    Called once, at bind time, by ``_cathode_solver``. A mismatch means the
    duplicated literals above have drifted from the authoritative ones, which
    would silently change the physics of every compiled-kernel run -- so it is
    a hard, immediate failure rather than a warning.
    """
    if pemr_py != _PEMR:
        raise ValueError(
            "compiled cathode kernels disagree with _cathode_solver on the "
            f"proton-to-electron mass ratio: compiled _PEMR={_PEMR!r} vs "
            f"python _pemr={pemr_py!r}. The duplicated mass literals in "
            "_cathode_kernels_cy.pyx have drifted; fix them rather than "
            "relaxing this check."
        )


def j_eth_crit(double psi, double J_i, double mu):
    """Scaled critical thermionic current J_eth_crit(psi_c_plus).

    Faithful transcription of ``_cathode_solver._j_eth_crit``; see that
    docstring for the physics. Operation order here is identical, statement
    for statement, so any difference in the result is a floating-point
    artifact of the C compilation and not of the formula.
    """
    cdef double prefactor = J_i * sqrt(mu * _PEMR)
    cdef double numer
    if psi <= 0.0:
        return 0.0
    if psi < 1e-3:
        # Taylor expansion to avoid catastrophic cancellation.
        numer = pow(psi, 3.0) * (1.0 / 3.0 - 7.0 * psi / 12.0)
        return prefactor * numer / sqrt(2.0 * psi)
    return (
        prefactor
        * (exp(-psi) + sqrt(1.0 + 2.0 * psi) - 2.0)
        / sqrt(2.0 * psi)
    )
