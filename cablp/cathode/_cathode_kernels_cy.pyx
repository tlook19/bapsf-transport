# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Compiled scalar kernels for the cathode sheath solver and the beam march.

Scope
-----
The two coherent scalar units a production run spends its time in: the
current-driven cathode solve (~31 % of a run) and the CSDA beam-deposition
substep march (~32 %). Both are call-overhead bound (2026-07-30 profile;
2026-08-02 cost read), so the question this module exists to answer is narrow
and empirical:

    does a libc.math transcription with UNCHANGED operation order reproduce
    the golden bit-exactly?

The module is named for the cathode unit, which came first. The beam march
lives here rather than in a second extension because it is BUILT ON the
cathode unit: its Coulomb stopping power calls ``_c_log_ei_c`` and it shares
``_ERG_PER_EV`` / ``_ME_CGS``. One translation unit means the shared leaf is
called directly and the shared constants cannot drift between two copies.

What is here, in dependency order:

``j_eth_crit``
    ``_cathode_solver._j_eth_crit`` -- the space-charge release current.
``c_log_ei``
    ``_cathode_solver._c_log_ei`` -- the NRL electron-ion Coulomb log.
``compute_l_b``
    ``_cathode_solver._compute_l_b`` -- the beam mean free path.
``schottky_lowering_eV``
    ``_cathode_solver_idriven._schottky_lowering_eV`` -- work-function
    lowering from the Child-Langmuir surface field.
``annular_state_schottky``
    ``_cathode_solver_idriven._annular_state_schottky`` -- the per-annulus
    emission state, i.e. the BODY of the root-find residual.
``solve_psi_annular_schottky``
    the whole root find for the production branch (annular + Schottky, no
    thermal bridge): the geometric bracket-doubling ladder, the
    float-degeneracy plateau fallback, and the bracketed solve itself,
    with the residual evaluated in C. This is where the payoff is: the
    root find calls the residual ~50-100 times per solve and every one of
    those was a Python round-trip through a closure.
``csda_march`` (+ ``CsdaTables``)
    ``_beam_deposition.deposit_beam``'s per-cell / per-substep double loop,
    with the three cross-section table interpolations, the secondary-energy,
    Coulomb-stopping and quasilinear-length leaves, and the eight per-cell
    accumulator banks all inside ONE compiled boundary. This is the whole of
    what the 2026-08-02 cost read calls Kernel 1; the prologue, the result
    dataclass and the product/tail walks stay in Python.

The root find uses ``scipy.optimize.cython_optimize.brentq`` -- SciPy's own
C Brent implementation, the SAME ``Zeros/brentq.c`` the Python
``scipy.optimize.brentq`` wraps. Nothing about the algorithm is
re-derived here, which is the whole point: a hand-transcribed Brent would
be a second thing to get bit-right.

CPython's ``math.exp``/``math.sqrt``/``math.log``/``math.pow`` are thin
wrappers over the platform libm -- the SAME libm this module links against
-- so the transcription has a real chance of being bit-identical rather
than merely close. It is not guaranteed: the compiler may contract
``1.0 + 2.0*psi`` into an FMA, which rounds once instead of twice. The
build therefore passes ``-ffp-contract=off`` (see ``build_ext.py``); that
is a flag that DISABLES a transform, which is the opposite of fastmath.

Faithfulness rules for anything added here
------------------------------------------
* identical operation order and identical parenthesisation to the Python;
* ``pow(psi, 3.0)`` where the Python writes ``psi**3`` -- CPython's
  ``float.__pow__`` calls libm ``pow()``, and ``psi*psi*psi`` is a DIFFERENT
  computation that can differ in the last bit;
* ``min(x, 700.0)`` transcribed as the Python conditional, NOT ``fmin`` --
  they disagree on NaN;
* no ``-ffast-math``, no reassociation, no ``--fast-math``-adjacent pragmas;
* every physical constant checked against the Python module's value at bind
  time (``check_constants``/``check_constants_idriven``/``check_constants_beam``),
  so a divergent literal is a loud failure and never a quiet change of physics;
* ONE deliberate exception to "no contraction", in ``_interp_scalar``: numpy's
  own ``arr_interp`` is compiled WITH contraction on this platform, so its
  lerp is a single-rounding fused multiply-add. Reproducing it needs an
  explicit ``fma()``. See that function's docstring for the measurement.

Nothing RUNS this module's kernels unless ``CABLP_COMPILED_KERNELS`` opts
in; see ``cablp.cathode.kernels``. (``smoke_sim1d`` imports it directly
regardless, to compare it against the pure kernels -- comparing is not
running.)
"""

from libc.math cimport (
    INFINITY, atan, exp, fabs, fma, isfinite, isnan, log, log1p, pow, sqrt,
)
from libc.stdlib cimport malloc, free

import numpy as np

from scipy.optimize.cython_optimize cimport brentq, zeros_full_output

# Proton-to-electron mass ratio, in the same two literals and the same order
# as ``_cathode_solver``. DUPLICATED DELIBERATELY: importing the values from
# ``_cathode_solver`` would be a circular import (that module is what binds
# this one). The ``check_constants*`` functions below assert the duplicates
# agree bit-for-bit at bind time, so the duplication cannot drift silently.
cdef double _ME_CGS = 9.1093837015e-28   # Electron mass [g]
cdef double _MP_CGS = 1.67262192369e-24  # Proton mass [g]
cdef double _PEMR = _MP_CGS / _ME_CGS
cdef double _E_SI = 1.602176634e-19      # Electron charge [C] = [J/eV]
cdef double _ERG_PER_EV = _E_SI * 1.0e7  # eV -> erg
# Schottky constant: dphi[eV] = sqrt(e E / (4 pi eps0)) with E in V/m.
cdef double _SCHOTTKY_EV_PER_SQRT_V_M = 3.7946865e-5

# Beam-deposition constants, duplicated from ``_beam_deposition`` on the same
# terms as the cathode ones above and checked by ``check_constants_beam``.
# ``_E4_CGS`` is not a literal there -- it is ``(4.80320425e-10) ** 4``, which
# CPython evaluates through libm ``pow`` -- so it is built the same way here
# rather than transcribed as a decimal, and the bind-time check is what proves
# the two agree bit-for-bit.
cdef double _PI = 3.141592653589793
cdef double _OMEGA_PE_COEFF = 5.64e4
cdef double _E4_CGS = 0.0
_E4_CGS = pow(4.80320425e-10, 4.0)

# Coulomb-model selectors, mirroring ``_beam_deposition.coulomb_stopping_eV_per_cm``.
# An unrecognised model never reaches this module: the Python dispatcher only
# calls in for a model it can name, so the historical ValueError still comes
# from the historical place.
cdef int _COULOMB_FAST_ELECTRON = 0
cdef int _COULOMB_LEGACY_TAU_EI = 1

# brentq controls, matching the Python call sites verbatim
# (``_cathode_solver_idriven.solve_idriven``). ``maxiter`` is SciPy's own
# default for ``scipy.optimize.brentq``, which the Python path does not
# override.
cdef double _BRENTQ_XTOL = 1.0e-12
cdef double _BRENTQ_RTOL = 1.0e-14
cdef int _BRENTQ_MAXITER = 100

# Reported in artifact provenance so a run names the kernels that produced it.
KERNEL_ID = "cython/_cathode_kernels_cy/tierA+csda"


def pemr():
    """Return this module's proton-to-electron mass ratio."""
    return _PEMR


def check_constants(double pemr_py, double erg_per_eV_py, double me_cgs_py):
    """Raise unless ``_cathode_solver``'s constants are bit-identical to ours.

    Called once, at bind time, by ``_cathode_solver``. A mismatch means the
    duplicated literals above have drifted from the authoritative ones, which
    would silently change the physics of every compiled-kernel run -- so it is
    a hard, immediate failure rather than a warning.
    """
    _require("proton-to-electron mass ratio", "_pemr", pemr_py, _PEMR)
    _require("eV->erg conversion", "_erg_per_eV", erg_per_eV_py, _ERG_PER_EV)
    _require("electron mass [g]", "_me_cgs", me_cgs_py, _ME_CGS)


def check_constants_beam(
    double erg_per_eV_py,
    double me_cgs_py,
    double e4_cgs_py,
    double omega_pe_coeff_py,
    double pi_py,
):
    """Raise unless ``_beam_deposition``'s constants are bit-identical to ours.

    Called once, at bind time, by ``_beam_deposition``. Same contract as
    ``check_constants``: a drifted duplicate is a hard failure, never a quiet
    change of physics.
    """
    _require("eV->erg conversion", "_ERG_PER_EV", erg_per_eV_py, _ERG_PER_EV)
    _require("electron mass [g]", "_ME_CGS", me_cgs_py, _ME_CGS)
    _require("e^4 [esu^4]", "_E4_CGS", e4_cgs_py, _E4_CGS)
    _require(
        "omega_pe coefficient", "_OMEGA_PE_COEFF",
        omega_pe_coeff_py, _OMEGA_PE_COEFF,
    )
    _require("pi", "math.pi", pi_py, _PI)


def check_constants_idriven(double schottky_py):
    """Raise unless ``_cathode_solver_idriven``'s constants match ours.

    Separate from ``check_constants`` because the Schottky constant lives in
    the current-driven module, which imports AFTER ``_cathode_solver`` binds.
    """
    _require(
        "Schottky lowering constant [eV/sqrt(V/m)]",
        "_SCHOTTKY_EV_PER_SQRT_V_M",
        schottky_py,
        _SCHOTTKY_EV_PER_SQRT_V_M,
    )


cdef _require(str what, str name, double python_value, double compiled_value):
    if python_value != compiled_value:
        raise ValueError(
            f"compiled cathode kernels disagree with the Python solver on "
            f"the {what}: compiled={compiled_value!r} vs "
            f"python {name}={python_value!r}. The duplicated literals in "
            "_cathode_kernels_cy.pyx have drifted; fix them rather than "
            "relaxing this check."
        )


# ----------------------------------------------------------------------
# Scalar leaves
# ----------------------------------------------------------------------

cdef inline double _min_700(double x) noexcept nogil:
    """``min(x, 700.0)`` with CPython's semantics (NaN propagates)."""
    return 700.0 if 700.0 < x else x


cdef inline double _exp_clamped_c(double x) noexcept nogil:
    """``_cathode_solver._exp_clamped``."""
    return exp(_min_700(x))


cdef double _j_eth_crit_c(double psi, double J_i, double mu) noexcept nogil:
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


def j_eth_crit(double psi, double J_i, double mu):
    """Scaled critical thermionic current J_eth_crit(psi_c_plus).

    Faithful transcription of ``_cathode_solver._j_eth_crit``; see that
    docstring for the physics. Operation order here is identical, statement
    for statement, so any difference in the result is a floating-point
    artifact of the C compilation and not of the formula.
    """
    return _j_eth_crit_c(psi, J_i, mu)


cdef double _c_log_ei_c(double T_e, double n_e) noexcept nogil:
    if T_e > 10.0:
        return 24.0 - log(sqrt(n_e) / T_e)
    return 23.0 - log(sqrt(n_e) * pow(T_e, -1.5))


def c_log_ei(double T_e, double n_e):
    """Electron-ion Coulomb logarithm; ``_cathode_solver._c_log_ei``."""
    return _c_log_ei_c(T_e, n_e)


cdef double _compute_l_b_c(
    double phi_c, double T_e, double n_e, double n_n, double sigma_b
) noexcept nogil:
    cdef double v_beam, tau_ei, l_bi, l_bn
    if phi_c <= 0.0:
        return 0.0
    v_beam = sqrt(2.0 * phi_c * _ERG_PER_EV / _ME_CGS)
    tau_ei = 3.44e5 * pow(T_e, 1.5) / n_e / _c_log_ei_c(T_e, n_e)
    l_bi = v_beam * tau_ei
    if sigma_b > 0.0 and n_n > 0.0:
        l_bn = 1.0 / (sigma_b * n_n)
        return 1.0 / (1.0 / l_bi + 1.0 / l_bn)
    return l_bi


def compute_l_b(
    double phi_c, double T_e, double n_e, double n_n, double sigma_b
):
    """Beam mean free path [cm]; ``_cathode_solver._compute_l_b``."""
    return _compute_l_b_c(phi_c, T_e, n_e, n_n, sigma_b)


cdef double _schottky_lowering_c(
    double phi_c_V, double T_e, double n_e
) noexcept nogil:
    cdef double lambda_D_cm, psi, s_cl_cm, E_V_per_m
    if phi_c_V <= 0.0 or T_e <= 0.0 or n_e <= 0.0:
        return 0.0
    lambda_D_cm = 743.0 * sqrt(T_e / n_e)
    psi = phi_c_V / T_e
    s_cl_cm = (sqrt(2.0) / 3.0) * lambda_D_cm * pow(2.0 * psi, 0.75)
    if s_cl_cm <= 0.0:
        return 0.0
    E_V_per_m = (4.0 / 3.0) * phi_c_V / s_cl_cm * 100.0
    return _SCHOTTKY_EV_PER_SQRT_V_M * sqrt(E_V_per_m)


def schottky_lowering_eV(double phi_c_V, double T_e, double n_e):
    """Work-function lowering [eV];
    ``_cathode_solver_idriven._schottky_lowering_eV``."""
    return _schottky_lowering_c(phi_c_V, T_e, n_e)


# ----------------------------------------------------------------------
# The annular Schottky emission state -- the residual's body
# ----------------------------------------------------------------------

cdef void _annular_schottky_c(
    double psi,
    double J_i,
    double mu,
    const double* J_eth_k,
    const double* delta_k,
    const double* ion_frac_k,
    Py_ssize_t n,
    double T_e,
    double n_e,
    double* J_star_out,
    double* psi_minus_out,
    bint* clamped_out,
) noexcept nogil:
    """``_cathode_solver_idriven._annular_state_schottky``, statement for
    statement."""
    cdef double dphi = _schottky_lowering_c(psi * T_e, T_e, n_e)
    cdef double J_star_total = 0.0
    cdef double weighted_pm = 0.0
    cdef bint any_clamped = 0
    cdef Py_ssize_t a
    cdef double J_eth_a, delta_a, frac_a, J_crit_a, J_eff_a
    for a in range(n):
        J_eth_a = J_eth_k[a]
        delta_a = delta_k[a]
        frac_a = ion_frac_k[a]
        if J_eth_a <= 0.0:
            continue
        J_crit_a = (
            _j_eth_crit_c(psi, J_i * frac_a, mu) if frac_a > 0.0 else 0.0
        )
        if J_eth_a > J_crit_a:
            any_clamped = 1
            if J_crit_a <= 0.0:
                continue
            J_star_total += J_crit_a
            weighted_pm += J_crit_a * delta_a * log(J_eth_a / J_crit_a)
            continue
        J_eff_a = J_eth_a * exp(dphi / (delta_a * T_e))
        if J_eff_a > J_crit_a:
            any_clamped = 1
            J_star_total += J_crit_a
            continue
        J_star_total += J_eff_a
    J_star_out[0] = J_star_total
    psi_minus_out[0] = (
        weighted_pm / J_star_total if J_star_total > 0.0 else 0.0
    )
    clamped_out[0] = any_clamped


cdef double* _to_buffer(object seq, Py_ssize_t n) except NULL:
    """Copy a Python sequence of ``n`` floats into a fresh C buffer."""
    cdef double* buf = <double*> malloc(n * sizeof(double))
    cdef Py_ssize_t i
    if buf == NULL:
        raise MemoryError()
    for i in range(n):
        buf[i] = <double> seq[i]
    return buf


def annular_state_schottky(
    double psi,
    double J_i,
    double mu,
    object J_eth_k,
    object delta_k,
    object ion_frac_k,
    double T_e,
    double n_e,
):
    """``(J_star_total, psi_minus_eff, any_clamped)`` for the annuli.

    Signature-compatible with
    ``_cathode_solver_idriven._annular_state_schottky``: the three per-annulus
    sequences are the caller's tuples and are copied into C buffers here.
    """
    cdef Py_ssize_t n = len(J_eth_k)
    if len(delta_k) != n or len(ion_frac_k) != n:
        raise ValueError(
            "J_eth_k, delta_k and ion_frac_k must have one length "
            f"(got {n}, {len(delta_k)}, {len(ion_frac_k)})"
        )
    cdef double* J_eth_buf = NULL
    cdef double* delta_buf = NULL
    cdef double* frac_buf = NULL
    cdef double J_star = 0.0
    cdef double psi_minus = 0.0
    cdef bint clamped = 0
    try:
        J_eth_buf = _to_buffer(J_eth_k, n)
        delta_buf = _to_buffer(delta_k, n)
        frac_buf = _to_buffer(ion_frac_k, n)
        _annular_schottky_c(
            psi, J_i, mu, J_eth_buf, delta_buf, frac_buf, n, T_e, n_e,
            &J_star, &psi_minus, &clamped,
        )
    finally:
        free(J_eth_buf)
        free(delta_buf)
        free(frac_buf)
    return J_star, psi_minus, bool(clamped)


# ----------------------------------------------------------------------
# The root find
# ----------------------------------------------------------------------

ctypedef struct _root_args:
    double J_i
    double mu
    double Lambda
    double T_e
    double n_e
    double* J_eth_k
    double* delta_k
    double* ion_frac_k
    Py_ssize_t n
    double target


cdef inline double _J_tot_c(double psi, _root_args* p) noexcept nogil:
    cdef double J_star = 0.0
    cdef double psi_minus = 0.0
    cdef bint clamped = 0
    _annular_schottky_c(
        psi, p.J_i, p.mu, p.J_eth_k, p.delta_k, p.ion_frac_k, p.n,
        p.T_e, p.n_e, &J_star, &psi_minus, &clamped,
    )
    return p.J_i * (1.0 - _exp_clamped_c(p.Lambda - psi)) + J_star


cdef inline double _net_phi_c_c(double psi, _root_args* p) noexcept nogil:
    cdef double J_star = 0.0
    cdef double psi_minus = 0.0
    cdef bint clamped = 0
    _annular_schottky_c(
        psi, p.J_i, p.mu, p.J_eth_k, p.delta_k, p.ion_frac_k, p.n,
        p.T_e, p.n_e, &J_star, &psi_minus, &clamped,
    )
    return (psi - psi_minus) * p.T_e


cdef inline double _reported_phi_c_c(double psi, _root_args* p) noexcept nogil:
    """``psi*T_e - psi_minus*T_e``: the caller's own phi_c arithmetic.

    Transcribes ``_cathode_solver_idriven._reported_phi_c``, which differs from
    ``_net_phi_c`` in its last bit and is the form the returned-root ceiling
    test must use (see the comment there).
    """
    cdef double J_star = 0.0
    cdef double psi_minus = 0.0
    cdef bint clamped = 0
    _annular_schottky_c(
        psi, p.J_i, p.mu, p.J_eth_k, p.delta_k, p.ion_frac_k, p.n,
        p.T_e, p.n_e, &J_star, &psi_minus, &clamped,
    )
    return psi * p.T_e - psi_minus * p.T_e


cdef double _f_current(double psi, void* args) noexcept:
    """``_J_tot(psi) - J_target``, the current-matching residual."""
    cdef _root_args* p = <_root_args*> args
    return _J_tot_c(psi, p) - p.target


cdef double _f_net_sheath(double psi, void* args) noexcept:
    """``_net_phi_c(psi) - phi_c_cap_V``, the ceiling residual."""
    cdef _root_args* p = <_root_args*> args
    return _net_phi_c_c(psi, p) - p.target


def solve_psi_annular_schottky(
    double J_i,
    double mu,
    double Lambda,
    double T_e,
    double n_e,
    object J_eth_k,
    object delta_k,
    object ion_frac_k,
    double J_imposed,
    double phi_c_cap_V,
    double psi_lo,
    double psi_top,
    double plateau_tol_rel,
):
    """Locate ``psi_c_plus`` for the annular + Schottky production branch.

    A statement-for-statement transcription of the bracket ladder and root
    find in ``_cathode_solver_idriven.solve_idriven`` (the block from
    ``capability_limited = False`` to the second ``brentq``), for the case
    ``annular and schottky and not bridge``. Returns
    ``(psi_c_plus, capability_limited, brentq_iterations)``; the caller
    re-derives everything else from ``psi_c_plus`` exactly as before.

    ``brentq_iterations`` is -1 when no root find ran (the ceiling was hit
    exactly at the bracket top); it exists so an equivalence check can report
    WHERE a divergence happened rather than only that one did.
    """
    cdef Py_ssize_t n = len(J_eth_k)
    if len(delta_k) != n or len(ion_frac_k) != n:
        raise ValueError(
            "J_eth_k, delta_k and ion_frac_k must have one length "
            f"(got {n}, {len(delta_k)}, {len(ion_frac_k)})"
        )
    cdef _root_args args
    cdef zeros_full_output out
    cdef bint capability_limited = 0
    cdef double J_target = J_imposed
    cdef double psi_c_plus
    cdef int i
    cdef int iterations = -1
    cdef double* buf_eth = NULL
    cdef double* buf_delta = NULL
    cdef double* buf_frac = NULL
    args.J_i = J_i
    args.mu = mu
    args.Lambda = Lambda
    args.T_e = T_e
    args.n_e = n_e
    args.n = n
    args.target = 0.0
    try:
        buf_eth = _to_buffer(J_eth_k, n)
        buf_delta = _to_buffer(delta_k, n)
        buf_frac = _to_buffer(ion_frac_k, n)
        args.J_eth_k = buf_eth
        args.delta_k = buf_delta
        args.ion_frac_k = buf_frac
        for i in range(200):
            if _J_tot_c(psi_top, &args) >= J_target:
                break
            if _net_phi_c_c(psi_top, &args) >= phi_c_cap_V:
                capability_limited = 1
                break
            psi_top *= 2.0
        else:
            capability_limited = 1

        if capability_limited and _J_tot_c(psi_top, &args) >= J_imposed - (
            plateau_tol_rel * fabs(J_imposed)
        ):
            capability_limited = 0
            J_target = J_imposed - plateau_tol_rel * fabs(J_imposed)

        if not capability_limited:
            args.target = J_target
            psi_c_plus = brentq(
                _f_current, psi_lo, psi_top, &args,
                _BRENTQ_XTOL, _BRENTQ_RTOL, _BRENTQ_MAXITER, &out,
            )
            _check_brentq(&out)
            iterations = out.iterations
            # The ceiling is enforced on the located root, not only on the
            # ladder's doubling grid (fix 2026-08-09) -- see the comment on the
            # same test in ``_cathode_solver_idriven.solve_idriven``.
            if _reported_phi_c_c(psi_c_plus, &args) >= phi_c_cap_V:
                capability_limited = 1

        if capability_limited:
            if _net_phi_c_c(psi_top, &args) > phi_c_cap_V:
                args.target = phi_c_cap_V
                psi_c_plus = brentq(
                    _f_net_sheath, psi_lo, psi_top, &args,
                    _BRENTQ_XTOL, _BRENTQ_RTOL, _BRENTQ_MAXITER, &out,
                )
                _check_brentq(&out)
                iterations = out.iterations
            else:
                psi_c_plus = psi_top
                iterations = -1
    finally:
        free(buf_eth)
        free(buf_delta)
        free(buf_frac)
    return psi_c_plus, bool(capability_limited), iterations


cdef _check_brentq(zeros_full_output* out):
    """Turn SciPy's C error codes into the loud failure Python brentq raises.

    ``scipy.optimize.brentq`` raises on a bad bracket or a non-converged
    solve; ``cython_optimize.brentq`` returns NaN and sets ``error_num``
    instead. Propagating that NaN into the sheath solution would be a silent
    wrong answer, so it is raised here.
    """
    if out.error_num == 0:
        return
    raise RuntimeError(
        "compiled cathode brentq failed: error_num="
        f"{out.error_num} (-1 sign error, -2 convergence error), "
        f"funcalls={out.funcalls}, iterations={out.iterations}, "
        f"root={out.root!r}"
    )


# ======================================================================
# Beam deposition: the CSDA substep march
# ======================================================================
#
# Kernel 1 of the 2026-08-02 cost read: ``deposit_beam``'s per-cell loop, its
# substep loop, the three cross-section table interpolations, the four scalar
# physics leaves and the eight per-cell banks, inside ONE compiled boundary.
# What that boundary deletes is not arithmetic -- it is ~873 Python calls and
# ~5 numpy SCALAR dispatches per substep, which the read measured at ~61 % of
# a 5.70 us substep.
#
# Everything below is a statement-for-statement transcription of
# ``_beam_deposition`` and ``_cross``, under the faithfulness rules in this
# module's docstring. The two places where "faithful" needed a measurement
# rather than a reading are documented on ``_interp_scalar`` (numpy's fused
# lerp) and ``_pydiv`` (CPython's ZeroDivisionError).


# Placeholders for the optional buffers, so a closure that is OFF leaves its
# memoryview slot pointing at something real rather than at NULL. Never read.
cdef const double[:] _DUMMY_IN = np.zeros(1)
cdef double[::1] _DUMMY_OUT = np.zeros(1)


cdef inline double _pydiv(double a, double b) except? -1.0:
    """``a / b`` with CPython's ZeroDivisionError rather than C's inf.

    ``cdivision=True`` (this module's directive, and the right one for the
    hot path) makes ``x / 0.0`` return an infinity the way C does, where the
    Python source would raise. Used ONLY at the handful of sites whose
    denominator is not already proved non-zero by a guard the transcription
    keeps, so a misconfigured call fails the same way on both paths.
    """
    if b == 0.0:
        raise ZeroDivisionError("float division by zero")
    return a / b


cdef double _interp_scalar(
    double x,
    const double* xp,
    const double* fp,
    Py_ssize_t n,
    double lval,
    double rval,
) noexcept:
    """``numpy.interp(x, xp, fp, left=lval, right=rval)`` for a scalar ``x``.

    A transcription of numpy 2.4.2's ``arr_interp``
    (``multiarray/compiled_base.c``): the ``binary_search_with_guess``
    convention (largest ``j`` with ``xp[j] <= x``; ``-1`` below the table,
    ``n`` above it), the endpoint and exact-node short circuits, and the
    NaN fallback ladder.

    **The lerp is a FUSED multiply-add, and that is a measurement, not a
    reading.** numpy's C is compiled with FP contraction enabled on this
    platform, so ``slope*(x - xp[j]) + fp[j]`` rounds ONCE, not twice. The
    unfused form disagrees with ``np.interp`` by 1 ULP on ~2.5e-4 of queries;
    the fused form is bit-identical on 1 235 520 queries across all three
    tables this module interpolates (exact nodes, both ``nextafter``
    neighbours of every node, out-of-range on both sides, and NaN included).
    ``build_ext.py`` passes ``-ffp-contract=off``, which is what keeps every
    OTHER expression in this file unfused -- so the fusion here has to be
    written out, and only here.

    ``n == 1`` is NOT transcribed, and the general path below is not its
    limit. numpy runs a separate single-node loop with NO NaN short-circuit:
    both of its comparisons are false for a NaN query, so it falls through and
    returns ``fp[0]``, where the path below propagates the NaN. For a FINITE
    ``x`` the two do agree on the three-way ``lval`` / ``rval`` / ``fp[0]``
    answer. The tables this module interpolates all have many nodes, so the
    shape does not arise here; ``_interp.py``'s ``interp_scalar_fused``, which
    is the general-purpose entry point, writes the single-node rule out.
    """
    cdef Py_ssize_t imin, imax, imid, j
    cdef double slope, res
    if isnan(x):
        return x
    if x > xp[n - 1]:
        return rval
    if x < xp[0]:
        return lval
    imin = 0
    imax = n
    while imin < imax:
        imid = imin + ((imax - imin) >> 1)
        if x >= xp[imid]:
            imin = imid + 1
        else:
            imax = imid
    j = imin - 1
    if j == n - 1:
        return fp[j]
    if xp[j] == x:
        # numpy's "avoid potential non-finite interpolation" short circuit.
        return fp[j]
    slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
    res = fma(slope, x - xp[j], fp[j])
    if isnan(res):
        res = fma(slope, x - xp[j + 1], fp[j + 1])
        if isnan(res) and fp[j] == fp[j + 1]:
            res = fp[j]
    return res


cdef class CsdaTables:
    """The three read-only cross-section tables the march interpolates.

    Bound ONCE by ``_beam_deposition`` (see ``_csda_tables`` there) so the
    march does not re-validate five numpy arrays on every call. The
    memoryviews are kept as attributes purely to own the buffers the raw
    pointers alias; ``double[::1]`` also enforces the C-contiguity the
    pointer arithmetic assumes.

    ``exc_top`` / ``exc_bottom`` are the excitation table's span, exposed
    because the Python side gates on them: above ``exc_top``
    ``He_beam_excitation_channel_lkup`` falls back to the exact manifold sum,
    which is not transcribed here, so such a ray takes the pure-Python march.
    """

    cdef const double[::1] _eii_x_mv
    cdef const double[::1] _eii_y_mv
    cdef const double[::1] _exc_x_mv
    cdef const double[::1] _exc_s_mv
    cdef const double[::1] _exc_se_mv
    cdef const double* eii_x
    cdef const double* eii_y
    cdef const double* exc_x
    cdef const double* exc_s
    cdef const double* exc_se
    cdef Py_ssize_t eii_n
    cdef Py_ssize_t exc_n
    cdef double eii_lval
    cdef double eii_rval
    cdef double exc_s_lval
    cdef double exc_s_rval
    cdef double exc_se_lval
    cdef double exc_se_rval
    cdef readonly double exc_top
    cdef readonly double exc_bottom

    def __cinit__(self, eii_log_eps, eii_log_sigma, exc_E, exc_sigma, exc_sigma_E):
        self._eii_x_mv = eii_log_eps
        self._eii_y_mv = eii_log_sigma
        self._exc_x_mv = exc_E
        self._exc_s_mv = exc_sigma
        self._exc_se_mv = exc_sigma_E
        self.eii_n = self._eii_x_mv.shape[0]
        self.exc_n = self._exc_x_mv.shape[0]
        if (
            self._eii_y_mv.shape[0] != self.eii_n
            or self._exc_s_mv.shape[0] != self.exc_n
            or self._exc_se_mv.shape[0] != self.exc_n
            or self.eii_n < 1
            or self.exc_n < 1
        ):
            raise ValueError("CsdaTables: table lengths disagree or are empty")
        self.eii_x = &self._eii_x_mv[0]
        self.eii_y = &self._eii_y_mv[0]
        self.exc_x = &self._exc_x_mv[0]
        self.exc_s = &self._exc_s_mv[0]
        self.exc_se = &self._exc_se_mv[0]
        # ``He_EII_cross_lkup`` passes left/right explicitly; the excitation
        # lookup does not, so numpy's defaults (the endpoint values) apply.
        self.eii_lval = self.eii_y[0]
        self.eii_rval = self.eii_y[self.eii_n - 1]
        self.exc_s_lval = self.exc_s[0]
        self.exc_s_rval = self.exc_s[self.exc_n - 1]
        self.exc_se_lval = self.exc_se[0]
        self.exc_se_rval = self.exc_se[self.exc_n - 1]
        self.exc_bottom = self.exc_x[0]
        self.exc_top = self.exc_x[self.exc_n - 1]


cdef inline double _he_eii_cross_lkup_c(double eps, CsdaTables t) noexcept:
    """``_cross.He_EII_cross_lkup``."""
    return exp(
        _interp_scalar(
            log(eps), t.eii_x, t.eii_y, t.eii_n, t.eii_lval, t.eii_rval
        )
    )


cdef inline void _he_beam_exc_lkup_c(
    double E, CsdaTables t, double* sigma_out, double* erad_out
) noexcept:
    """``_cross.He_beam_excitation_channel_lkup``.

    The ``E >= E_grid[-1]`` exact-function fallback is NOT transcribed; the
    Python dispatcher refuses the compiled march for any ray that could reach
    it (``E`` only ever decreases, so ``E0 < exc_top`` settles it for the
    whole march).
    """
    cdef double s
    if E <= t.exc_bottom:
        sigma_out[0] = 0.0
        erad_out[0] = 0.0
        return
    s = _interp_scalar(
        E, t.exc_x, t.exc_s, t.exc_n, t.exc_s_lval, t.exc_s_rval
    )
    if s <= 0.0:
        sigma_out[0] = 0.0
        erad_out[0] = 0.0
        return
    sigma_out[0] = s
    erad_out[0] = _interp_scalar(
        E, t.exc_x, t.exc_se, t.exc_n, t.exc_se_lval, t.exc_se_rval
    ) / s


cdef inline double _beam_speed_c(double E_eV) noexcept:
    """``_beam_deposition.beam_speed_cm_s``."""
    return sqrt(2.0 * E_eV * _ERG_PER_EV / _ME_CGS)


cdef inline double _he_mean_secondary_energy_c(
    double E_eV, double I_ion_eV, double ebar_eV
) noexcept:
    """``_beam_deposition.he_mean_secondary_energy_eV``."""
    cdef double W_max = 0.5 * (E_eV - I_ion_eV)
    cdef double x
    if W_max <= 0.0:
        return 0.0
    x = W_max / ebar_eV
    return ebar_eV * log1p(x * x) / (2.0 * atan(x))


cdef inline double _coulomb_stopping_c(
    double E_eV, double ne, double Te, int model
) except? -1.0:
    """``_beam_deposition.coulomb_stopping_eV_per_cm``.

    ``max(Te, 0.1)`` is transcribed as CPython's conditional (``b if b > a
    else a``), not ``fmax``, so a NaN temperature propagates the same way.
    """
    cdef double lnL, tau_ei
    cdef double Te_clamped
    if ne <= 0.0 or E_eV <= 0.0:
        return 0.0
    Te_clamped = 0.1 if 0.1 > Te else Te
    lnL = _c_log_ei_c(Te_clamped, ne)
    if model == _COULOMB_FAST_ELECTRON:
        return (
            2.0 * _PI * _E4_CGS * ne * lnL / (E_eV * _ERG_PER_EV) / _ERG_PER_EV
        )
    # legacy_tau_ei
    tau_ei = _pydiv(3.44e5 * pow(Te, 1.5) / ne, lnL)
    return _pydiv(E_eV, _beam_speed_c(E_eV) * tau_ei)


cdef inline double _ql_relaxation_length_c(
    double E_eV, double ne, double n_b
) noexcept:
    """``_beam_deposition.quasilinear_relaxation_length_cm``."""
    cdef double omega_pe, ratio
    if ne <= 0.0 or n_b <= 0.0 or n_b >= 0.1 * ne:
        return INFINITY
    omega_pe = _OMEGA_PE_COEFF * sqrt(ne)
    ratio = ne / n_b
    return ratio * (_beam_speed_c(E_eV) / omega_pe) * log(ratio)


def csda_march(
    CsdaTables tables,
    double E0_eV,
    double Gamma0_per_s,
    const double[:] nn,
    const double[:] ne,
    const double[:] Te,
    const double[:] dz_cm,
    Py_ssize_t launch,
    int direction,
    double I_ion_eV,
    double E_stop_eV,
    double frac,
    double ebar_eV,
    int coulomb_code,
    bint anomalous_quasilinear,
    object area,
    Py_ssize_t anode_cross_index,
    double anode_eta,
    bint walk_products,
    bint walk_tail,
    double[::1] ionization_events,
    double[::1] excitation_events,
    double[::1] heating,
    double[::1] radiated,
    double[::1] ionization_cost,
    double[::1] E_entry,
    double[::1] heat_coulomb,
    double[::1] heat_anomalous,
    double[::1] heat_secondary,
    double[::1] heat_terminal,
    object sec_flux,
    object sec_power_eV,
    object anom_power_eV,
):
    """The CSDA march of ``deposit_beam``, statement for statement.

    Writes the ten per-cell banks (and, when their closure is on, the three
    WP-D/WP-E withholding banks) in place with ``+=``, exactly as the Python
    does, and returns the marching state the Python epilogue still needs::

        (E, gamma, absorbed, anode_intercepted,
         terminal_cell, terminal_flux, terminal_E)

    ``anode_cross_index`` is ``-1`` when interception is off (the Python's
    ``intercept_active`` gate folded into the index, which is safe because a
    real index is always >= 0). ``coulomb_code`` is 0 ``fast_electron`` /
    1 ``legacy_tau_ei``. The three withholding banks may be ``None`` when
    their closure is off; they are never touched then.

    The caller is responsible for the preconditions this kernel does not
    re-check: ``0 <= launch < cells``, ``I_ion_eV != 0``, a recognised
    coulomb model, and ``E0_eV < tables.exc_top``. Each of those is a case
    where the Python path does something this transcription deliberately does
    not reproduce (a negative-index walk, a ZeroDivisionError, a ValueError,
    the exact-manifold fallback), so the dispatcher routes them to Python.
    """
    cdef Py_ssize_t cells = dz_cm.shape[0]
    cdef const double[:] area_v = _DUMMY_IN
    cdef double[::1] sec_flux_v = _DUMMY_OUT
    cdef double[::1] sec_power_v = _DUMMY_OUT
    cdef double[::1] anom_power_v = _DUMMY_OUT
    if anomalous_quasilinear:
        area_v = area
    if walk_products:
        sec_flux_v = sec_flux
        sec_power_v = sec_power_eV
    if walk_tail:
        anom_power_v = anom_power_eV

    cdef double E = E0_eV
    cdef double gamma = Gamma0_per_s
    cdef double anode_intercepted = 0.0
    cdef bint absorbed = 0
    cdef bint intercept_active = anode_cross_index >= 0 and anode_eta > 0.0
    cdef Py_ssize_t terminal_cell = -1
    cdef double terminal_flux = 0.0
    cdef double terminal_E = 0.0

    cdef Py_ssize_t cell
    cdef int step = 1 if direction > 0 else -1
    cdef double remaining, nn_c, ne_c, Te_c
    cdef double sigma_i, sigma_x, E_rad, W_sec
    cdef double L_pot, L_sec, L_exc, L_coul, L_anom, L_tot
    cdef double n_b, l_ql, dz_sub, cand
    cdef double d_pot, d_sec, d_exc, d_coul, d_anom, d_anom_local
    cdef double acc_ionization_events, acc_excitation_events, acc_heating
    cdef double acc_radiated, acc_ionization_cost
    cdef double acc_heat_coulomb, acc_heat_anomalous
    cdef double acc_heat_secondary, acc_heat_terminal
    cdef double acc_sec_flux, acc_sec_power_eV, acc_anom_power_eV

    cell = launch
    while 0 <= cell < cells:
        if intercept_active and cell == anode_cross_index:
            anode_intercepted = (
                anode_intercepted + anode_eta * gamma * E * _ERG_PER_EV
            )
            gamma = gamma * (1.0 - anode_eta)
            intercept_active = 0
        E_entry[cell] = E
        remaining = dz_cm[cell]
        nn_c = nn[cell]
        ne_c = ne[cell]
        Te_c = Te[cell]
        acc_ionization_events = 0.0
        acc_excitation_events = 0.0
        acc_heating = 0.0
        acc_radiated = 0.0
        acc_ionization_cost = 0.0
        acc_heat_coulomb = 0.0
        acc_heat_anomalous = 0.0
        acc_heat_secondary = 0.0
        acc_heat_terminal = 0.0
        acc_sec_flux = 0.0
        acc_sec_power_eV = 0.0
        acc_anom_power_eV = 0.0
        while remaining > 0.0:
            sigma_i = (
                _he_eii_cross_lkup_c(E / I_ion_eV, tables)
                if E > I_ion_eV
                else 0.0
            )
            _he_beam_exc_lkup_c(E, tables, &sigma_x, &E_rad)
            W_sec = _he_mean_secondary_energy_c(E, I_ion_eV, ebar_eV)
            L_pot = nn_c * sigma_i * I_ion_eV
            L_sec = nn_c * sigma_i * W_sec
            L_exc = nn_c * sigma_x * E_rad
            L_coul = _coulomb_stopping_c(E, ne_c, Te_c, coulomb_code)
            L_anom = 0.0
            if anomalous_quasilinear:
                n_b = _pydiv(gamma, area_v[cell] * _beam_speed_c(E))
                l_ql = _ql_relaxation_length_c(E, ne_c, n_b)
                if isfinite(l_ql) and l_ql > 0.0:
                    L_anom = E / l_ql
            L_tot = L_pot + L_sec + L_exc + L_coul + L_anom
            if L_tot <= 0.0:
                break  # vacuum cell: free streaming
            # ``min(remaining, frac * E / L_tot)`` with CPython's semantics.
            cand = frac * E / L_tot
            dz_sub = cand if cand < remaining else remaining
            if E - L_tot * dz_sub <= E_stop_eV:
                dz_sub = (E - E_stop_eV) / L_tot
            if dz_sub <= 0.0:
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating = acc_heating + gamma * E * _ERG_PER_EV
                    acc_heat_terminal = (
                        acc_heat_terminal + gamma * E * _ERG_PER_EV
                    )
                E = 0.0
                absorbed = 1
                break
            d_pot = L_pot * dz_sub
            d_sec = L_sec * dz_sub
            d_exc = L_exc * dz_sub
            d_coul = L_coul * dz_sub
            d_anom = L_anom * dz_sub
            acc_ionization_cost = (
                acc_ionization_cost + gamma * d_pot * _ERG_PER_EV
            )
            if walk_tail:
                acc_anom_power_eV = acc_anom_power_eV + gamma * d_anom
                d_anom_local = 0.0
            else:
                d_anom_local = d_anom
            if walk_products:
                acc_heating = (
                    acc_heating + gamma * (d_coul + d_anom_local) * _ERG_PER_EV
                )
                acc_sec_flux = acc_sec_flux + gamma * nn_c * sigma_i * dz_sub
                acc_sec_power_eV = acc_sec_power_eV + gamma * d_sec
            else:
                acc_heating = (
                    acc_heating
                    + gamma * (d_sec + d_coul + d_anom_local) * _ERG_PER_EV
                )
                acc_heat_secondary = (
                    acc_heat_secondary + gamma * d_sec * _ERG_PER_EV
                )
            acc_heat_coulomb = acc_heat_coulomb + gamma * d_coul * _ERG_PER_EV
            acc_heat_anomalous = (
                acc_heat_anomalous + gamma * d_anom_local * _ERG_PER_EV
            )
            acc_radiated = acc_radiated + gamma * d_exc * _ERG_PER_EV
            acc_ionization_events = (
                acc_ionization_events + gamma * nn_c * sigma_i * dz_sub
            )
            acc_excitation_events = (
                acc_excitation_events + gamma * nn_c * sigma_x * dz_sub
            )
            E = E - (d_pot + d_sec + d_exc + d_coul + d_anom)
            remaining = remaining - dz_sub
            if E <= E_stop_eV:
                if walk_products:
                    terminal_cell = cell
                    terminal_flux = gamma
                    terminal_E = E
                else:
                    acc_heating = acc_heating + gamma * E * _ERG_PER_EV
                    acc_heat_terminal = (
                        acc_heat_terminal + gamma * E * _ERG_PER_EV
                    )
                E = 0.0
                absorbed = 1
                break
        ionization_events[cell] += acc_ionization_events
        excitation_events[cell] += acc_excitation_events
        heating[cell] += acc_heating
        radiated[cell] += acc_radiated
        ionization_cost[cell] += acc_ionization_cost
        heat_coulomb[cell] += acc_heat_coulomb
        heat_anomalous[cell] += acc_heat_anomalous
        heat_secondary[cell] += acc_heat_secondary
        heat_terminal[cell] += acc_heat_terminal
        if walk_products:
            sec_flux_v[cell] += acc_sec_flux
            sec_power_v[cell] += acc_sec_power_eV
        if walk_tail:
            anom_power_v[cell] += acc_anom_power_eV
        if absorbed:
            break
        cell += step

    return (
        E,
        gamma,
        bool(absorbed),
        anode_intercepted,
        terminal_cell,
        terminal_flux,
        terminal_E,
    )
