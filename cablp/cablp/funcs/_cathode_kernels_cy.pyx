# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""Compiled scalar kernels for the cathode sheath solver (Tier A).

Scope
-----
The coherent scalar unit the production (current-driven) cathode solve
spends its time in. The cathode solve is ~31 % of a production run and is
call-overhead bound (2026-07-30 profile), so the question this module
exists to answer is narrow and empirical:

    does a libc.math transcription with UNCHANGED operation order reproduce
    the golden bit-exactly?

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
  time (``check_constants``/``check_constants_idriven``), so a divergent
  literal is a loud failure and never a quiet change of physics.

Nothing RUNS this module's kernels unless ``CABLP_COMPILED_KERNELS`` opts
in; see ``cablp.funcs._kernels``. (``smoke_sim1d`` imports it directly
regardless, to compare it against the pure kernels -- comparing is not
running.)
"""

from libc.math cimport exp, fabs, log, sqrt, pow
from libc.stdlib cimport malloc, free

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

# brentq controls, matching the Python call sites verbatim
# (``_cathode_solver_idriven.solve_idriven``). ``maxiter`` is SciPy's own
# default for ``scipy.optimize.brentq``, which the Python path does not
# override.
cdef double _BRENTQ_XTOL = 1.0e-12
cdef double _BRENTQ_RTOL = 1.0e-14
cdef int _BRENTQ_MAXITER = 100

# Reported in artifact provenance so a run names the kernels that produced it.
KERNEL_ID = "cython/_cathode_kernels_cy/tierA"


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
    """Locate ``psi_c_plus`` for the annular + Schottty production branch.

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
        else:
            args.target = J_target
            psi_c_plus = brentq(
                _f_current, psi_lo, psi_top, &args,
                _BRENTQ_XTOL, _BRENTQ_RTOL, _BRENTQ_MAXITER, &out,
            )
            _check_brentq(&out)
            iterations = out.iterations
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
