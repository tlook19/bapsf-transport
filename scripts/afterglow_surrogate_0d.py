"""Zero-dimensional two-temperature afterglow surrogate for the ES1 ports.

Discriminator D1 of the decay-lead registration. For each ES1 port the script
initialises a single-cell (0D) plasma from a saved LAPDSim1D trajectory at
beam shutoff and integrates the model's own afterglow operators forward over
the campaign's stage-(iii) decay window, then fits the electron-saturation-
current proxy the same way ``compare_sim1d_es1`` fits it.

State and units (CGS throughout, temperatures in eV where noted)
---------------------------------------------------------------
``n``   plasma density [cm^-3]; ``Ee``, ``Ei`` conservative electron and ion
energy densities [erg cm^-3] with ``E = 1.5 n T e``. Temperatures are derived
as ``T = (2/3) E / (n_safe * ev_to_erg)`` and floored at the run's ``Te_floor``
/ ``Ti_floor``, mirroring ``core.state.derive_state``. The neutral background
(``nn``, ``Tn``) is FROZEN at its shutoff value: this is a frozen-background
instrument, not a coupled neutral solve.

Channels (each transcribed from the operator named in parentheses)
-----------------------------------------------------------------
* electron-ion exchange ``Q_ie`` (``plasma.heat.Q_ie`` as
  ``physics.energy.electron_ion_exchange_rhs`` books it): positive when
  ``Te > Ti``; sink on ``Ee``, source on ``Ei``.
* conduction-limited end loss (``plasma.heat.elec_par_heat_loss`` /
  ``ion_par_heat_loss``): ``kappa_par * T / (L_p * L_hf)``, a sink on both
  energies, scaled by ``end_loss_coefficient`` (1.0 = unmodified).
* ion-neutral (charge-exchange) energy sink, the thermal channel of
  ``physics.sources.ion_neutral_collision_rhs``:
  ``1.5 * n * nu_mt * (Tn - Ti) * e`` with
  ``nu_mt = nn * phelps_momentum_transfer_rate_cm3_s((Ti + Tn) / 2)``.
  Negative on ``Ei`` while ``Ti > Tn`` -- the ion is lost at ``Ti`` and
  replaced at ``Tn``. The drift-frictional half of that operator is absent: a
  0D cell carries no relative velocity.
* sound-speed particle loss: ``dn/dt = -n * alpha_isat * c_s / L_col`` with
  ``c_s`` from ``physics.flux.plasma_wave_speed`` under the run's own
  ``hyperbolic_wave_speed`` convention. Particles leave carrying the local
  per-particle enthalpy ``2.5 T``, so the channel cools as well as drains.

Closure lengths are read from the run, never fitted: ``L_p`` is the axial
distance from the port cell to the nearest plasma-terminating cell, ``L_hf``
is the model's own axial temperature scale length ``Te / |dTe/dz|`` at the
shutoff sample (``--closure gradient``, the default) or ``L_p`` again
(``--closure geometric``), and ``L_col`` is half the plasma-active axial
extent.

Observable and fit
------------------
``Isat = n * sqrt(max(Te, 0))``, fitted log-linearly over the campaign's
stage-(iii) window on the saved trajectory's own sample times, using
``compare_sim1d_es1._efold_time_ms`` itself.

Raises
------
``ValueError`` for a malformed or unusable input: an HDF5 file missing a
required dataset, a run with no ``main_discharge`` phase, a trajectory that
does not span the decay window, a port absent from the overlay, a
non-positive closure length, or a non-finite initial state.
``SystemExit(1)`` when the pre-registered instrument gate fails; the gate
table is printed first and no variation is run.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_sim1d_es1 import DECAY_WINDOW_MS, _efold_time_ms  # noqa: E402

from cablp.atomic.cross_sections import (  # noqa: E402
    phelps_momentum_transfer_rate_cm3_s,
)
from cablp.plasma.heat import (  # noqa: E402
    Q_ie,
    _resolve_per_particle,
    elec_par_heat_loss,
    kappa_par_ion,
)
from cablp.plasma.params import c_log  # noqa: E402
from cablp.solvers._sim1d.physics.flux import plasma_wave_speed  # noqa: E402
from cablp.constants import ev_to_erg, m_He_cgs, m_p_cgs  # noqa: E402

# The R3b-deleted ``plasma.heat.ion_par_heat_loss``, transcribed here verbatim
# because this instrument is frozen as-run.
def _ion_par_heat_loss(Ti, ni, L_p, L_hf, mu, lnlambda, per_particle=True, *, rk=None):
    """
    Ion parallel heat loss rate per unit volume [eV/s] or [eV·cm⁻³/s].

    Q = kappa_par_ion * Ti / L_p / L_hf

    Parameters
    ----------
    Ti : float or array
        Ion temperature [eV].
    ni : float or array
        Ion density [cm⁻³].
    L_p : float or array
        Plasma half-length [cm].
    L_hf : float or array
        Heat-flux scale length [cm].
    mu : float
        Ion mass number.
    lnlambda : float or array
        Coulomb logarithm.
    per_particle : bool
        Passed through to kappa_par_ion.

    Returns
    -------
    float or array
        Ion parallel heat loss [eV/s or eV·cm⁻³/s].
    """
    per_particle = _resolve_per_particle(per_particle, rk)
    return kappa_par_ion(Ti, ni, mu, lnlambda, per_particle=per_particle) * Ti / L_p / L_hf


SCRIPTS = Path(__file__).resolve().parent

# ES1 ports of the decay-lead registration, in axial order.
PORTS = (11, 21, 29, 41, 50)

# Pre-registered instrument gate: the unmodified surrogate must reproduce each
# port's MODEL e-fold time this closely, or the run stops here.
GATE_REL_TOL = 0.15

# Measured/model e-fold ratios of record, plotnotes_decay_starting_temperatures.txt
# lines 28-32 (column "tau ratio"). The measured e-fold follows as
# tau_model / ratio; the script cross-checks that against the overlay refit.
NOTE_TAU_RATIO = {11: 1.48, 21: 1.79, 29: 1.71, 41: 1.56, 50: 1.59}

# V1 replacement Te(0) [eV]. p11/p21 are the raw-masked corrections of record;
# p29/p41/p50 are the plot note's face values (lines 30-32) and carry the
# unquantified overlay residual flagged in CAVEATS_V1.
V1_TE0_EV = {11: 8.93, 21: 8.37, 29: 6.98, 41: 5.34, 50: 3.49}
CAVEATS_V1 = (
    "V1 CAVEAT: p11 (8.93 eV) and p21 (8.37 eV) are raw-masked measured Te(0), "
    "the corrections of record for the quantified ES1 overlay defect. p29 "
    "(6.98), p41 (5.34) and p50 (3.49) are the plot note's FACE VALUES and "
    "carry an unquantified residual bias until the overlay rebuild lands; "
    "their recovery fractions inherit that bias."
)

# Fixed RK4 sub-step [s]. The fastest surrogate channel at shutoff runs on
# ~3e-4 s, so this is ~3e3 steps per e-fold.
SUB_DT_S = 1.0e-7

# Energy carried out per lost particle, in units of that species' temperature.
# 2.5 is the convective enthalpy (internal 1.5 T plus the pV work), matching
# the pair of terms the solver books separately as ``plasma_advective_flux``
# and ``pressure_work``; 1.5 would drain energy in proportion to density and
# leave the temperature untouched.
ENTHALPY_FACTOR = 2.5


def _gas_constants(gas_type):
    """Return ``(ion_mass_g, mu)`` for a supported gas, as the solver sets them."""
    if gas_type == "He":
        return m_He_cgs, 4
    if gas_type == "H":
        return m_p_cgs, 1
    raise ValueError(f"unsupported gas_type {gas_type!r}; expected 'He' or 'H'")


def _require(group, name, where):
    if name not in group:
        raise ValueError(f"{where} is missing the required dataset {name!r}")
    return group[name]


class PortInitialState:
    """Frozen shutoff state and closure lengths for one port."""

    def __init__(self, port, z_cm, iz, n, Te, Ti, nn, Tn, L_p, L_hf, L_col):
        for label, value in (
            ("n", n), ("Te", Te), ("Ti", Ti), ("nn", nn), ("Tn", Tn),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"port {port}: shutoff {label} = {value!r} is not a finite "
                    f"positive value; the surrogate cannot be initialised"
                )
        for label, value in (("L_p", L_p), ("L_hf", L_hf), ("L_col", L_col)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"port {port}: closure length {label} = {value!r} is not a "
                    f"finite positive length [cm]"
                )
        self.port = port
        self.z_cm = z_cm
        self.iz = iz
        self.n = n
        self.Te = Te
        self.Ti = Ti
        self.nn = nn
        self.Tn = Tn
        self.L_p = L_p
        self.L_hf = L_hf
        self.L_col = L_col


class RunContext:
    """Everything the surrogate needs from one saved trajectory."""

    def __init__(self, h5_path, overlay_path, window_ms, closure):
        self.h5_path = Path(h5_path)
        self.overlay_path = Path(overlay_path)
        self.window_ms = (float(window_ms[0]), float(window_ms[1]))
        self.closure = closure
        if closure not in ("gradient", "geometric"):
            raise ValueError(
                f"closure must be 'gradient' or 'geometric' (got {closure!r})"
            )
        if not self.h5_path.is_file():
            raise ValueError(f"no such run artifact: {self.h5_path}")
        if not self.overlay_path.is_file():
            raise ValueError(f"no such overlay: {self.overlay_path}")

        overlay = np.load(self.overlay_path, allow_pickle=False)
        for key in ("port", "z_cm", "isat_decay_port", "isat_decay_time_ms",
                    "isat_decay_mean_a"):
            if key not in overlay:
                raise ValueError(
                    f"{self.overlay_path.name} is missing the required array "
                    f"{key!r}"
                )
        self.port_z = {
            int(p): float(z)
            for p, z in zip(overlay["port"], overlay["z_cm"])
        }
        self.overlay = overlay

        with h5py.File(self.h5_path, "r") as f:
            where = self.h5_path.name
            params = _read_json_attr(f, "params_json", where)
            self.params = params
            time_s = np.asarray(_require(f, "time", where)[:], dtype=float)
            phase = np.asarray(_require(f, "phase", where)[:], dtype=str)
            hits = np.flatnonzero(phase == "main_discharge")
            if not hits.size:
                raise ValueError(
                    f"{where} never entered the 'main_discharge' phase, so it "
                    f"has no discharge origin and cannot be used"
                )
            origin = float(time_s[hits[0]])
            self.t_ms = (time_s - origin) * 1.0e3
            geom = _require(f, "geometry", where)
            self.z_cm = np.asarray(_require(geom, "z_cm", where)[:], dtype=float)
            active = np.asarray(
                _require(geom, "plasma_active", where)[:], dtype=bool
            )
            self.Te_all = _require(f, "Te", where)
            self.n_all = _require(f, "n", where)
            i0 = int(np.argmin(np.abs(self.t_ms - self.window_ms[0])))
            self.i0 = i0
            self.t0_ms = float(self.t_ms[i0])
            Te0 = np.asarray(self.Te_all[i0], dtype=float)
            self.model_state = {
                "n": np.asarray(_require(f, "n", where)[i0], dtype=float),
                "Te": Te0,
                "Ti": np.asarray(_require(f, "Ti", where)[i0], dtype=float),
                "nn": np.asarray(_require(f, "nn", where)[i0], dtype=float),
                "Tn": np.asarray(_require(f, "Tn", where)[i0], dtype=float),
            }
            mask = (self.t_ms >= self.window_ms[0]) & (
                self.t_ms <= self.window_ms[1]
            )
            self.window_idx = np.flatnonzero(mask)
            if self.window_idx.size < 8:
                raise ValueError(
                    f"{where} carries only {self.window_idx.size} samples in the "
                    f"decay window {self.window_ms} ms; the log-linear fit needs "
                    f"at least 8"
                )
            if float(self.t_ms.max()) < self.window_ms[1]:
                raise ValueError(
                    f"{where} ends at {float(self.t_ms.max()):.4g} ms on the "
                    f"main-discharge clock, before the decay window closes at "
                    f"{self.window_ms[1]:.4g} ms"
                )
            self.model_n_win = np.asarray(
                self.n_all[self.window_idx, :], dtype=float
            )
            self.model_Te_win = np.asarray(
                self.Te_all[self.window_idx, :], dtype=float
            )
            total = _require(f, "total_rhs", where)
            self.model_rhs0 = {
                key: np.asarray(_require(total, key, where)[i0], dtype=float)
                for key in ("n", "Ee", "Ei")
            }

        self.t_win_ms = self.t_ms[self.window_idx]
        active_idx = np.flatnonzero(active)
        if active_idx.size < 2:
            raise ValueError(
                f"{self.h5_path.name} has fewer than two plasma-active cells"
            )
        self.z_end_lo = float(self.z_cm[active_idx[0]])
        self.z_end_hi = float(self.z_cm[active_idx[-1]])
        self.L_col = 0.5 * (self.z_end_hi - self.z_end_lo)

        self.gas_type = str(self.params.get("gas_type", "He"))
        self.ion_mass_g, self.mu = _gas_constants(self.gas_type)
        self.Te_floor = float(self.params["Te_floor"])
        self.Ti_floor = float(self.params["Ti_floor"])
        self.ne_floor = float(self.params["ne_floor"])
        self.ln_lambda_min = float(self.params.get("ln_lambda_min", 1.0))
        self.alpha_isat = float(self.params["alpha_isat"])
        self.wave_speed = str(
            self.params.get("hyperbolic_wave_speed", "isothermal")
        )
        self.b_drag = float(self.params.get("b_ion_neutral_drag", 1.0))

        grad_Te = np.gradient(self.model_state["Te"], self.z_cm)
        self.ports = {}
        for port in PORTS:
            if port not in self.port_z:
                raise ValueError(
                    f"port {port} is absent from {self.overlay_path.name}"
                )
            z_port = self.port_z[port]
            iz = int(np.argmin(np.abs(self.z_cm - z_port)))
            L_p = min(
                self.z_cm[iz] - self.z_end_lo, self.z_end_hi - self.z_cm[iz]
            )
            if closure == "gradient":
                g = abs(float(grad_Te[iz]))
                if g <= 0.0:
                    raise ValueError(
                        f"port {port}: the axial Te gradient vanishes at the "
                        f"shutoff sample, so the heat-flux scale length is "
                        f"undefined under --closure gradient"
                    )
                L_hf = float(self.model_state["Te"][iz]) / g
            else:
                L_hf = float(L_p)
            self.ports[port] = PortInitialState(
                port=port,
                z_cm=float(self.z_cm[iz]),
                iz=iz,
                n=float(self.model_state["n"][iz]),
                Te=float(self.model_state["Te"][iz]),
                Ti=float(self.model_state["Ti"][iz]),
                nn=float(self.model_state["nn"][iz]),
                Tn=float(self.model_state["Tn"][iz]),
                L_p=float(L_p),
                L_hf=float(L_hf),
                L_col=float(self.L_col),
            )

    def model_tau_ms(self, port):
        """Return the scorer's model e-fold time [ms] at ``port``."""
        iz = self.ports[port].iz
        proxy = self.model_n_win[:, iz] * np.sqrt(
            np.maximum(self.model_Te_win[:, iz], 0.0)
        )
        return _efold_time_ms(self.t_win_ms, proxy)

    def measured_tau_ms(self, port):
        """Return the measured e-fold time [ms] at ``port``, scorer convention."""
        ov = self.overlay
        t_exp = np.asarray(ov["isat_decay_time_ms"], dtype=float)
        isat = np.asarray(ov["isat_decay_mean_a"], dtype=float)
        ports = np.asarray(ov["isat_decay_port"])
        rows = np.flatnonzero(ports.astype(int) == int(port))
        if not rows.size:
            raise ValueError(
                f"port {port} has no isat_decay trace in "
                f"{self.overlay_path.name}"
            )
        p = int(rows[0])
        tail = isat[p, t_exp >= t_exp.max() - 5.0]
        noise = 5.0 * 1.4826 * np.nanmedian(np.abs(tail - np.nanmedian(tail)))
        win = (t_exp >= self.window_ms[0]) & (t_exp <= self.window_ms[1])
        return _efold_time_ms(t_exp[win], isat[p, win], noise)


def _read_json_attr(f, name, where):
    if name not in f.attrs:
        raise ValueError(f"{where} is missing the required attribute {name!r}")
    return json.loads(f.attrs[name])


def _derive(ctx, n, Ee, Ei):
    """Return floored ``(n_safe, Te, Ti)`` from conservative densities."""
    n_safe = max(n, ctx.ne_floor)
    Te = max((2.0 / 3.0) * Ee / (n_safe * ev_to_erg), ctx.Te_floor)
    Ti = max((2.0 / 3.0) * Ei / (n_safe * ev_to_erg), ctx.Ti_floor)
    return n_safe, Te, Ti


def _rhs(ctx, port_state, nn, end_loss_coefficient, y):
    """Return ``d(n, Ee, Ei)/dt`` [cm^-3 s^-1, erg cm^-3 s^-1]."""
    n, Ee, Ei = y
    n_safe, Te, Ti = _derive(ctx, n, Ee, Ei)
    ln_lambda = max(c_log(Te, n_safe, kind="ei"), ctx.ln_lambda_min)

    q_ie = (
        Q_ie(Te, Ti, n_safe, ctx.mu, ln_lambda, per_particle=False)
        * ev_to_erg
    )
    q_end_e = (
        end_loss_coefficient
        * elec_par_heat_loss(
            Te, n_safe, port_state.L_p, port_state.L_hf, ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
    )
    q_end_i = (
        end_loss_coefficient
        * _ion_par_heat_loss(
            Ti, n_safe, port_state.L_p, port_state.L_hf, ctx.mu, ln_lambda,
            per_particle=False,
        )
        * ev_to_erg
    )
    T_eff = 0.5 * (Ti + port_state.Tn)
    nu_mt = nn * float(
        phelps_momentum_transfer_rate_cm3_s(T_eff, gas_type=ctx.gas_type)
    )
    q_cx = (
        1.5 * ctx.b_drag * nu_mt * n_safe * (port_state.Tn - Ti) * ev_to_erg
    )

    c_s = float(plasma_wave_speed(Te, Ti, ctx.mu, ctx.wave_speed))
    nu_loss = ctx.alpha_isat * c_s / port_state.L_col
    dn = -n_safe * nu_loss
    convect = ENTHALPY_FACTOR * ev_to_erg * n_safe * nu_loss
    dEe = -q_ie - q_end_e - convect * Te
    dEi = q_ie + q_cx - q_end_i - convect * Ti
    return np.array([dn, dEe, dEi], dtype=float)


def integrate_port(ctx, port, *, Te0=None, Ti0=None, nn_scale=1.0,
                   end_loss_coefficient=1.0):
    """Return the surrogate ``Isat`` proxy on the run's own window sample times."""
    ps = ctx.ports[port]
    Te = ps.Te if Te0 is None else float(Te0)
    Ti = ps.Ti if Ti0 is None else float(Ti0)
    nn = ps.nn * float(nn_scale)
    if Te <= 0.0 or Ti <= 0.0 or nn <= 0.0:
        raise ValueError(
            f"port {port}: variation produced a non-positive initial "
            f"Te={Te!r}, Ti={Ti!r} or nn={nn!r}"
        )
    y = np.array(
        [ps.n, 1.5 * ps.n * Te * ev_to_erg, 1.5 * ps.n * Ti * ev_to_erg],
        dtype=float,
    )
    t_targets_s = (ctx.t_win_ms - ctx.t0_ms) * 1.0e-3
    out = np.empty(t_targets_s.size, dtype=float)
    t = 0.0
    for k, t_target in enumerate(t_targets_s):
        span = float(t_target) - t
        if span < 0.0:
            raise ValueError(
                f"port {port}: window sample time {t_target!r} s precedes the "
                f"shutoff sample; the trajectory's time axis is not increasing"
            )
        steps = max(1, int(math.ceil(span / SUB_DT_S)))
        h = span / steps
        for _ in range(steps):
            k1 = _rhs(ctx, ps, nn, end_loss_coefficient, y)
            k2 = _rhs(ctx, ps, nn, end_loss_coefficient, y + 0.5 * h * k1)
            k3 = _rhs(ctx, ps, nn, end_loss_coefficient, y + 0.5 * h * k2)
            k4 = _rhs(ctx, ps, nn, end_loss_coefficient, y + h * k3)
            y = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            if not np.all(np.isfinite(y)):
                raise ValueError(
                    f"port {port}: the surrogate state went non-finite at "
                    f"t = {t:.6g} s into the window"
                )
        t = float(t_target)
        n_safe, Te_now, _ = _derive(ctx, y[0], y[1], y[2])
        out[k] = n_safe * math.sqrt(max(Te_now, 0.0))
    return out


def surrogate_tau_ms(ctx, port, **kwargs):
    """Return the surrogate's fitted e-fold time [ms] at ``port``."""
    return _efold_time_ms(ctx.t_win_ms, integrate_port(ctx, port, **kwargs))


class Report:
    """Collect report lines for stdout and the artifact file."""

    def __init__(self):
        self.lines = []

    def __call__(self, text=""):
        print(text)
        self.lines.append(text)

    def write(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n")


def _rule(say, text):
    say("=" * 78)
    say(text)
    say("=" * 78)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=(
            "0D two-temperature afterglow surrogate for the ES1 ports "
            "(discriminator D1)."
        )
    )
    ap.add_argument(
        "--h5", default=str(SCRIPTS / "g1atrim_arm.h5"),
        help="saved LAPDSim1D trajectory to initialise from",
    )
    ap.add_argument(
        "--overlay", default=None,
        help="ES overlay npz (default: data/es<N>_sim1d_overlay.npz)",
    )
    ap.add_argument("--es", type=int, default=1, help="experiment-set rung")
    ap.add_argument(
        "--closure", choices=("gradient", "geometric"), default="gradient",
        help=(
            "heat-flux scale length: the model's own axial Te scale length at "
            "shutoff (gradient) or the distance to the nearest end (geometric)"
        ),
    )
    ap.add_argument(
        "--gate-tol", type=float, default=GATE_REL_TOL,
        help="pre-registered instrument-gate relative tolerance",
    )
    ap.add_argument("--output", default=None, help="write the report here too")
    args = ap.parse_args(argv)

    overlay = (
        Path(args.overlay)
        if args.overlay
        else SCRIPTS / "data" / f"es{args.es}_sim1d_overlay.npz"
    )
    ctx = RunContext(args.h5, overlay, DECAY_WINDOW_MS, args.closure)
    say = Report()

    _rule(say, "D1 -- 0D AFTERGLOW SURROGATE, ES%d PORTS" % args.es)
    say(f"artifact       : {ctx.h5_path}")
    say(f"overlay        : {ctx.overlay_path}")
    say(f"decay window   : {ctx.window_ms[0]:.4g}-{ctx.window_ms[1]:.4g} ms "
        f"(main-discharge clock), {ctx.window_idx.size} samples")
    say(f"shutoff sample : t = {ctx.t0_ms:.6f} ms, index {ctx.i0}")
    say(f"gas            : {ctx.gas_type} (mu={ctx.mu}, "
        f"m_i={ctx.ion_mass_g:.6e} g)")
    say(f"floors         : Te {ctx.Te_floor} eV, Ti {ctx.Ti_floor} eV, "
        f"n {ctx.ne_floor:.3g} cm^-3")
    say(f"c_s convention : plasma_wave_speed(..., {ctx.wave_speed!r}); "
        f"alpha_isat = {ctx.alpha_isat:.6f}")
    say(f"column         : plasma-active z {ctx.z_end_lo:.1f}..{ctx.z_end_hi:.1f} "
        f"cm, L_col = {ctx.L_col:.1f} cm")
    say(f"heat-flux L_hf : --closure {ctx.closure}")
    say(f"enthalpy factor: {ENTHALPY_FACTOR} T per lost particle")
    say(f"RK4 sub-step   : {SUB_DT_S:.1e} s")
    say()

    say("PRE-REGISTERED INSTRUMENT GATE (hard)")
    say(f"  The UNMODIFIED surrogate must reproduce each port's MODEL e-fold")
    say(f"  time within {args.gate_tol * 100:.0f} %. Any port outside that band")
    say("  stops the run: no variation is scored, and the failure is the")
    say("  deliverable. The surrogate is not tuned to pass.")
    say()
    say("PRE-REGISTERED VERDICT RULES")
    say("  A (state-mismatch) CARRIES if V1 alone recovers >= 70 % of the")
    say("    ln-tau gap to measurement.")
    say("  B (ion reservoir) SURVIVES TO D2 if V2 or V3 recovers >= 50 % at")
    say("    p21-p50 AND the cross-port pattern (p11 least anomalous) is")
    say("    preserved.")
    say("  If V4 does as well as V2/V3 the result is UNDERDETERMINED and all")
    say("    three are bracketed.")
    say("  Operationalisation of 'as well as' (disclosed reading, fixed before")
    say("    the run): the best V4 arm's mean recovery over p21-p50 is >= 0.9x")
    say("    the best V2/V3 arm's mean recovery over the same ports.")
    say()

    say("SHUTOFF STATE AND CLOSURE LENGTHS")
    say(f"{'port':>5} {'z[cm]':>8} {'n[cm^-3]':>11} {'Te[eV]':>8} {'Ti[eV]':>8} "
        f"{'nn[cm^-3]':>11} {'Tn[eV]':>8} {'L_p[cm]':>9} {'L_hf[cm]':>9}")
    for port in PORTS:
        ps = ctx.ports[port]
        say(f"{port:>5} {ps.z_cm:8.1f} {ps.n:11.4e} {ps.Te:8.4f} {ps.Ti:8.4f} "
            f"{ps.nn:11.4e} {ps.Tn:8.4f} {ps.L_p:9.1f} {ps.L_hf:9.1f}")
    say()

    model_tau = {p: ctx.model_tau_ms(p) for p in PORTS}
    exp_tau = {p: ctx.measured_tau_ms(p) for p in PORTS}
    say("MEASURED e-fold TARGETS")
    say("  tau_note = tau_model / (plot-note ratio, "
        "plotnotes_decay_starting_temperatures.txt:28-32)")
    say("  tau_ovl  = refit of the overlay isat_decay_mean_a trace on the same")
    say("             window, compare_sim1d_es1._efold_time_ms (line 748)")
    say(f"{'port':>5} {'ratio':>7} {'tau_model':>10} {'tau_note':>9} "
        f"{'tau_ovl':>9} {'note/ovl':>9}")
    for port in PORTS:
        ratio = NOTE_TAU_RATIO[port]
        t_note = model_tau[port] / ratio
        say(f"{port:>5} {ratio:7.2f} {model_tau[port]:9.4f}m "
            f"{t_note:8.4f}m {exp_tau[port]:8.4f}m "
            f"{t_note / exp_tau[port]:9.3f}")
    say()

    say("INSTRUMENT GATE -- unmodified surrogate vs MODEL")
    say(f"{'port':>5} {'tau_model':>10} {'tau_surr':>10} {'rel.dev':>9} "
        f"{'verdict':>8}")
    base_tau = {}
    failures = []
    for port in PORTS:
        tau = surrogate_tau_ms(ctx, port)
        base_tau[port] = tau
        if not np.isfinite(tau):
            dev = float("nan")
            ok = False
        else:
            dev = tau / model_tau[port] - 1.0
            ok = abs(dev) <= args.gate_tol
        if not ok:
            failures.append(port)
        say(f"{port:>5} {model_tau[port]:9.4f}m {tau:9.4f}m {dev:+9.1%} "
            f"{'PASS' if ok else 'FAIL':>8}")
    say()

    if failures:
        _rule(
            say,
            "INSTRUMENT GATE FAILED at port(s) "
            + ", ".join(f"p{p}" for p in failures)
            + f" (tolerance {args.gate_tol:.0%})",
        )
        say("The pre-registered protocol stops here: NO variation was run and")
        say("no ownership verdict is available from D1 on this artifact. The")
        say("surrogate was not adjusted to pass.")
        say()
        say("DIAGNOSTIC -- surrogate vs model rates AT the shutoff sample")
        say("  The surrogate opens the window at its asymptotic loss rate; the")
        say("  model's port cell is still near drive balance and unwinds into")
        say("  the afterglow over the first part of the window.")
        say(f"{'port':>5} {'dn/dt surr':>12} {'dn/dt model':>12} "
            f"{'dEe/dt surr':>13} {'dEe/dt model':>13} "
            f"{'dEi/dt surr':>13} {'dEi/dt model':>13}")
        for port in PORTS:
            ps = ctx.ports[port]
            y0 = np.array(
                [ps.n,
                 1.5 * ps.n * ps.Te * ev_to_erg,
                 1.5 * ps.n * ps.Ti * ev_to_erg],
                dtype=float,
            )
            d = _rhs(ctx, ps, ps.nn, 1.0, y0)
            iz = ps.iz
            say(f"{port:>5} {d[0]:+12.3e} {ctx.model_rhs0['n'][iz]:+12.3e} "
                f"{d[1] * 1.0e-7:+13.3e} "
                f"{ctx.model_rhs0['Ee'][iz] * 1.0e-7:+13.3e} "
                f"{d[2] * 1.0e-7:+13.3e} "
                f"{ctx.model_rhs0['Ei'][iz] * 1.0e-7:+13.3e}")
        say("  units: dn/dt [cm^-3 s^-1], dE/dt [W cm^-3]")
        if args.output:
            say.write(args.output)
        return 1

    _rule(say, "INSTRUMENT GATE PASSED -- proceeding to the variations")
    say(CAVEATS_V1)
    say()

    def ln_gap(tau, port):
        return math.log(tau) - math.log(exp_tau[port])

    base_gap = {p: ln_gap(base_tau[p], p) for p in PORTS}

    arms = [
        ("V1 Te(0)->measured", dict(kwargs_per_port=lambda p: {"Te0": V1_TE0_EV[p]})),
        ("V2 Ti(0)->1 eV", dict(kwargs_per_port=lambda p: {"Ti0": 1.0})),
        ("V3 nn x3", dict(kwargs_per_port=lambda p: {"nn_scale": 3.0})),
        ("V3 nn x10", dict(kwargs_per_port=lambda p: {"nn_scale": 10.0})),
        ("V4 end-loss x2", dict(kwargs_per_port=lambda p: {"end_loss_coefficient": 2.0})),
        ("V4 end-loss x4", dict(kwargs_per_port=lambda p: {"end_loss_coefficient": 4.0})),
    ]

    recovery = {}
    say("VARIATIONS -- fitted tau and ln-tau gap recovery")
    say("  recovery = 1 - (ln tau_arm - ln tau_exp) / (ln tau_base - ln tau_exp)")
    for label, spec in arms:
        say()
        say(f"--- {label}")
        say(f"{'port':>5} {'tau_base':>9} {'tau_arm':>9} {'tau_exp':>9} "
            f"{'gap_base':>9} {'gap_arm':>9} {'recovered':>10}")
        rec = {}
        for port in PORTS:
            tau = surrogate_tau_ms(ctx, port, **spec["kwargs_per_port"](port))
            if not np.isfinite(tau):
                raise ValueError(
                    f"{label}: port {port} produced a non-decaying surrogate "
                    f"proxy, so no e-fold time exists"
                )
            gap = ln_gap(tau, port)
            r = 1.0 - gap / base_gap[port]
            rec[port] = r
            say(f"{port:>5} {base_tau[port]:8.4f}m {tau:8.4f}m "
                f"{exp_tau[port]:8.4f}m {base_gap[port]:+9.4f} {gap:+9.4f} "
                f"{r:10.1%}")
        recovery[label] = rec
        say(f"  mean recovery, all ports : "
            f"{np.mean([rec[p] for p in PORTS]):.1%}")
        say(f"  mean recovery, p21-p50   : "
            f"{np.mean([rec[p] for p in PORTS[1:]]):.1%}")

    say()
    _rule(say, "PRE-REGISTERED VERDICT")

    v1 = recovery["V1 Te(0)->measured"]
    v1_mean = float(np.mean([v1[p] for p in PORTS]))
    rule_a = v1_mean >= 0.70
    say(f"Rule A: V1 mean recovery over all ports = {v1_mean:.1%} "
        f"({'>=' if rule_a else '<'} 70 %)")

    b_arms = ["V2 Ti(0)->1 eV", "V3 nn x3", "V3 nn x10"]
    b_means = {a: float(np.mean([recovery[a][p] for p in PORTS[1:]])) for a in b_arms}
    best_b = max(b_means, key=b_means.get)
    # Cross-port pattern: p11 stays the least anomalous port (smallest
    # residual ln-gap) after the variation, as it is in the measurement.
    pattern = {}
    for a in b_arms:
        resid = {
            p: abs(base_gap[p] * (1.0 - recovery[a][p])) for p in PORTS
        }
        pattern[a] = min(resid, key=resid.get) == 11
    rule_b = b_means[best_b] >= 0.50 and pattern[best_b]
    for a in b_arms:
        say(f"Rule B: {a:<20} mean recovery p21-p50 = {b_means[a]:6.1%}, "
            f"p11-least-anomalous {'preserved' if pattern[a] else 'BROKEN'}")

    v4_arms = ["V4 end-loss x2", "V4 end-loss x4"]
    v4_means = {a: float(np.mean([recovery[a][p] for p in PORTS[1:]])) for a in v4_arms}
    best_v4 = max(v4_means, key=v4_means.get)
    say(f"Rule C: best V4 ({best_v4}) mean recovery p21-p50 = "
        f"{v4_means[best_v4]:.1%} vs best V2/V3 ({best_b}) "
        f"{b_means[best_b]:.1%}")
    underdetermined = (
        b_means[best_b] > 0.0
        and v4_means[best_v4] >= 0.9 * b_means[best_b]
    )

    say()
    if underdetermined:
        say("RULE FIRED: UNDERDETERMINED -- V4 (understated end-loss drain) "
            "does as well")
        say("  as the best ion-reservoir arm, so A, B and C are bracketed and "
            "no single")
        say("  ownership candidate is separated by D1.")
    elif rule_a and not rule_b:
        say("RULE FIRED: A CARRIES -- V1 alone recovers >= 70 % of the ln-tau "
            "gap.")
    elif rule_b and not rule_a:
        say("RULE FIRED: B SURVIVES TO D2 -- the ion-reservoir arm clears 50 % "
            "at")
        say("  p21-p50 with the cross-port pattern preserved.")
    elif rule_a and rule_b:
        say("RULE FIRED: A CARRIES *and* B SURVIVES -- both thresholds are met, "
            "so the")
        say("  two candidates are not separated by D1 and both go forward.")
    else:
        say("RULE FIRED: NONE -- no pre-registered threshold was met. D1 "
            "separates")
        say("  nothing on this artifact; the surrogate reproduces the model but "
            "none")
        say("  of the four input perturbations recovers the measured decay.")

    if args.output:
        say.write(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
