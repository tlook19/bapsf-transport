"""Breakdown-progress diagnostics and the ignition-stall joint condition.

The solver records, at every trajectory save taken while the cathode drive is
active and the run is still in ``pre_breakdown``/``breakdown``, a small set of
scalars that describe how the discharge is (or is not) building:

* ``gamma_N``  -- windowed ``d(ln N_plasma)/dt`` of the column-integrated
  plasma inventory. The growth rate of the thing that must grow.
* ``gamma_nn`` -- the same for the neutral inventory. **Diagnosis only.** It
  appears in NO trip condition; it exists so the beam-turn-on spike ->
  initial-inventory-burn trough -> puff-restored climb story is visible in the
  saved artifact rather than having to be inferred.
* ``dEe_total/dt`` -- the electron thermal-energy inventory rate. This is the
  prognosis variable: a discharge that is going to ignite is retaining heat
  even while its density dips.
* the electron power-balance split -- beam coupled power against conduction,
  cooling, and the ionization cost, plus the WP-D beam end-loss ledger (which
  is a real loss channel that never enters an RHS row, so it has to be read
  from the cathode diagnostics, and is identically zero under
  ``beam_product_transport='local'``).

``IgnitionMonitor`` owns the ring buffer and the trip logic. It is deliberately
a plain, side-effect-free object over ``(t, N, N_n, Ee_tot)`` tuples so the
joint condition can be unit-tested on synthetic histories and replayed offline
against a saved trajectory.

The trip condition is STRUCTURAL -- two zero-crossings, no tuned rates:

    gamma_N <= 0  AND  d(Ee_total)/dt <= 0,
    for EVERY sample in a sustained window,
    armed only while the cathode drive is on.

Rationale (settled design, do not relax): during the refueling trough of a
healthy start-up the plasma inventory genuinely falls, but the electrons are
still being heated and have just lost their largest sink (ionization cooling
dropped with the fuel), so ``dEe_total/dt`` stays positive and the JOINT
condition does not trip. False positives (a killed good run) cost far more
than false negatives (wall time, which the tau_prebreakdown outer guard bounds
anyway), so every ambiguous state -- oscillating, slowly positive, hot but not
growing -- must fall through untripped. There is NO CLOCK IN THE VERDICT: the
device's ignition span runs from 4-6 ms at high density to 10-15 ms at low
density, and low-density parameters may be requested with no calibration data,
so the window is transient rejection and never a patience limit.
"""

import math


# --- Window constants (solver constants, deliberately NOT config keys) -----
#
# IGNITION_STALL_WINDOW_S is the sustained window the joint condition must
# hold across before the run is declared stalled. It is anchored on the
# reference trigger scale of the adopted ES1 production point: that run
# crosses its I_prebreakdown threshold 2.34 ms into the shot
# (es1_prod_promoted_nx240.h5, t_prebreakdown_trigger = 2.3359e-3 s) and
# reaches I_breakdown at 2.72 ms, so 2.5 ms is longer than the ENTIRE healthy
# high-density pre-ignition transient -- the detector cannot assemble a full
# window inside one. At low density, where ignition takes 10-15 ms, it rejects
# transients of the same duration as a whole high-density ignition while
# leaving several window-lengths of headroom before the slow build completes.
# It is a transient-rejection scale, not a deadline: a run that is still
# heating never trips no matter how long it takes.
IGNITION_STALL_WINDOW_S = 2.5e-3

# Sub-scale over which each sample's own rates are differenced. One fifth of
# the sustained window: long enough that the rate is not a single-save
# difference (which flickers sign on the save cadence and would make the
# detector unable to fire at all), short enough that five independent rate
# estimates fit inside one sustained window.
IGNITION_RATE_WINDOW_S = IGNITION_STALL_WINDOW_S / 5.0

# Minimum number of saved samples inside the sustained window. Guards the
# coarse-``dt_save`` limit, where a window could otherwise be satisfied by two
# or three points.
IGNITION_STALL_MIN_SAMPLES = 5


# --- Saved per-sample diagnostic fields ------------------------------------
IGNITION_DIAGNOSTIC_FIELDS = (
    "armed",
    "N_plasma",
    "N_neutral",
    "Ee_total_erg",
    "gamma_N_per_s",
    "gamma_nn_per_s",
    "dEe_total_W",
    "P_beam_W",
    "P_conduction_W",
    "P_cooling_W",
    "P_ionization_W",
    "P_transport_W",
    "P_beam_end_loss_W",
    "joint_negative",
    "stalled",
)

# Electron power-balance grouping over the solver's named Ee RHS rows. Every
# remaining row is booked into P_transport_W, so the five power lines sum to
# the total electron RHS power and the ledger closes.
IGNITION_POWER_GROUPS = {
    "P_beam_W": ("beam_power_deposition",),
    "P_conduction_W": ("heat_conduction",),
    "P_cooling_W": (
        "ei_exchange",
        "electron_ion_cooling",
        "electron_neutral_cooling",
        "beam_excitation_radiation",
        "recombination_rad_loss",
        "recombination_3b_loss",
        "recombination_energy_return",
    ),
    "P_ionization_W": (
        "ionization_energy_cost",
        "beam_ionization_cost",
    ),
}

# WP-D / WP-E end ledger keys (cathode diagnostics). Beam energy that leaves
# the column axially without thermalizing; never booked into any RHS row.
# The first four are the WP-D event-product ledger (identically zero under
# beam_product_transport="local"); the last four are the WP-E QL tail ledger
# (identically zero under heating_anomalous_transport="local"). Both defaults
# are the off value, so P_beam_end_loss_W is exactly 0.0 on every run that
# enables neither, and adding the tail terms cannot perturb a WP-D-only run
# (adding 0.0 is exact).
IGNITION_BEAM_END_LOSS_KEYS = (
    "source_beam_end_loss_low_W",
    "source_beam_end_loss_high_W",
    "end_beam_end_loss_low_W",
    "end_beam_end_loss_high_W",
    "source_beam_end_loss_tail_low_W",
    "source_beam_end_loss_tail_high_W",
    "end_beam_end_loss_tail_low_W",
    "end_beam_end_loss_tail_high_W",
)


def empty_ignition_diagnostics():
    """Return the NaN/zero-defaulted per-sample diagnostic record."""
    record = {name: math.nan for name in IGNITION_DIAGNOSTIC_FIELDS}
    record["armed"] = 0.0
    record["joint_negative"] = 0.0
    record["stalled"] = 0.0
    return record


class _Sample:
    __slots__ = ("time", "N", "N_n", "Ee", "joint")

    def __init__(self, time, N, N_n, Ee, joint):
        self.time = time
        self.N = N
        self.N_n = N_n
        self.Ee = Ee
        self.joint = joint


class IgnitionMonitor:
    """Ring buffer over ``(t, N_plasma, N_neutral, Ee_total)`` save samples.

    ``record`` is called once per trajectory save. When ``armed`` is false the
    buffer is cleared and the returned rates are NaN, so a sustained window can
    never straddle an arming boundary (the detector cannot inherit history from
    a phase in which the drive was off).
    """

    def __init__(
        self,
        window_s=IGNITION_STALL_WINDOW_S,
        rate_window_s=IGNITION_RATE_WINDOW_S,
        min_samples=IGNITION_STALL_MIN_SAMPLES,
    ):
        window_s = float(window_s)
        rate_window_s = float(rate_window_s)
        min_samples = int(min_samples)
        if not (window_s > 0.0 and math.isfinite(window_s)):
            raise ValueError(
                "IgnitionMonitor window_s must be a positive finite duration "
                f"[s] (got {window_s!r})"
            )
        if not (0.0 < rate_window_s <= window_s):
            raise ValueError(
                "IgnitionMonitor rate_window_s must satisfy "
                f"0 < rate_window_s <= window_s (got {rate_window_s!r} with "
                f"window_s={window_s!r})"
            )
        if min_samples < 2:
            raise ValueError(
                "IgnitionMonitor min_samples must be >= 2 (a rate needs two "
                f"points; got {min_samples!r})"
            )
        self.window_s = window_s
        self.rate_window_s = rate_window_s
        self.min_samples = min_samples
        self._samples = []
        self._stalled = False

    @property
    def stalled(self):
        return self._stalled

    def reset(self):
        """Drop the buffered history (the trip latch is NOT cleared)."""
        self._samples = []

    def record(self, time, N_plasma, N_neutral, Ee_total, armed):
        """Buffer one save sample and return its diagnostics.

        Returns a dict with ``gamma_N_per_s``, ``gamma_nn_per_s``,
        ``dEe_total_erg_per_s``, ``joint_negative`` and ``stalled``.
        """
        if not armed:
            self.reset()
            return {
                "gamma_N_per_s": math.nan,
                "gamma_nn_per_s": math.nan,
                "dEe_total_erg_per_s": math.nan,
                "joint_negative": False,
                "stalled": self._stalled,
            }

        time = float(time)
        reference = None
        for sample in reversed(self._samples):
            if time - sample.time >= self.rate_window_s:
                reference = sample
                break

        gamma_N = math.nan
        gamma_nn = math.nan
        dEe = math.nan
        if reference is not None and time > reference.time:
            span = time - reference.time
            gamma_N = _log_rate(reference.N, N_plasma, span)
            gamma_nn = _log_rate(reference.N_n, N_neutral, span)
            dEe = (float(Ee_total) - reference.Ee) / span

        joint = bool(
            math.isfinite(gamma_N)
            and math.isfinite(dEe)
            and gamma_N <= 0.0
            and dEe <= 0.0
        )
        self._samples.append(
            _Sample(time, float(N_plasma), float(N_neutral), float(Ee_total), joint)
        )
        # Keep the sustained window plus one rate window, and always retain one
        # sample older than that cutoff so the window-coverage test below can
        # be satisfied at any save cadence.
        cutoff = time - (self.window_s + self.rate_window_s)
        while len(self._samples) > 2 and self._samples[1].time <= cutoff:
            self._samples.pop(0)

        if self._trip_condition(time):
            self._stalled = True
        return {
            "gamma_N_per_s": gamma_N,
            "gamma_nn_per_s": gamma_nn,
            "dEe_total_erg_per_s": dEe,
            "joint_negative": joint,
            "stalled": self._stalled,
        }

    def _trip_condition(self, time):
        if not self._samples:
            return False
        window_start = time - self.window_s
        # The armed history must actually COVER the window; otherwise a run
        # that has only just armed could trip on a short opening transient.
        if self._samples[0].time > window_start:
            return False
        in_window = [
            sample for sample in self._samples if sample.time >= window_start
        ]
        if len(in_window) < self.min_samples:
            return False
        return all(sample.joint for sample in in_window)


def _log_rate(before, after, span):
    before = float(before)
    after = float(after)
    if not (before > 0.0 and after > 0.0):
        return math.nan
    return (math.log(after) - math.log(before)) / span


def longest_joint_negative_span(times, gamma_N, dEe_dt, armed=None):
    """Return the longest contiguous armed joint-negative stretch [s].

    Offline companion to the trip logic: the acceptance metric a replay
    reports against ``IGNITION_STALL_WINDOW_S``. A sample counts only when
    both rates are finite, non-positive, and (optionally) armed.
    """
    worst = 0.0
    start = None
    for index, time in enumerate(times):
        is_armed = True if armed is None else bool(armed[index])
        g = float(gamma_N[index])
        e = float(dEe_dt[index])
        joint = (
            is_armed
            and math.isfinite(g)
            and math.isfinite(e)
            and g <= 0.0
            and e <= 0.0
        )
        if joint:
            if start is None:
                start = float(time)
            worst = max(worst, float(time) - start)
        else:
            start = None
    return worst
