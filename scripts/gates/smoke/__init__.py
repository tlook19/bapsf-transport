"""The LAPDSim1D smoke suite, one module per subsystem.

``scripts/gates/smoke_sim1d.py`` is the entry point; it calls ``main``. This
package imports the harness and every case module, sorts the registry into
``_CASE_ORDER``, and runs the two import-time registry checks (the case
census and case-body reachability).

A case module registers its cases with ``@_case`` when it is imported, so
registration order follows the module list below. The run order is
``_CASE_ORDER``, the order the cases had in the single-file suite. A new case
goes at the END of ``_CASE_ORDER``; a removed case leaves it. The loader
refuses a registry that does not match the list exactly, in either
direction.
"""

from ._harness import (
    _CASES,
    _PRODUCER,
    _assert_case_bodies_reachable,
    _assert_case_census,
    main,
)
from . import (  # noqa: F401  (imported to register their cases)
    configuration,
    geometry,
    end_wall,
    circuit_cathode,
    beam,
    neutrals,
    atomic_rates,
    dvm,
    phases,
    numerics,
    results_io,
    compiled,
)

#: Every case name, in run order. ``--list`` prints this order and the full
#: suite runs it.
_CASE_ORDER = (
    "production-construction-warning-free",
    "shipped-defaults-and-base-geometry",
    "source-fixed-grid",
    "variable-area-well-balancedness",
    "twin-cathode-plateau-multigroup",
    "cathode-spitzer-and-base-boundary",
    "cathode-boundary-beam-terms",
    "cathode-annular-solve-fixtures",
    "cathode-current-driven-sheath-solve",
    "cathode-clamp-census",
    "circuit-current-driven-integration",
    "cathode-power-balance-under-current-drive",
    "beam-manifold-excitation-model",
    "beam-csda-deposition-model",
    "beam-gap-transmission-probe",
    "beam-gap-ledger-tripwire",
    "beam-probe-skip",
    "beam-anode-mesh-interception",
    "beam-walked-tail-fixtures",
    "tail-forward-default-inert",
    "tail-forward-energy-closure",
    "tail-forward-direction",
    "tail-forward-refusals",
    "beam-plateau-multigroup",
    "anode-tail-cull-crossing-rule",
    "anode-tail-sheath-reflection",
    "anode-tail-circuit-coupling",
    "beam-deposition-smoothing-conservation",
    "beam-smoothing-matrix-cache",
    "ionization-birth-energy-model",
    "gas-puff-diagnostics-and-fluid-operators",
    "energy-exchange-rate-bound",
    "helium-only-reaction-rates",
    "no-source-run-and-results",
    "cathode-power-balance-warming",
    "timestep-dt-growth-reapproach",
    "timestep-surface-loss-floor-exempt-hysteresis",
    "breakdown-retry-near-vacuum",
    "current-phase-raise-on-timeout",
    "ignition-failure-diagnostics",
    "non-ignition-guards",
    "equilibration-puff-duty",
    "equilibration-puff-width",
    "ion-neutral-closure-knobs",
    "neutral-momentum-state-foundations",
    "two-zone-neutral-state-foundations",
    "neutral-momentum-sources",
    "neutral-two-zone-particle-channel",
    "transient-dvm-neutrals-k2a",
    "obstruction-geometry-production-style",
    "neutral-wind-advection",
    "gas-puff-orifice-profile",
    "anode-disc-radius",
    "sigma-in-phelps",
    "adas-atomic-rate-model",
    "retired-gas-type-and-rate-model-keys",
    "he-singlet-manifold-registry",
    "csda-module-standalone",
    "csda-per-cell-accumulators",
    "csda-hoisted-stopping-coefficient",
    "csda-ql-heating-locality",
    "csda-walk-window-reflection-k7",
    "directed-recycle-jets",
    "cathode-jet-hot-carrier",
    "adas-low-te-extension-retired",
    "gcr-recombination-energy-pair",
    "square-gas-puff-waveform",
    "electrode-sample-smoothing",
    "restart-saved-evidence-r1b",
    "resolved-config-manifest-r1e",
    "neutral-equilibration-run-warning",
    "gas-puff-source-born-at-rest",
    "compiled-kernel-equivalence",
    "dt-min-lock",
    "ql-relaxation-onset-gate",
    "ql-relaxation-books-and-conserves",
    "ql-relaxation-module-refusals",
    "ql-relaxation-compiled-kernel-refusal",
    "ql-relaxation-presence-gating",
    "ql-relaxation-solver-refusals",
    "ql-relaxation-km-table",
    "shaped-initial-neutral-fill-sp3",
    "fill-spreading-knudsen-operator",
    "equilibration-map-slicer",
    "prescribed-area-geometry",
    "prescribed-area-trivial-profile-identity",
    "prescribed-area-well-balancedness",
    "config-key-namespace-and-seed-cache",
    "closed-experiment-keys-retired",
    "hot-channel-internal-wall",
    "mirror-field-loader",
    "mirror-field-loader-refusals",
    "kinetic-geff-thermal-floor",
    "ionization-birth-neutral-temperature",
    "golden-digest-gate-deterministic",
    "phase3-artifact-locator-battery",
    "golden-baseline-config-constructs",
    "cathode-closed-audit-export",
    "configuration-derived-resolution",
    "configuration-restated-delta-refusal",
    "configuration-chain-depth-refusal",
    "configuration-unknown-key-delta-refusal",
    "configuration-load-config-refuses-configuration-form",
    "configuration-hdf5-lineage-round-trip",
    "configuration-drivers-refuse-unnamed-runs",
    "configuration-drivers-refuse-rung-owned-supersession",
    "configuration-fluid-comparator-example",
    "configuration-every-committed-example-constructs",
    "configuration-restated-block-refusal",
    "circuit-cathode-retired-keys-refuse",
    "ts-retirement-successor-key",
    "dvm-particle-ledger-export",
    "dvm-neutral-moment-export",
    "parallel-momentum-sink-refusals",
    "parallel-momentum-sink-mechanism",
    "prescribed-drive-refusals",
    "prescribed-drive-handoff",
    "golden-fixture-packed-row-count",
    "dvm-jet-rn-interval-refusals",
    "floor-audit-names-its-configuration",
    "floating-open-circuit-current-balance",
    "afterglow-tail-handoff-criterion",
    "tail-handoff-surface-continuity",
    "end-face-full-debit-split",
    "end-wall-debit-armed-by-geometry-role",
    "cathode-emitted-fall-beam-row-non-overlap",
    "end-wall-lambda-eff-barrier-bracket",
    "end-wall-rename-retired-names",
    "cell-role-whitelist-on-load",
    "configuration-file-value-typed-to-template",
    "dt-min-lock-union-summary",
    "far-end-double-ratio-area-cancels",
    "far-end-double-ratio-per-family-gating",
    "far-end-double-ratio-empty-legend-guard",
    "far-end-double-ratio-shared-keys-skip",
    "effective-cathode-flags-refuses-driven-override-in-floating-phase",
    "kep-pressure-work-closure",
    "kep-constant-area-discriminators",
    "kep-flare-adiabat",
    "kep-terminating-cells",
    "kep-acoustic-symbol",
    "sound-speed-true-ion-mass",
    "end-wall-face-sheath-edge-flux",
    "cathode-face-one-ion-current",
    "cathode-jet-incident-power-one-book",
    "implicit-ee-sink-substep-identity",
    "implicit-ee-sink-pure-decay-exact",
    "implicit-ee-sink-substep-order",
    "anode-e-sheath-realised-equals-booked",
    "cathode-e-climb-realised-equals-booked",
    "anode-cells-no-within-step-sawtooth",
    "anode-ion-collection-counted-vs-circuit",
    "dt-not-bound-by-anode-row",
    "anode-e-sheath-row-reported-not-applied",
    "implicit-ee-sink-no-solve-bit-identity",
    "numerics-retired-keys-refuse",
    "result-bitdiff-compare-synthetic",
    "beam-tail-retired-keys-refuse",
    "neutral-retired-keys-refuse",
    "beam-l-b-profile-at-fed-back-cross",
    "mirror-half-column-mesh",
    "mirror-refusals",
    "mirror-face-flux-unit",
    "mirror-fluid-march",
    "twin-mirror-mesh-identity",
    "mirror-puff-row-reflection",
    "order-gate-envelope-fit",
    "dvm-mirror-plane-specular",
    "dvm-mirror-no-lagged-buffer",
    "mirror-tail-per-face-rule",
    "mirror-tail-full-window-fold",
    "mirror-tail-cull-rearmed",
    "mirror-tail-leg-cap-residual",
    "mirror-beam-smoothing-fold",
    "mirror-csda-primary-turn",
    "mirror-cathode-coupling-constructs",
    "mirror-compiled-equivalence",
    "mirror-primary-needs-absorbing-anode",
    "mirror-residual-bound-raises",
    "mirror-tail-sheath-share-merged",
    "anode-tail-booking-identity",
    "anode-tail-booking-conservation-assert",
    "anode-balance-floor-probe-vs-dispatched",
    "anode-tail-booking-mirror-walked-tail",
    "anode-direct-booking-sheath-rule",
    "anode-booking-evaluators-plumbed",
    "anode-gap-born-outbound-only",
    "anode-booking-consumer-assert",
    "walker-fate-assert-negative-control",
    "anode-booking-diagnostics-saved",
    "anode-outbound-primary-sheath-rule",
)


def _apply_case_order():
    """Sort ``_CASES`` into ``_CASE_ORDER`` and rebuild ``_PRODUCER`` in it.

    Raises AssertionError when a registered case is missing from the list,
    when the list names a case no module registers, or when the list repeats
    a name.
    """
    registered = [entry.name for entry in _CASES]
    unlisted = sorted(set(registered) - set(_CASE_ORDER))
    unregistered = sorted(set(_CASE_ORDER) - set(registered))
    repeated = sorted({n for n in _CASE_ORDER if _CASE_ORDER.count(n) > 1})
    if unlisted or unregistered or repeated:
        raise AssertionError(
            "smoke _CASE_ORDER does not match the registry: "
            f"registered but not listed {unlisted}, listed but not "
            f"registered {unregistered}, listed twice {repeated}. Update "
            "_CASE_ORDER in smoke/__init__.py in the same commit that adds "
            "or removes a case."
        )
    rank = {name: index for index, name in enumerate(_CASE_ORDER)}
    _CASES.sort(key=lambda entry: rank[entry.name])
    # First producer in RUN order, as the single-file registry recorded it.
    _PRODUCER.clear()
    for entry in _CASES:
        for key in entry.provides:
            _PRODUCER.setdefault(key, entry.name)


_apply_case_order()
_assert_case_census()
_assert_case_bodies_reachable()
