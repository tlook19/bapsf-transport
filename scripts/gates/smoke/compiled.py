"""Smoke cases: compiled-kernel equivalence against the pure path."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from cablp.cathode import kernels as _kernel_selector

from ._harness import _case


# --------------------------------------------------------------------
# compiled-kernel-equivalence
# --------------------------------------------------------------------
@_case("compiled-kernel-equivalence")
def _case_compiled_kernel_equivalence():
    # --- Compiled-kernel END-TO-END equivalence (D3/D4, opt-in) -----------
    # The suite itself runs on the pure path (asserted above), and the D4
    # block compares each compiled kernel against its pure twin one function
    # at a time. Neither answers the end-to-end question: does a SOLVER
    # driven through the compiled kernels reach the same state? The kernel
    # binding happens once, at import, so the two paths cannot coexist in one
    # process -- each is a subprocess carrying its own CABLP_COMPILED_KERNELS.
    #
    # Gated on the extension being BUILT, not on this process having opted
    # in: the parent smoke deliberately runs pure (the
    # no-source-run-and-results case asserts it), so
    # keying this off the parent's env var would leave the section dead in
    # every gate invocation. On a checkout with no extension it SKIPS and
    # never fails -- the compiled path is opt-in by design and a pure
    # checkout must stay green.
    _ck_expected_kernel_id = "cython/_cathode_kernels_cy/tierA+csda"
    try:
        import importlib as _ck_importlib

        _ck_module = _ck_importlib.import_module(
            "cablp.cathode._cathode_kernels_cy"
        )
    except ImportError:
        _ck_module = None
    if _ck_module is None:
        print(
            "compiled-kernel equivalence: SKIPPED -- "
            "cablp.cathode._cathode_kernels_cy is not built "
            "(`python build_ext.py --inplace` enables it)"
        )
    else:
        assert _ck_module.KERNEL_ID == _ck_expected_kernel_id, (
            _ck_module.KERNEL_ID
        )
        # TWO scenarios, run through the same child harness:
        #
        # * ``meanfield`` -- a short current-driven discharge on the shared
        #   base stated below. The cathode sheath solve (Tier A) runs on every
        #   sample and the CSDA ray fires, so both halves of "tierA+csda" are
        #   on the hot path.
        # * ``initial_profile`` -- the shaped initial neutral fill armed.
        #
        # LAYOUT (R2a fold-in, 2026-08-20): both scenarios share ONE base,
        # and it is the pre-R2a 6-field cold-neutral stance -- the child spells
        # _pin_pre_r2a_neutral_stance out itself, since a subprocess cannot
        # import the parent's helper (the TOML block in the
        # cli-run-and-plot-end-to-end case, since retired, spelled the
        # same pin out for the same reason). This block asks a KERNEL
        # EQUIVALENCE question, not a closure question: the compiled kernels
        # are the tier-A sheath solve and the CSDA march, neither of which
        # contains any neutral-closure code, so composing the folded closure
        # family would add no compiled coverage while moving every trajectory
        # the expected step counts and anti-vacuity thresholds below were
        # calibrated against. The closure family keeps its own blocks, which
        # build their own configs and exercise the shipped defaults.
        #
        # The child counts the nested marches by wrapping ``deposit_beam`` in
        # its DEFINING module, so the count is exactly the nested legs and
        # excludes the top-level rays ``cathode`` calls through its own
        # imported name. Both
        # children carry the identical wrapper, so the census cannot perturb
        # the comparison it makes non-vacuous.
        _ck_child_source = '''
import json
import sys

import numpy as np

from cablp.cathode import beam_deposition as _beam_dep
from cablp.cathode import kernels as K
from cablp.solvers._sim1d import LAPDSim1D, default_config

scenario = sys.argv[1]
params, flags = default_config()
# The shared base for every scenario below: _pin_pre_r2a_neutral_stance,
# spelled out because this child is a separate process. See the LAYOUT note
# in the parent for why kernel equivalence is compared on the pinned stance.
flags["neutral_momentum"] = False
flags["neutral_energy"] = False
flags["neutral_hot_internal_wall"] = False
params["cathode_neutral_jet"] = False
params["cathode_jet_surface_debit"] = False
params["cathode_jet_energy_convention"] = "legacy"
params.update({
    "dt_save": 0.0,
    "phase_transition_mode": "scheduled",
    "tau_neutral_prebreakdown": 0.0,
    "tau_prebreakdown": 0.0,
    "tau_breakdown": 0.0,
    "tau_discharge": 1.0,
    "tau_afterglow": 0.0,
})
if scenario == "meanfield":
    params["nx"] = 24
    t_end = 2.0e-6
elif scenario == "initial_profile":
    # sp3: the shaped initial neutral fill ARMED. The array enters the state
    # before any kernel runs, so what this asks is whether an initial
    # condition the compiled path has never seen still leads both paths
    # through the same arithmetic -- the profile is strongly non-uniform, so
    # the two runs disagree everywhere if it does not.
    params.update({
        "nx": 12,
        "beam_anomalous_model": "quasilinear",
        "cathode_Ts_base_K": 1998.15,
        "cathode_cleaning_E_th_eV": None,
    })
    params["initial_neutral_state"] = "fill"
    _cells = int(LAPDSim1D(dict(params), dict(flags)).geometry.cells)
    params["nn0_profile"] = (
        float(params["nn0"])
        * (1.5 + np.sin(np.arange(_cells, dtype=float)))
    ).tolist()
    params["nn0"] = None
    params["initial_neutral_state"] = "profile"
    t_end = 1.0e-6
else:
    raise SystemExit(f"unknown scenario {scenario!r}")

_marches = [0]
_deposit_beam = _beam_dep.deposit_beam


def _counted_deposit_beam(*args, **kwargs):
    _marches[0] += 1
    return _deposit_beam(*args, **kwargs)


_beam_dep.deposit_beam = _counted_deposit_beam

result = LAPDSim1D(params, flags).run(t_end=t_end, dt=1.0e-7)
diag = result.cathode_diagnostics
print(json.dumps({
    "provenance": K.PROVENANCE,
    "kernel_id": (
        None if K.COMPILED_KERNELS is None
        else str(K.COMPILED_KERNELS.KERNEL_ID)
    ),
    "requested": bool(K.compiled_kernels_requested()),
    "steps": int(result.steps),
    "nested_marches": int(_marches[0]),
    "solve_enabled": float(np.min(diag["solve_enabled"])),
    "has_solution": float(np.min(diag["has_solution"])),
    "beam_csda_active": float(np.max(diag["beam_csda_active"])),
    # The tail end ledger: identically zero unless a walk carried power.
    "tail_ledger_W": float(
        np.max(diag["source_beam_end_loss_tail_low_W"])
        + np.max(diag["source_beam_end_loss_tail_high_W"])
    ),
    # sp3 anti-vacuity: the initial neutral fill's spread across the grid. A
    # uniform fill gives exactly 0, so a nonzero value is proof the shaped IC
    # was live on the path being compared.
    "nn0_spread": float(np.max(result.nn[0]) - np.min(result.nn[0])),
    "I_tot": float(diag["source_I_tot"][-1]),
    "phi_c": float(diag["source_phi_c"][-1]),
    "y": np.ascontiguousarray(result.y[-1], dtype=float).tobytes().hex(),
}))
'''
        _ck_expected_steps = {
            "meanfield": 20, "initial_profile": 10,
        }
        _CK_SCENARIOS = ("meanfield", "initial_profile")
        _ck_results = {}
        with tempfile.TemporaryDirectory() as _ck_tmpdir:
            _ck_script = Path(_ck_tmpdir) / "compiled_equivalence_child.py"
            _ck_script.write_text(_ck_child_source)
            for _ck_scenario in _CK_SCENARIOS:
                for _ck_tag, _ck_optin in (("pure", None), ("compiled", "1")):
                    # Inherit the environment (PYTHONPATH decides WHICH
                    # checkout the child imports) and override only the opt-in.
                    _ck_env = dict(os.environ)
                    if _ck_optin is None:
                        _ck_env.pop(_kernel_selector.ENV_VAR, None)
                    else:
                        _ck_env[_kernel_selector.ENV_VAR] = _ck_optin
                    _ck_proc = subprocess.run(
                        [sys.executable, str(_ck_script), _ck_scenario],
                        env=_ck_env,
                        capture_output=True,
                        text=True,
                    )
                    assert _ck_proc.returncode == 0, (
                        _ck_scenario,
                        _ck_tag,
                        _ck_proc.returncode,
                        _ck_proc.stderr[-2000:],
                    )
                    # Warnings go to stderr; the JSON is the last stdout line.
                    _ck_results[_ck_scenario, _ck_tag] = json.loads(
                        _ck_proc.stdout.strip().splitlines()[-1]
                    )
        for _ck_scenario in _CK_SCENARIOS:
            _ck_pure = _ck_results[_ck_scenario, "pure"]
            _ck_compiled = _ck_results[_ck_scenario, "compiled"]
            # Each child really took the path it was asked for -- an opt-in
            # that silently ran pure would make the comparison meaningless.
            assert _ck_pure["requested"] is False, _ck_pure
            assert _ck_pure["kernel_id"] is None, _ck_pure
            assert _ck_pure["provenance"] == _kernel_selector.PURE_PROVENANCE, (
                _ck_pure
            )
            assert _ck_compiled["requested"] is True, _ck_compiled
            assert _ck_compiled["kernel_id"] == _ck_expected_kernel_id, (
                _ck_compiled
            )
            assert _ck_compiled["provenance"] == _ck_expected_kernel_id, (
                _ck_compiled
            )
            # ...and the kernels were actually exercised. Without this the
            # state comparison could pass vacuously on a run that never solved.
            for _ck_tag, _ck_res in (
                ("pure", _ck_pure), ("compiled", _ck_compiled)
            ):
                assert _ck_res["steps"] == _ck_expected_steps[_ck_scenario], (
                    _ck_scenario, _ck_tag, _ck_res["steps"]
                )
                assert _ck_res["solve_enabled"] == 1.0, (_ck_scenario, _ck_tag)
                assert _ck_res["has_solution"] == 1.0, (_ck_scenario, _ck_tag)
                assert _ck_res["beam_csda_active"] == 1.0, (
                    _ck_scenario, _ck_tag
                )
                if _ck_scenario == "initial_profile":
                    # The shaped fill was really the initial condition: a
                    # uniform nn0 gives a spread of exactly zero, which would
                    # make the bit-identity below a mean-field comparison
                    # under another name.
                    assert _ck_res["nn0_spread"] > 0.0, (
                        _ck_scenario, _ck_tag, _ck_res["nn0_spread"]
                    )
            # Bit-identical, not merely close: the compiled path is a faithful
            # transcription, so the raw state bytes must match exactly -- the
            # same standard the golden holds on the compiled path.
            assert _ck_compiled["y"] == _ck_pure["y"], (
                f"compiled and pure solver states differ at the bit level "
                f"({_ck_scenario})"
            )
            assert _ck_compiled["I_tot"] == _ck_pure["I_tot"], (
                _ck_scenario, _ck_compiled["I_tot"], _ck_pure["I_tot"]
            )
            assert _ck_compiled["phi_c"] == _ck_pure["phi_c"], (
                _ck_scenario, _ck_compiled["phi_c"], _ck_pure["phi_c"]
            )
            assert (
                _ck_compiled["tail_ledger_W"] == _ck_pure["tail_ledger_W"]
            ), (_ck_scenario, _ck_compiled["tail_ledger_W"],
                _ck_pure["tail_ledger_W"])
            assert _ck_compiled["nn0_spread"] == _ck_pure["nn0_spread"], (
                _ck_scenario, _ck_compiled["nn0_spread"],
                _ck_pure["nn0_spread"],
            )
            print(
                f"compiled-kernel equivalence [{_ck_scenario}]: ok "
                f"({_ck_compiled['provenance']}, {_ck_pure['steps']} steps, "
                f"{_ck_pure['nested_marches']} nested marches, "
                "final state bit-identical)"
            )


# --------------------------------------------------------------------
# mirror-compiled-equivalence
# --------------------------------------------------------------------
@_case("mirror-compiled-equivalence")
def _case_mirror_compiled_equivalence():
    """The mirror branch marches the same floats on the compiled CSDA march.

    Every leg the mirror branch adds -- the returning primary's legs, which
    hand their anomalous drag to the ray's bank through the march's
    withholding argument, and the mirror chains' walker legs -- is a
    ``deposit_beam`` march, so it takes the compiled CSDA kernel whenever the
    kernel is loaded. The mirror fixture corpus's arms
    (``scripts/verify/deposit_beam_mirror_reference.py``) are replayed in a
    pure child and in a compiled child, and every array must be bit-identical
    between them. Each child reports which path it took, and the compiled
    child's march count is positive, so the comparison is not vacuous. SKIPS
    on a checkout with no built extension, like the solver-level
    equivalence case.
    """
    try:
        import importlib as _mc_importlib

        _mc_module = _mc_importlib.import_module(
            "cablp.cathode._cathode_kernels_cy"
        )
    except ImportError:
        _mc_module = None
    if _mc_module is None:
        print(
            "mirror compiled equivalence: SKIPPED -- "
            "cablp.cathode._cathode_kernels_cy is not built "
            "(`python build_ext.py --inplace` enables it)"
        )
        return
    corpus_dir = Path(__file__).resolve().parents[2] / "verify"
    child = f'''
import hashlib
import json
import sys

sys.path.insert(0, {str(corpus_dir)!r})
import numpy as np

from cablp.cathode import beam_deposition as bd
from cablp.cathode import kernels as K
import deposit_beam_mirror_reference as R

_marches = [0]
_kernel = bd._CSDA_MARCH


def _counted(*args, **kwargs):
    _marches[0] += 1
    return _kernel(*args, **kwargs)


if _kernel is not None:
    bd._CSDA_MARCH = _counted
out = R._replay()
print(json.dumps({{
    "kernel_id": (
        None if K.COMPILED_KERNELS is None
        else str(K.COMPILED_KERNELS.KERNEL_ID)
    ),
    "kernel_marches": _marches[0],
    "arrays": {{
        key: hashlib.sha256(
            np.ascontiguousarray(value, dtype=float).tobytes()
        ).hexdigest()
        for key, value in sorted(out.items())
    }},
}}))
'''
    results = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        script = Path(tmpdir) / "mirror_compiled_child.py"
        script.write_text(child)
        for tag, optin in (("pure", None), ("compiled", "1")):
            env = dict(os.environ)
            if optin is None:
                env.pop(_kernel_selector.ENV_VAR, None)
            else:
                env[_kernel_selector.ENV_VAR] = optin
            proc = subprocess.run(
                [sys.executable, str(script)], env=env,
                capture_output=True, text=True,
            )
            assert proc.returncode == 0, (tag, proc.stderr[-2000:])
            results[tag] = json.loads(proc.stdout.strip().splitlines()[-1])
    pure, compiled = results["pure"], results["compiled"]
    assert pure["kernel_id"] is None, pure["kernel_id"]
    assert compiled["kernel_id"] == _mc_module.KERNEL_ID, compiled["kernel_id"]
    assert pure["kernel_marches"] == 0
    assert compiled["kernel_marches"] > 0
    assert sorted(pure["arrays"]) == sorted(compiled["arrays"])
    differing = [
        key for key in pure["arrays"]
        if pure["arrays"][key] != compiled["arrays"][key]
    ]
    assert not differing, differing
    print(
        "mirror compiled equivalence: ok "
        f"({compiled['kernel_id']}, {len(pure['arrays'])} arrays, "
        f"{compiled['kernel_marches']} compiled marches, bit-identical)"
    )
