"""Verify exact resolved LAPDSim1D configs without running campaign points.

The cases here are the config-complete production/campaign entry points. Their
canonical JSON hashes are reviewed snapshots: a changed default, precedence
rule, or driver choice fails before a solver run can silently inherit it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py

from cablp.solvers._sim1d import config_manifest, default_config, resolve_config

from baseline_sim1d import build_baseline_config
from compare_sim1d_es1 import FLAG_OVERRIDES, PARAM_OVERRIDES
from compare_sim1d_es1 import PRODUCTION_NX
from run_m6_point import ELECTRON_BIRTH_POLICY as M6_ELECTRON_BIRTH_POLICY
from run_mechanism_ladder import (
    ELECTRON_BIRTH_POLICY as LADDER_ELECTRON_BIRTH_POLICY,
    ES_OPERATING,
)


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "cablp"
    / "solvers"
    / "_sim1d"
    / "config_snapshots.json"
)


def _production_base():
    params, flags = default_config()
    params.update(PARAM_OVERRIDES)
    flags.update(FLAG_OVERRIDES)
    params["neutral_exchange_model"] = "knudsen"
    return params, flags


def config_cases():
    """Return exact resolved configs for every config-complete campaign driver."""
    compare_params, compare_flags = _production_base()

    golden_params, golden_flags = build_baseline_config()

    ladder_params, ladder_flags = _production_base()
    ladder_op = ES_OPERATING[1]
    ladder_params.update(
        {
            "nx": 120,
            "V_bank": ladder_op["V_bank"],
            "cathode_solver_model": "current_driven",
            "beam_deposition_model": "csda",
            "beam_anomalous_model": "quasilinear",
            "cathode_emission_profile": "gaussian",
            "cathode_warming_model": "power_balance",
            "Te_birth_ionization": LADDER_ELECTRON_BIRTH_POLICY,
            "T_s": ladder_op["Ts_standby_K"],
            "cathode_Ts_base_K": ladder_op["Ts_standby_K"],
            "cathode_heat_capacity_J_per_K": 120.0,
            "cathode_conduction_W_per_K": 1500.0,
            "cathode_emissivity": 0.7,
        }
    )

    m6_params, m6_flags = _production_base()
    m6_op = ES_OPERATING[1]
    m6_params.update(
        {
            # run_m6_point --nx default, promoted to the production resolution.
            "nx": PRODUCTION_NX,
            "V_bank": m6_op["V_bank"],
            "cathode_solver_model": "current_driven",
            "beam_deposition_model": "csda",
            "beam_anomalous_model": "quasilinear",
            "cathode_emission_profile": "gaussian",
            "cathode_warming_model": "power_balance",
            "T_s": m6_op["Ts_standby_K"],
            "cathode_Ts_base_K": m6_op["Ts_standby_K"],
            "cathode_heat_capacity_J_per_K": 120.0,
            # cathode_conduction_W_per_K is deliberately absent: --g-cond now
            # defaults to None and defers to the shared production config (7c).
            "cathode_emissivity": 0.7,
            "phi_wf": 2.869,
            "cathode_surface_model": "ads_des",
            "cathode_phiwf_clean_eV": 2.809,
            "cathode_cleaning_sigma_cm2": 3.5e-16,
            "cathode_cleaning_E_th_eV": 20.0,
            "Te_birth_ionization": M6_ELECTRON_BIRTH_POLICY,
            "gas_puff_mode": "square",
            "S_gp": 3649.84,
            "cathode_sample_smoothing": "presheath",
        }
    )

    return {
        "production_golden": resolve_config(golden_params, golden_flags),
        "compare_sim1d_es1": resolve_config(compare_params, compare_flags),
        "run_mechanism_ladder_es1_defaults": resolve_config(
            ladder_params, ladder_flags
        ),
        "run_m6_point_es1_sgp3649_defaults": resolve_config(
            m6_params, m6_flags
        ),
    }


def canonical_payload(params, flags):
    return json.dumps(
        {"params": params, "flags": flags},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_digest(params, flags):
    return hashlib.sha256(canonical_payload(params, flags).encode()).hexdigest()


def current_snapshots():
    manifest = config_manifest()
    return {
        "schema": "lapdsim1d-resolved-config-snapshots-v1",
        "manifest_sha256": hashlib.sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
        "parameter_count": len(manifest["parameters"]),
        "flag_count": len(manifest["flags"]),
        "cases": {
            name: {
                "sha256": config_digest(params, flags),
                "Te_birth_ionization": params["Te_birth_ionization"],
            }
            for name, (params, flags) in config_cases().items()
        },
    }


def verify_snapshots():
    expected = json.loads(SNAPSHOT_PATH.read_text())
    actual = current_snapshots()
    if actual != expected:
        raise AssertionError(
            "resolved LAPDSim1D config snapshots changed:\n"
            f"expected={json.dumps(expected, sort_keys=True, indent=2)}\n"
            f"actual={json.dumps(actual, sort_keys=True, indent=2)}"
        )
    return actual


def scan_h5_birth_metadata(root):
    """Return recorded birth selectors for every H5 below ``root``.

    This deliberately reports metadata only. It does not infer which driver
    produced a file or relabel an artifact from its filename.
    """
    root = Path(root)
    records = []
    for path in sorted(root.rglob("*.h5")):
        try:
            with h5py.File(path, "r") as h5:
                raw_params = h5.attrs.get("params_json")
                if raw_params is None:
                    recorded = None
                    status = "missing_params_json"
                else:
                    params = json.loads(raw_params)
                    recorded = params.get("Te_birth_ionization")
                    status = (
                        "recorded"
                        if "Te_birth_ionization" in params
                        else "missing_selector"
                    )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            recorded = None
            status = f"unreadable:{type(error).__name__}"
        records.append(
            {
                "path": str(path.relative_to(root)),
                "Te_birth_ionization": recorded,
                "status": status,
            }
        )
    counts = {}
    for record in records:
        key = (
            record["Te_birth_ionization"]
            if record["status"] == "recorded"
            else record["status"]
        )
        counts[str(key)] = counts.get(str(key), 0) + 1
    return {
        "schema": "lapdsim1d-h5-birth-metadata-audit-v1",
        "root": str(root),
        "artifact_count": len(records),
        "recorded_value_counts": counts,
        "artifacts": records,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--print-current",
        action="store_true",
        help="print current snapshot material for explicit review",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="print the complete machine-readable default manifest",
    )
    parser.add_argument(
        "--scan-h5",
        type=Path,
        help="print recorded Te_birth_ionization metadata below this directory",
    )
    args = parser.parse_args(argv)
    if args.print_manifest:
        print(json.dumps(config_manifest(), sort_keys=True, indent=2))
    if args.print_current:
        print(json.dumps(current_snapshots(), sort_keys=True, indent=2))
    if args.scan_h5 is not None:
        print(
            json.dumps(
                scan_h5_birth_metadata(args.scan_h5),
                sort_keys=True,
                indent=2,
            )
        )
    if not args.print_current and not args.print_manifest and args.scan_h5 is None:
        snapshots = verify_snapshots()
        print(
            "sim1d config snapshots OK: "
            f"params={snapshots['parameter_count']}, "
            f"flags={snapshots['flag_count']}, "
            f"cases={len(snapshots['cases'])}"
        )


if __name__ == "__main__":
    main()
