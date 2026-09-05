import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from cablp.cathode.kernels import PURE_PROVENANCE as PURE_KERNEL_PROVENANCE

from ..core.config import ConfigurationLineage, resolve_config
from ..core.timestep import TimestepDiagnostics
from ..physics.hot_neutrals import HOT_CHANNEL_DIAGNOSTIC_FIELDS
from ..physics.kinetic_dvm import LEDGER_PARTICLE_ROW_DOC
from ..physics.sources import (
    IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS,
)
from .cathode_diagnostics import CathodeDiagnostics


RESULT_VERSION = "sim1d-hdf5-v1"

#: What ``configuration_name`` says for a run that named no configuration.
#: It is a written-down absence, not a guess: an in-process configuration --
#: the golden builder's, the smoke suite's, a unit instrument's -- has no
#: committed file behind it and must not borrow one's name.
UNNAMED_CONFIGURATION = "<unnamed>"

#: The configuration-lineage root attributes, in the order they are written.
#: PRESENCE-GATED on read: a file written before 2026-09-03 carries none of
#: them and ``load_result_hdf5`` reports ``None`` for each, which means "this
#: file does not say", never "unnamed" and never a reconstructed value. Nothing
#: here is inferred from ``params_json``: a configuration's identity is a fact
#: about which file a run named, and two different files can resolve alike.
_CONFIGURATION_ATTRS = (
    "configuration_name",
    "configuration_base_chain",
    "configuration_file_sha256",
    "configuration_delta_keys",
    "configuration_identity",
)
#: The three of those whose stored form is a JSON list.
_CONFIGURATION_JSON_ATTRS = (
    "configuration_base_chain",
    "configuration_file_sha256",
    "configuration_delta_keys",
)

# The optional per-sample arrays a run may or may not carry. ``_write_arrays``
# skips the ones the result lacks and ``_read_arrays`` skips the ones the file
# lacks, so this list is additive: adding a name here cannot change any file a
# previous run wrote, and a file written before a name existed still loads with
# that attribute simply absent.
#
# HOT-CHANNEL DIAGNOSTICS, 2026-08-14. The hot channel's per-cell readings were
# computed and attached to the live result but were NOT in this list, so
# ``save_result_hdf5`` silently dropped them and every artifact written before
# this date is missing them. Those files are not migrated: absence means "never
# persisted", never zero.
_OPTIONAL_ARRAY_FIELDS = (
    # Present only when the run evolved them (the neutral_momentum /
    # neutral_two_zone / neutral_energy flags); readers tolerate their
    # absence, so the 5-field format is unchanged. With nn_a present, nn is
    # the COLUMN density. En / Tn ride the same volume as nn.
    "M_n",
    "u_n",
    "nn_a",
    "M_n_a",
    "u_n_a",
    "En",
    "Tn",
) + HOT_CHANNEL_DIAGNOSTIC_FIELDS + IONIZATION_BIRTH_DEFICIT_DIAGNOSTIC_FIELDS

_ARRAY_FIELDS = (
    "time",
    "phase",
    "phase_elapsed",
    "phase_cathode_enabled",
    "phase_gas_puff_enabled",
    "phase_floating",
    "y",
    "n",
    "nn",
    "M",
    "momentum",
    "Ee",
    "Ei",
    "u",
    "Te",
    "Ti",
    "pe",
    "pi",
    "p",
) + _OPTIONAL_ARRAY_FIELDS

# LABEL-SEMANTICS BOUNDARY, 2026-08-05. ``diagnostics/active_constraint`` now
# always names the bound that actually minimized. Files written BEFORE this
# date instead carry the literal "dt_min" on every step whose timestep was
# clamped up to ``dt_min``, overwriting the true bound's name. Those files are
# historical records and are NOT migrated: a "dt_min" entry in one of them
# means "this step was clamped, by an unrecorded bound". From this date the
# clamp is carried by the separate ``clamped_to_dt_min`` flag (0.0/1.0)
# alongside ``dt_raw``, the unclamped request. Both are defaulted fields, so
# older files still load, reading 0.0 and NaN respectively.


def _resolve_config_namespace(kind, supplied):
    """Resolve one namespace the way ``LAPDSim1D.__init__`` resolves it.

    ``resolve_config`` owns both namespaces; the untouched one falls back to
    its template and is discarded here, so ``params`` and ``flags`` stay
    independent.
    """
    if kind == "params":
        return resolve_config(params=supplied)[0]
    return resolve_config(flags=supplied)[1]


def _check_config_metadata(kind, supplied, constructed):
    """Raise unless ``supplied`` would construct ``constructed``'s config.

    Callers naturally hold the PRE-resolution override mapping they passed to
    ``LAPDSim1D``, while ``result.params``/``result.flags`` are the resolved
    config. Compare post-resolution so any input that would build the same
    solver config passes, and a genuinely different one still raises.
    """
    supplied = dict(supplied)
    constructed = dict(constructed)
    if supplied == constructed:
        return
    resolved_supplied = _resolve_config_namespace(kind, supplied)
    resolved_constructed = _resolve_config_namespace(kind, constructed)
    if resolved_supplied == resolved_constructed:
        return
    differing = sorted(
        key
        for key in set(resolved_supplied) | set(resolved_constructed)
        if resolved_supplied.get(key) != resolved_constructed.get(key)
    )
    raise ValueError(
        f"{kind} metadata differs from the constructed LAPDSim1D config "
        f"after resolution; differing keys: {differing}"
    )


def save_result_hdf5(path, result, params=None, flags=None):
    """Write a ``LAPDSim1D.run`` result namespace to an HDF5 file.

    Format ``sim1d-hdf5-v1``. Beside the trajectory arrays and the term
    ledgers the file carries PRESENCE-GATED groups, each written only by a run
    that produced it, so its absence means "never recorded" and never "zero":
    ``dvm_transfer_ledger`` (the DVM arm's deferred-transfer closure),
    ``dvm_particle_ledger`` (that arm's per-save PARTICLE ledger -- births by
    channel, losses, pump, end returns and inventory, one dataset per row and
    the row documentation in the group's own attributes), ``jet_arming``,
    ``atomic_rate_domain``, ``floor_ledger`` and the diagnostics groups.
    """
    result_params = getattr(result, "params", None)
    result_flags = getattr(result, "flags", None)
    if result_params is not None:
        if params is not None:
            _check_config_metadata("params", params, result_params)
        params = result_params
    if result_flags is not None:
        if flags is not None:
            _check_config_metadata("flags", flags, result_flags)
        flags = result_flags
    path = Path(path)
    with h5py.File(path, "w") as h5:
        h5.attrs["format"] = RESULT_VERSION
        h5.attrs["solver"] = "LAPDSim1D"
        h5.attrs["steps"] = int(result.steps)
        h5.attrs["final_time"] = float(result.final_time)
        h5.attrs["t_prebreakdown_trigger"] = float(
            getattr(result, "t_prebreakdown_trigger", np.nan)
        )
        h5.attrs["t_breakdown_trigger"] = float(
            getattr(result, "t_breakdown_trigger", np.nan)
        )
        # Present only on opt-in max_steps_action="stop" runs; default runs
        # never carry it, keeping their files byte-identical to before.
        run_status = getattr(result, "run_status", None)
        if run_status is not None:
            h5.attrs["run_status"] = str(run_status)
        # Kernel provenance (D3/D4): which arithmetic produced this
        # trajectory -- "pure" for the default Python path, otherwise the
        # compiled kernel module's own id. Written whenever the result carries
        # it, so no new artifact is anonymous about its kernel path; an
        # artifact with NO compiled_kernels attribute predates the selector
        # and is pure by construction.
        compiled_kernels = getattr(result, "compiled_kernels", None)
        if compiled_kernels is not None:
            h5.attrs["compiled_kernels"] = str(compiled_kernels)
        # Presence-gated execution identity. Ordinary runs and their files do
        # not acquire an identity implicitly; qualified Phase 3 runs supply it
        # before solver construction and preserve it here.
        run_id = getattr(result, "run_id", None)
        if run_id is not None:
            from .phase3_capture import validate_run_id

            h5.attrs["run_id"] = validate_run_id(run_id)
        # WHICH MEASURED DRIVE this run was given (cathode_solver_model =
        # "prescribed_measured"). PRESENCE-GATED both ways: a run on any other
        # solver model carries neither record and its file keeps exactly the
        # attributes it always had. The trace's sha256 is the digest of the
        # FILE's bytes, so the artifact names the measured product itself and
        # not merely a path that may since have been rewritten; the hand-off
        # attributes appear only once the switch actually happened, and their
        # absence on a prescribed run says the run ended inside its calibrated
        # foot.
        prescribed_drive = getattr(result, "prescribed_drive", None)
        if prescribed_drive is not None:
            h5.attrs["cathode_prescribed_trace_path"] = str(
                prescribed_drive["trace_path"]
            )
            h5.attrs["cathode_prescribed_trace_sha256"] = str(
                prescribed_drive["trace_sha256"]
            )
            h5.attrs["cathode_prescribed_t0_s"] = float(
                prescribed_drive["t0_s"]
            )
            h5.attrs["cathode_prescribed_start_s"] = float(
                prescribed_drive["start_s"]
            )
        prescribed_handoff = getattr(result, "prescribed_handoff", None)
        if prescribed_handoff is not None:
            h5.attrs["cathode_prescribed_handoff_time_s"] = float(
                prescribed_handoff["time_s"]
            )
            h5.attrs[
                "cathode_prescribed_handoff_current_calibrated_a"
            ] = float(prescribed_handoff["current_calibrated_A"])
            h5.attrs["cathode_prescribed_handoff_current_trace_a"] = float(
                prescribed_handoff["current_trace_A"]
            )
            h5.attrs["cathode_prescribed_handoff_relative_jump"] = float(
                prescribed_handoff["relative_jump"]
            )
        # WHICH CONFIGURATION this run named. ``params_json`` records what the
        # run resolved TO; this records what it WAS -- the configuration's own
        # name, what it was derived from, the bytes of each file in that chain,
        # the keys it moves, and the resolved configuration's identity. A file
        # that carries both can be matched back to a committed configuration
        # instead of only to a bag of values.
        #
        # ``configuration_name`` is always written, "<unnamed>" for a run that
        # named none (the golden builder, the smoke suite, the unit
        # instruments); the other four are PRESENCE-GATED on a lineage
        # existing, because an unnamed run has no chain, no files and no
        # declared deltas, and writing empty ones would read as facts.
        configuration = getattr(result, "configuration", None)
        h5.attrs["configuration_name"] = (
            UNNAMED_CONFIGURATION if configuration is None
            else str(configuration.name)
        )
        if configuration is not None:
            h5.attrs["configuration_base_chain"] = _json_dumps(
                list(configuration.base_chain)
            )
            h5.attrs["configuration_file_sha256"] = _json_dumps(
                list(configuration.file_sha256)
            )
            h5.attrs["configuration_delta_keys"] = _json_dumps(
                list(configuration.delta_keys)
            )
            h5.attrs["configuration_identity"] = str(configuration.identity)
        if params is not None:
            h5.attrs["params_json"] = _json_dumps(params)
        if flags is not None:
            h5.attrs["flags_json"] = _json_dumps(flags)

        _write_arrays(h5, result, _ARRAY_FIELDS)
        _write_geometry(h5.create_group("geometry"), result)
        _write_nested_fields(h5.create_group("rhs_terms"), result.rhs_terms)
        _write_term_arrays(
            h5.create_group("electron_energy_terms_W_cm3"),
            result.electron_energy_terms_W_cm3,
        )
        _write_term_arrays(
            h5.create_group("ion_energy_terms_W_cm3"),
            result.ion_energy_terms_W_cm3,
        )
        _write_field_arrays(h5.create_group("total_rhs"), result.total_rhs)
        if hasattr(result, "floor_ledger"):
            _write_field_arrays(
                h5.create_group("floor_ledger"), result.floor_ledger
            )
        # K2d transfer-ledger census. Written only by a run that built the
        # DVM arm, so a moment-model file's dataset layout is unchanged and
        # the absence of the group means "never recorded", never "zero".
        #
        # The group grows ADDITIVELY and is read by name, so a reader must
        # not assume a fixed key set. The exponential transfer hold added
        # ``transfer_hold`` (a string attr naming the selector),
        # ``{Ei,M}_hold_debt`` (per-cell) with their ``_total`` /
        # ``_max_abs`` scalars, and ``sample_{Ei,M}_hold_debt_total``; on a
        # file written before it, their absence means the run predates the
        # hold, i.e. it ran the zero-order hold with no hold debt to carry.
        # The residual the ``*_residual_rel`` scalars report changed meaning
        # with them: it is now ``applied + debt + hold_debt - booked``.
        dvm_ledger = getattr(result, "dvm_transfer_ledger", None)
        if dvm_ledger:
            _write_census(h5.create_group("dvm_transfer_ledger"), dvm_ledger)
        # The DVM arm's PARTICLE ledger, at save cadence. Presence-gated on
        # the same condition as the census above -- only a run that built the
        # arm carries the attribute -- so a moment-model file is unchanged by
        # this group's existence and its absence means "never recorded".
        #
        # Additive and read by name, like every group above it: it touches no
        # field-array schema, so by this file's own rule (see the
        # ``_OPTIONAL_ARRAY_FIELDS`` note and the ``jet_arming`` write below)
        # the format version does not move, and a file written before the
        # group existed still loads with the attribute simply absent.
        dvm_particle_ledger = getattr(result, "dvm_particle_ledger", None)
        if dvm_particle_ledger:
            _write_dvm_particle_ledger(
                h5.create_group("dvm_particle_ledger"), dvm_particle_ledger
            )
        # How many times the neutral clock ticked. Presence-gated on the same
        # condition as the census above -- only a run that built the DVM arm
        # carries the attribute -- so a moment-model file is BYTE-unchanged by
        # this dataset's existence. Absent means "never recorded", not zero.
        # Kept OUTSIDE the ledger group because it is a property of the run
        # rather than a row of the transfer census, and because that group is
        # written only when the census is non-empty.
        if hasattr(result, "dvm_tick_count"):
            h5.create_dataset(
                "dvm_tick_count", data=np.int64(result.dvm_tick_count)
            )
        # Cathode-jet arming census. Presence-gated on the same discipline as
        # the DVM census above -- only a run that DECLARED an arming criterion
        # carries the attribute -- so a file from a run at the inert default
        # is byte-unchanged by this group's existence, and absent means "no
        # criterion was declared", never "the latch never fired". Scalars, via
        # the shared census writer, so no field-array schema is touched and
        # the format version does not move.
        jet_arming = getattr(result, "jet_arming", None)
        if jet_arming:
            _write_census(h5.create_group("jet_arming"), jet_arming)
        if hasattr(result, "atomic_rate_domain"):
            _write_field_arrays(
                h5.create_group("atomic_rate_domain"),
                result.atomic_rate_domain,
            )
        if hasattr(result, "phase_events"):
            _write_field_arrays(h5.create_group("phase_events"), result.phase_events)
        if hasattr(result, "timestep_rejection_events"):
            _write_field_arrays(
                h5.create_group("timestep_rejection_events"),
                result.timestep_rejection_events,
            )
        if hasattr(result, "current_trigger_samples"):
            _write_field_arrays(
                h5.create_group("current_trigger_samples"),
                result.current_trigger_samples,
            )
        if hasattr(result, "cathode_diagnostics"):
            _write_field_arrays(
                h5.create_group("cathode_diagnostics"),
                result.cathode_diagnostics,
            )
        if hasattr(result, "ignition_diagnostics"):
            _write_field_arrays(
                h5.create_group("ignition_diagnostics"),
                result.ignition_diagnostics,
            )
        if hasattr(result, "gas_puff_diagnostics"):
            _write_field_arrays(
                h5.create_group("gas_puff_diagnostics"),
                result.gas_puff_diagnostics,
            )
        # Present only on a run whose cathode switch was opened by an ignition
        # guard; a normal run never carries it, so its file is unchanged.
        ignition_abort = getattr(result, "ignition_abort", None)
        if ignition_abort is not None:
            group = h5.create_group("ignition_abort")
            for key, value in ignition_abort.items():
                if isinstance(value, str):
                    group.attrs[key] = value
                else:
                    group.attrs[key] = float(value)
        _write_diagnostics(h5.create_group("diagnostics"), result.diagnostics)
    return path


def load_result_hdf5(path):
    """Load a ``save_result_hdf5`` file into a run-result namespace."""
    path = Path(path)
    with h5py.File(path, "r") as h5:
        file_format = h5.attrs.get("format", "")
        if _decode_string(file_format) != RESULT_VERSION:
            raise ValueError(
                f"unsupported sim1d result format {file_format!r}; "
                f"expected {RESULT_VERSION!r}"
            )

        arrays = _read_arrays(h5, _ARRAY_FIELDS)
        geometry = _read_arrays(
            h5["geometry"],
            (
                "z_cm",
                "length_cm",
                "Rp_cm",
                "Rm_cm",
                "plasma_volume_cm3",
                "neutral_volume_cm3",
                "volume_ratio",
                "plasma_active",
            ),
        )
        geometry["cell_role"] = _read_string_array(h5["geometry/cell_role"])
        params = _read_json_attr(h5, "params_json")
        flags = _read_json_attr(h5, "flags_json")

        result = SimpleNamespace(
            **arrays,
            **geometry,
            rhs_terms=_read_nested_fields(h5["rhs_terms"]),
            total_rhs=_read_field_arrays(h5["total_rhs"]),
            floor_ledger=(
                _read_field_arrays(h5["floor_ledger"])
                if "floor_ledger" in h5
                else {}
            ),
            atomic_rate_domain=(
                _read_field_arrays(h5["atomic_rate_domain"])
                if "atomic_rate_domain" in h5
                else {}
            ),
            phase_events=(
                _read_field_arrays(h5["phase_events"])
                if "phase_events" in h5
                else _empty_phase_events()
            ),
            timestep_rejection_events=(
                _read_field_arrays(h5["timestep_rejection_events"])
                if "timestep_rejection_events" in h5
                else _empty_timestep_rejection_events()
            ),
            current_trigger_samples=(
                _read_field_arrays(h5["current_trigger_samples"])
                if "current_trigger_samples" in h5
                else _empty_current_trigger_samples()
            ),
            # Wrapped so a read of one of the six retired circuit scalars
            # raises naming its successor instead of a bare KeyError or a
            # ``.get`` default. A file saved before the retirement carries the
            # datasets and reads normally; see results/cathode_diagnostics.py.
            cathode_diagnostics=CathodeDiagnostics(
                _read_field_arrays(h5["cathode_diagnostics"])
                if "cathode_diagnostics" in h5
                else {}
            ),
            # Defaulted dataset: runs saved before 2026-07-28 have no
            # ignition_diagnostics group and load with NaN/zero-defaulted
            # per-sample records of the right length.
            ignition_diagnostics=(
                _read_field_arrays(h5["ignition_diagnostics"])
                if "ignition_diagnostics" in h5
                else _empty_ignition_diagnostics(
                    len(_read_dataset(h5["time"]))
                )
            ),
            # Defaulted dataset: runs saved before 2026-07-29 have no
            # gas_puff_diagnostics group and load with NaN-defaulted
            # per-sample records of the right length.
            gas_puff_diagnostics=(
                _read_field_arrays(h5["gas_puff_diagnostics"])
                if "gas_puff_diagnostics" in h5
                else _empty_gas_puff_diagnostics(
                    len(_read_dataset(h5["time"]))
                )
            ),
            electron_energy_terms_W_cm3=_read_term_arrays(
                h5["electron_energy_terms_W_cm3"]
            ),
            ion_energy_terms_W_cm3=_read_term_arrays(
                h5["ion_energy_terms_W_cm3"]
            ),
            diagnostics=_read_diagnostics(h5["diagnostics"]),
            steps=int(h5.attrs["steps"]),
            final_time=float(h5.attrs["final_time"]),
            t_prebreakdown_trigger=_read_float_attr(
                h5,
                "t_prebreakdown_trigger",
                np.nan,
            ),
            t_breakdown_trigger=_read_float_attr(
                h5,
                "t_breakdown_trigger",
                np.nan,
            ),
            params=params,
            flags=flags,
            path=path,
        )
        # Configuration lineage, presence-gated attribute by attribute. Absent
        # reads as None -- "this file does not say" -- and nothing is inferred
        # from the file's other attributes.
        for _attr in _CONFIGURATION_ATTRS:
            if _attr not in h5.attrs:
                setattr(result, _attr, None)
            elif _attr in _CONFIGURATION_JSON_ATTRS:
                setattr(result, _attr, _read_json_attr(h5, _attr))
            else:
                setattr(result, _attr, _decode_string(h5.attrs[_attr]))
        # The lineage RECONSTRUCTED, so a load -> save round trip carries the
        # configuration through instead of dropping a named run to "<unnamed>"
        # (a re-save of a scored artifact is a routine step, and losing the
        # name there would be worse than never having written it). Built only
        # from a file that carries the whole record: a name alone is what an
        # unnamed run writes, and there is nothing to reconstruct from it.
        result.configuration = (
            None
            if any(
                getattr(result, _attr) is None
                for _attr in _CONFIGURATION_ATTRS
            )
            or result.configuration_name == UNNAMED_CONFIGURATION
            else ConfigurationLineage(
                name=result.configuration_name,
                base_chain=tuple(result.configuration_base_chain),
                file_sha256=tuple(result.configuration_file_sha256),
                delta_keys=tuple(result.configuration_delta_keys),
                identity=result.configuration_identity,
            )
        )
        if "run_status" in h5.attrs:
            result.run_status = _decode_string(h5.attrs["run_status"])
        # Absent on artifacts written before the kernel selector existed;
        # those ran the pure path, which is what the default resolves to.
        result.compiled_kernels = (
            _decode_string(h5.attrs["compiled_kernels"])
            if "compiled_kernels" in h5.attrs
            else PURE_KERNEL_PROVENANCE
        )
        if "run_id" in h5.attrs:
            from .phase3_capture import validate_run_id

            result.run_id = validate_run_id(_decode_string(h5.attrs["run_id"]))
        # Set only when the file carries it: a reader must be able to tell a
        # run whose census was never persisted from one whose census is zero.
        if "dvm_transfer_ledger" in h5:
            result.dvm_transfer_ledger = _read_census(h5["dvm_transfer_ledger"])
        if "dvm_particle_ledger" in h5:
            result.dvm_particle_ledger = _read_dvm_particle_ledger(
                h5["dvm_particle_ledger"]
            )
        if "dvm_tick_count" in h5:
            result.dvm_tick_count = int(h5["dvm_tick_count"][()])
        if "jet_arming" in h5:
            result.jet_arming = _read_census(h5["jet_arming"])
        if "ignition_abort" in h5:
            result.ignition_abort = {
                key: (
                    _decode_string(value)
                    if isinstance(value, (bytes, str))
                    else float(value)
                )
                for key, value in h5["ignition_abort"].attrs.items()
            }
        return result


def _write_arrays(group, owner, names):
    for name in names:
        if hasattr(owner, name):
            _write_array_dataset(group, name, getattr(owner, name))


def _write_geometry(group, result):
    numeric_names = (
        "z_cm",
        "length_cm",
        "Rp_cm",
        "Rm_cm",
        "plasma_volume_cm3",
        "neutral_volume_cm3",
        "volume_ratio",
        "plasma_active",
    )
    _write_arrays(group, result, numeric_names)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset(
        "cell_role",
        data=np.asarray(result.cell_role, dtype=object),
        dtype=str_dtype,
    )


def _write_nested_fields(group, terms):
    for term_name, fields in terms.items():
        term_group = group.create_group(term_name)
        _write_field_arrays(term_group, fields)


def _write_field_arrays(group, fields):
    for field_name, values in fields.items():
        _write_array_dataset(group, field_name, values)


def _write_array_dataset(group, name, values):
    values = np.asarray(values)
    if values.dtype.kind in {"U", "S", "O"}:
        group.create_dataset(
            name,
            data=np.asarray(values, dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    else:
        group.create_dataset(name, data=values)


def _write_census(group, census):
    """Write a scalar/array census mapping: arrays as datasets, scalars as attrs.

    Integer counts are written as integers so the round trip returns the count,
    not a float that a report would have to re-cast, and STRING entries (a
    census that records which selector produced it -- the DVM transfer
    ledger's ``transfer_hold``) are written as string attrs and read back as
    ``str``, not coerced through float.
    """
    for name, value in census.items():
        if isinstance(value, np.ndarray):
            group.create_dataset(name, data=np.asarray(value, dtype=float))
        elif isinstance(value, str):
            group.attrs[name] = value
        elif isinstance(value, (bool, int, np.integer)):
            group.attrs[name] = int(value)
        else:
            group.attrs[name] = float(value)


def _write_dvm_particle_ledger(group, ledger):
    """Write the DVM particle ledger: one dataset per row, plus its own docs.

    Every row is one value per save frame, in trajectory order, so the group
    is a rectangle aligned with the file's ``time`` dataset. The three string
    attributes are aligned by index and make the artifact self-describing:
    ``channels`` names the rows in the group's own order, ``channel_units``
    gives each row's unit, and ``channel_meanings`` says what each row counts.

    It gets its own writer rather than the shared census one because the
    census writer's rule is "arrays are datasets, everything else is a scalar
    attribute", and these documentation attributes are string LISTS, which
    that reader would try to read back as numbers. It keeps that writer's
    other rule: it writes the rows it is HANDED, in the order it is handed
    them, so re-saving a result loaded from a file that predates a row does
    not fail on the row that file never carried. A fresh run hands them in
    :data:`LEDGER_PARTICLE_FRAME_KEYS` order, which is therefore the order a
    new file carries.
    """
    str_dtype = h5py.string_dtype(encoding="utf-8")
    names = tuple(ledger)
    for name in names:
        group.create_dataset(name, data=np.asarray(ledger[name], dtype=float))
    group.attrs.create(
        "channels", np.asarray(names, dtype=object), dtype=str_dtype
    )
    for attr, index in (("channel_units", 0), ("channel_meanings", 1)):
        group.attrs.create(
            attr,
            np.asarray(
                [LEDGER_PARTICLE_ROW_DOC[name][index] for name in names],
                dtype=object,
            ),
            dtype=str_dtype,
        )


def _read_dvm_particle_ledger(group):
    """Read a :func:`_write_dvm_particle_ledger` group back into its rows.

    By NAME, over the rows the file actually holds, so a file written when the
    engine booked fewer channels loads with those rows simply absent rather
    than failing. The documentation attributes are not returned: they describe
    the artifact for a reader of the file, and the caller has the live
    declarations.
    """
    return {name: _read_dataset(dataset) for name, dataset in group.items()}


def _read_census(group):
    """Read a :func:`_write_census` group back into its scalar/array mapping."""
    census = {name: _read_dataset(dataset) for name, dataset in group.items()}
    for name, value in group.attrs.items():
        if isinstance(value, (str, bytes)):
            census[name] = (
                value.decode("utf-8") if isinstance(value, bytes) else value
            )
            continue
        value = np.asarray(value)
        census[name] = int(value) if value.dtype.kind in "iub" else float(value)
    return census


def _write_term_arrays(group, terms):
    for term_name, values in terms.items():
        group.create_dataset(term_name, data=np.asarray(values))


def _write_diagnostics(group, diagnostics):
    group.attrs["count"] = len(diagnostics)
    if not diagnostics:
        return
    field_names = tuple(diagnostics[0].__dataclass_fields__)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    for field_name in field_names:
        values = [getattr(diag, field_name) for diag in diagnostics]
        if isinstance(values[0], str):
            group.create_dataset(
                field_name,
                data=np.asarray(values, dtype=object),
                dtype=str_dtype,
            )
        else:
            group.create_dataset(field_name, data=np.asarray(values, dtype=float))


def _read_arrays(group, names):
    return {name: _read_dataset(group[name]) for name in names if name in group}


def _read_nested_fields(group):
    return {
        term_name: _read_field_arrays(term_group)
        for term_name, term_group in group.items()
    }


def _read_field_arrays(group):
    return {
        field_name: _read_dataset(dataset)
        for field_name, dataset in group.items()
    }


def _read_term_arrays(group):
    return {term_name: dataset[()] for term_name, dataset in group.items()}


def _read_dataset(dataset):
    if h5py.check_string_dtype(dataset.dtype) is not None:
        return _read_string_array(dataset)
    return dataset[()]


def _read_diagnostics(group):
    count = int(group.attrs.get("count", 0))
    if count == 0:
        return []
    diagnostics = []
    field_names = tuple(TimestepDiagnostics.__dataclass_fields__)
    loaded = {}
    defaults = {
        field_name: field.default
        for field_name, field in TimestepDiagnostics.__dataclass_fields__.items()
    }
    for field_name in field_names:
        if field_name in group:
            loaded[field_name] = group[field_name][()]
        else:
            loaded[field_name] = [defaults[field_name]] * count
    for i in range(count):
        kwargs = {}
        for field_name, values in loaded.items():
            value = values[i]
            if field_name in {
                "active_constraint",
                "phase",
                "step_cap",
                "rejection_reason",
            }:
                value = _decode_string(value)
            else:
                value = float(value)
            kwargs[field_name] = value
        diagnostics.append(TimestepDiagnostics(**kwargs))
    return diagnostics


def _read_string_array(dataset):
    return np.asarray([_decode_string(value) for value in dataset[()]], dtype=object)


def _read_json_attr(group, name):
    if name not in group.attrs:
        return None
    return json.loads(_decode_string(group.attrs[name]))


def _read_float_attr(group, name, default):
    if name not in group.attrs:
        return float(default)
    return float(group.attrs[name])


def _empty_phase_events():
    return {
        "time": np.asarray([], dtype=float),
        "phase": np.asarray([], dtype=object),
        "reason": np.asarray([], dtype=object),
    }


def _empty_ignition_diagnostics(samples):
    """Return NaN/zero-defaulted ignition diagnostics for a pre-2026-07-28 file."""
    from ..core.ignition import IGNITION_DIAGNOSTIC_FIELDS

    zero_fields = {"armed", "joint_negative", "stalled"}
    return {
        name: np.zeros(samples, dtype=float)
        if name in zero_fields
        else np.full(samples, np.nan, dtype=float)
        for name in IGNITION_DIAGNOSTIC_FIELDS
    }


def _empty_gas_puff_diagnostics(samples):
    """Return NaN-defaulted puff waveform records for a pre-2026-07-29 file.

    NaN, not zero: an old file does not record that the puff was off, it
    records nothing, and a reader must not read that silence as no fuel.
    """
    from ..physics.neutrals import GAS_PUFF_DIAGNOSTIC_FIELDS

    return {
        name: np.full(samples, np.nan, dtype=float)
        for name in GAS_PUFF_DIAGNOSTIC_FIELDS
    }


def _empty_current_trigger_samples():
    return {
        "time": np.asarray([], dtype=float),
        "I_tot": np.asarray([], dtype=float),
    }


def _empty_timestep_rejection_events():
    return {
        "time": np.asarray([], dtype=float),
        "attempted_dt": np.asarray([], dtype=float),
        "retry_index": np.asarray([], dtype=float),
        "reason": np.asarray([], dtype=object),
        "phase": np.asarray([], dtype=object),
        "active_constraint": np.asarray([], dtype=object),
    }


def _decode_string(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _json_dumps(value):
    return json.dumps(value, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
