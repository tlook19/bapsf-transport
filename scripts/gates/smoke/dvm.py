"""Smoke cases: the transient discrete-velocity neutral model and its exports."""

from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np

from cablp.solvers._sim1d import (
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    LAPDSim1D,
    default_config,
    load_result_hdf5,
)
from cablp.solvers._sim1d.physics.kinetic_dvm import (
    ledger_residual as kinetic_dvm_ledger_residual,
)

from ._harness import _case


# --------------------------------------------------------------------
# transient-dvm-neutrals-k2a
# --------------------------------------------------------------------
@_case(
    "transient-dvm-neutrals-k2a",
    historical_stance=True,
    provides=("kd_flags", "kd_params"),
)
def _case_transient_dvm_neutrals_k2a(p2z_flags, p2z_params, p2z_sim):
    # --- K2a transient DVM neutrals: the LIVE distribution arm on its own
    # neutral clock. Default off and presence-gated; once engaged it owns
    # every neutral row and the ion-side momentum/energy of the channels it
    # models, the saved nn IS the column zeroth moment, the particle ledger
    # closes to roundoff, and each unsupported combination is refused at
    # construction naming the offender.
    assert "neutral_kinetic_dvm_coupling" not in p2z_sim.rhs_terms()
    assert p2z_sim._dvm is None
    kd_params = dict(p2z_params)
    kd_params["neutral_model"] = "kinetic_dvm"
    # Smoke-scale clock: a few 1 ns steps must cross it.
    kd_params["neutral_kinetic_dvm_cadence_s"] = 2.0e-9
    kd_params["neutral_kinetic_dvm_nvz"] = 16
    kd_params["neutral_kinetic_dvm_nvp"] = 6
    kd_flags = dict(p2z_flags)
    kd_params["initial_neutral_state"] = "fill"

    # Every refusal, and the offender it must name.
    for kd_bad_params, kd_bad_flags, kd_offender in (
        # Arming ``neutral_momentum`` back on is no longer a refusal and
        # cannot be one: the flag is a MEMBER of the
        # ``neutral_model='kinetic_dvm'`` family and True is its config
        # default, so the model-preset resolver cannot tell "I chose True"
        # from "I left it alone" and clears it instead.
        # The reachable refusal is an EXPLICIT family conflict -- a member
        # whose default is already compatible, set to something the
        # selection refuses -- and the one collected error names the
        # complete member set, so it still names ``neutral_momentum``.
        (
            dict(kd_params, neutral_mesh_accommodation=True),
            kd_flags,
            "neutral_momentum",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_cadence_s=0.0),
            kd_flags,
            "neutral_kinetic_dvm_cadence_s",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_accommodation=-0.1),
            kd_flags,
            "neutral_kinetic_dvm_accommodation",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_nvz=17),
            kd_flags,
            "nvz",
        ),
        (
            dict(kd_params, neutral_kinetic_dvm_annulus_flights="chord"),
            kd_flags,
            "neutral_kinetic_dvm_annulus_flights",
        ),
        (
            dict(
                kd_params,
                neutral_kinetic_dvm_annulus_flights="bounded_chord",
                neutral_model="moment",
            ),
            kd_flags,
            "neutral_kinetic_dvm_annulus_flights",
        ),
    ):
        try:
            LAPDSim1D(dict(kd_bad_params), dict(kd_bad_flags))
        except ValueError as kd_error:
            assert kd_offender in str(kd_error), (kd_offender, str(kd_error))
        else:
            raise AssertionError(
                f"kinetic_dvm accepted an unsupported configuration "
                f"({kd_offender})"
            )

    # The resolver's other half: naming the selection on an OTHERWISE
    # UNTOUCHED default config is enough, because every member left at its
    # config default is set to the value the family requires. Before the
    # resolver this build was unconstructible -- the package ships each
    # member armed, and each one had to be cleared by hand.
    kd_bare_params, kd_bare_flags = default_config()
    kd_bare_params["neutral_model"] = "kinetic_dvm"
    kd_bare_sim = LAPDSim1D(kd_bare_params, kd_bare_flags)
    kd_bare_got_p, kd_bare_got_f = kd_bare_sim.get_config()
    for kd_space, kd_key, kd_required, _kd_why in (
        KINETIC_DVM_INCOMPATIBLE_DEFAULTS
    ):
        kd_got = (
            kd_bare_got_f if kd_space == "flags" else kd_bare_got_p
        ).get(kd_key)
        assert kd_got == kd_required, (kd_space, kd_key, kd_got, kd_required)

    kd_sim = LAPDSim1D(dict(kd_params), dict(kd_flags))
    assert kd_sim._dvm is not None
    # Tn consumption is its OWN switch and defaults off.
    # Pre-engagement: the coupling key exists and is all-zero, and the fluid
    # neutral rows are still live (the moment terms carry the fill).
    kd_pre = kd_sim.rhs_terms()
    assert "neutral_kinetic_dvm_coupling" in kd_pre
    assert np.all(kd_pre["neutral_kinetic_dvm_coupling"].M == 0.0)
    assert np.all(kd_pre["neutral_kinetic_dvm_coupling"].Ei == 0.0)
    assert np.any(kd_pre["neutral_sources"].nn != 0.0)

    for _ in range(6):
        kd_sim.advance_one_step(dt=1.0e-9)
    kd_dvm = kd_sim._dvm
    assert kd_sim._dvm_engaged and kd_dvm.updates >= 1
    assert np.all(np.isfinite(kd_dvm.f_c)) and np.all(np.isfinite(kd_dvm.f_a))
    assert np.all(kd_dvm.f_c >= 0.0) and np.all(kd_dvm.f_a >= 0.0)
    kd_state = kd_sim.state
    assert np.all(np.isfinite(kd_state.nn)) and np.all(np.isfinite(kd_state.nn_a))
    assert np.all(np.isfinite(kd_sim.rhs()))

    # Moment-consistency contract: the saved nn IS the column zeroth moment
    # (floored), nn_a the annulus moment. Exact equality, not a tolerance.
    kd_floor = kd_sim.floors["nn"]
    assert np.array_equal(
        kd_state.nn, np.maximum(kd_dvm.column_density(), kd_floor)
    )
    assert np.array_equal(
        kd_state.nn_a, np.maximum(kd_dvm.annulus_density(), kd_floor)
    )

    # Particle ledger: births minus losses closes to roundoff in both the
    # distribution and domain forms.
    kd_res = kinetic_dvm_ledger_residual(kd_dvm.last_ledger)
    assert abs(kd_res["distribution_rel"]) < 1.0e-12, kd_res
    assert abs(kd_res["domain_rel"]) < 1.0e-12, kd_res

    # Transfer antisymmetry: the fluid coupling rows ARE minus the measured
    # kinetic moments -- the M*Vp + M_n*Vm == 0 discipline, extended to a
    # velocity-resolved neutral state.
    kd_terms = kd_sim.rhs_terms()
    kd_active = np.asarray(kd_sim.geometry.plasma_active, dtype=bool)
    kd_coupling = kd_terms["neutral_kinetic_dvm_coupling"]

    def kd_expected(values):
        # The coupling term is plasma-coupled, so it takes the same
        # dead-cell mask as every other plasma term.
        return np.where(kd_active, values, 0.0)

    assert np.array_equal(
        np.asarray(kd_coupling.M, dtype=float), kd_expected(kd_dvm.M_transfer)
    )
    assert np.array_equal(
        np.asarray(kd_coupling.Ei, dtype=float), kd_expected(kd_dvm.Ei_transfer)
    )

    # Supersession: every neutral row is zeroed, the superseded ion-transfer
    # rows are zeroed, and the particle / electron rows keep their forms.
    for kd_name, kd_term in kd_terms.items():
        if kd_name == "neutral_kinetic_dvm_coupling":
            continue
        assert np.all(np.asarray(kd_term.nn, dtype=float) == 0.0), kd_name
        if kd_term.nn_a is not None:
            assert np.all(np.asarray(kd_term.nn_a, dtype=float) == 0.0), kd_name
    for kd_name in ("ionization_birth", "recombination_rad_loss"):
        assert np.all(kd_terms[kd_name].M == 0.0), kd_name
        assert np.all(kd_terms[kd_name].Ei == 0.0), kd_name
    assert np.any(kd_terms["ionization_birth"].n != 0.0)

    # Rejected attempts mutate nothing: the distribution, the lagged end
    # buffers and the coupling accumulators are bit-identical afterwards.
    kd_before = (
        kd_dvm.f_c.tobytes(),
        kd_dvm.f_a.tobytes(),
        kd_dvm.pend_L_c.tobytes(),
        kd_dvm.pend_R_c.tobytes(),
        kd_dvm.M_transfer.tobytes(),
        kd_dvm.Ei_transfer.tobytes(),
        kd_dvm.updates,
    )
    kd_sim._attempt_step(dt=1.0e-9)
    kd_sim._attempt_step(dt=1.0e-13)
    kd_sim.rhs(y=kd_sim._y * 1.01)
    assert kd_before == (
        kd_dvm.f_c.tobytes(),
        kd_dvm.f_a.tobytes(),
        kd_dvm.pend_L_c.tobytes(),
        kd_dvm.pend_R_c.tobytes(),
        kd_dvm.M_transfer.tobytes(),
        kd_dvm.Ei_transfer.tobytes(),
        kd_dvm.updates,
    )

    # K2e counted-particle ionization handshake. The arm debits the count
    # the PLASMA booked over the tick, so no neutral becomes an ion without
    # leaving the kinetic state -- the identity below is the whole point,
    # and it is a conservation law, not a tolerance.
    kd_resid = (
        kd_dvm.ion_removed_cum + kd_dvm.ion_debt - kd_dvm.ion_booked_cum
    )
    kd_ion_scale = max(float(np.max(np.abs(kd_dvm.ion_booked_cum))), 1e-300)
    assert float(np.max(np.abs(kd_resid))) / kd_ion_scale < 1.0e-12, kd_resid
    assert np.any(kd_dvm.ion_booked_cum > 0.0)
    # Nothing was withheld on a healthy tick, so the debit IS the booking.
    assert kd_dvm.ion_shortfall_updates == 0
    assert np.allclose(
        kd_dvm.ion_removed_cum, kd_dvm.ion_booked_cum, rtol=1.0e-12, atol=0.0
    )
    # The pending booking is attempt-local: the rejected attempts above did
    # not add to it, and it is cleared at every tick.
    assert np.all(np.isfinite(kd_sim._dvm_ion_booked))
    assert kd_sim._dvm_ion_stage_accum is None
    assert kd_sim._dvm_ion_stage_weight == 0.0
    # A standalone update -- no partner, no booked count -- leaves the
    # march's own tally standing and books nothing into the handshake. Run
    # against a snapshot so the arm this sim carries is put back bit for
    # bit, which also exercises the snapshot's new ledger fields.
    kd_snap = kd_dvm.snapshot()
    kd_bare_led = kd_dvm.update(
        1.0e-9,
        n_i=np.asarray(kd_sim.state.n, dtype=float),
        Ti_eV=np.asarray(kd_sim.derived.Ti, dtype=float),
        u_i=np.asarray(kd_sim.derived.u, dtype=float),
        nu_ion=np.full(kd_dvm.nz, 1.0e4),
    )
    assert kd_bare_led["ion_booked"] == 0.0
    assert kd_bare_led["ion_limited_cells"] == 0.0
    assert kd_bare_led["loss_ionization"] > 0.0
    assert np.array_equal(kd_dvm.ion_booked_cum, kd_snap["ion_booked_cum"])
    assert np.array_equal(kd_dvm.ion_removed_cum, kd_snap["ion_removed_cum"])
    kd_dvm.restore(kd_snap)
    assert np.array_equal(kd_dvm.f_c, kd_snap["f_c"])
    assert np.array_equal(kd_dvm.ion_debt, kd_snap["ion_debt"])

    # The DVM's Tn moment stays an in-process DIAGNOSTIC: it is computed
    # whenever the arm is on and consumed by nothing. Its one consumer -- the
    # presheath collisionality behind the legacy volumetric absorber, reached
    # only through neutral_kinetic_dvm_tn_feedback -- was retired with that
    # absorber (see commit 1fc05c9), so the consumption A/B that used to sit
    # here has no operand left. The moment itself is still asserted finite
    # and positive by the census block above.
    return locals()


# --------------------------------------------------------------------
# dvm-particle-ledger-export
# --------------------------------------------------------------------
@_case("dvm-particle-ledger-export", historical_stance=True)
def _case_dvm_particle_ledger_export(kd_flags, kd_params):
    # The engine computes a per-tick PARTICLE ledger and the solver kept only
    # the last one, so the far-end column neutrals could not be attributed to
    # the end wall face, the puff or the wall return from any saved run. The
    # export books those counts at save cadence. Four statements: the moment
    # path writes no such group at all, a DVM run carries every declared row
    # with finite values, the rows are the ENGINE's own numbers rather than a
    # re-derivation (the internal channels' partners still cancel and every
    # tick lands in exactly one frame), and the group round-trips through the
    # file carrying its own documentation.
    from cablp.solvers._sim1d.physics.kinetic_dvm import (
        LEDGER_ENERGY_BIRTH_CHANNELS as _PL_E_BIRTH_CHANNELS,
        LEDGER_ENERGY_BIRTH_KEYS as _PL_E_BIRTH_KEYS,
        LEDGER_FLIGHT_CELL_KEY as _PL_FLIGHT_KEY,
        LEDGER_PARTICLE_FLOW_KEYS as _PL_FLOW_KEYS,
        LEDGER_SAVED_FRAME_KEYS as _PL_FRAME_KEYS,
        LEDGER_PARTICLE_ROW_DOC as _PL_ROW_DOC,
    )
    from cablp.solvers._sim1d.results.io import (
        save_result_hdf5 as _save_result_hdf5_pl,
    )

    # The K2d observation geometry: the shipped end wall block is a 7.8 cm
    # cell, which this case's fixed dt = 1 ns steps cannot afford.
    pl_params = dict(kd_params)
    pl_params.update(
        {
            "Lm": 2000.0,
            "plenum_length_cm": 100.0,
            "end_wall_length_cm": 100.0,
            "gas_puff_z_cm": 60.0,
            "Rp": 15.0,
            "R_cath": 15.0,
            "Rcs": 40.0,
            "Lcs": 25.0,
            "Rsup": 0.0,
            "end_expansion_cells": 10,
            "end_expansion_machine_radius_cm": 100.0,
            "end_expansion_plasma_radius_cm": 15.0,
            "cathode_anode_gap_cm": 50.0,
            "source_region_length_cm": 100.0,
            "source_region_dz_cm": 10.0,
            "dt_save": 5.0e-9,
        }
    )
    pl_flags = dict(kd_flags)
    pl_flags["end_expansion_geometry"] = True
    pl_flags["source_fixed_grid"] = True

    # The moment control, differing from the DVM build below in neutral_model
    # and nothing else, so a layout difference between the two files can be
    # nothing else either.
    pl_mom_params = dict(pl_params)
    pl_mom_params["neutral_model"] = "moment"
    pl_mom_sim = LAPDSim1D(pl_mom_params, dict(pl_flags))
    assert pl_mom_sim._dvm is None
    pl_mom_result = pl_mom_sim.run(t_end=2.0e-8, dt=1.0e-9)
    assert not hasattr(pl_mom_result, "dvm_particle_ledger")

    pl_sim = LAPDSim1D(dict(pl_params), dict(pl_flags))
    for _ in range(8):
        pl_sim.advance_one_step(dt=1.0e-9)
    assert pl_sim._dvm_engaged
    # Ticks fired before the first save frame of the run below, which is
    # exactly what the accumulator has to carry into it.
    pl_pre_ticks = pl_sim._dvm_tick_count
    assert pl_pre_ticks > 0
    pl_result = pl_sim.run(t_end=pl_sim.time + 4.0e-8, dt=1.0e-9)
    pl = pl_result.dvm_particle_ledger

    # Every declared row, and nothing else. This arm runs the default
    # ``rates`` annulus closure, which computes no flight landings, so the
    # per-cell row is ABSENT here -- the armed direction is below.
    assert set(pl) == set(_PL_FRAME_KEYS)
    assert set(_PL_ROW_DOC) == set(_PL_FRAME_KEYS) | {_PL_FLIGHT_KEY}
    assert _PL_FLIGHT_KEY not in pl
    assert set(_PL_E_BIRTH_KEYS) <= set(_PL_FRAME_KEYS)
    pl_saves = len(pl_result.time)
    assert pl_saves > 1
    for pl_name in _PL_FRAME_KEYS:
        pl_row = pl[pl_name]
        assert pl_row.shape == (pl_saves,), pl_name
        assert np.all(np.isfinite(pl_row)), pl_name
    # At save cadence, on the file's own clock.
    assert np.array_equal(pl["time"], pl_result.time)
    # EVERY tick lands in exactly one frame -- including the ones that fired
    # before the run's first save -- so the flow rows partition the run
    # rather than sampling it.
    assert pl["ticks"].sum() == pl_result.dvm_tick_count
    assert pl["ticks"].sum() > pl_pre_ticks
    assert np.all(pl["ticks"] >= 0.0)
    # The ENGINE's numbers, not a re-derivation: an internal channel is a
    # loss and its equal-and-opposite birth, and the pair still cancels after
    # the summing.
    assert np.array_equal(pl["birth_mesh_reemit"], pl["loss_mesh_blocked"])
    assert np.allclose(
        pl["birth_wall_accommodated"] + pl["birth_wall_reflected"],
        pl["loss_wall"],
        rtol=1.0e-12,
        atol=0.0,
    )
    # The state rows are read AT the frame, not summed over it.
    assert pl["inventory"][-1] == pl_sim._dvm.total_inventory()
    assert pl["inventory"][-1] > 0.0
    # THE ENERGY THE BIRTHS ARRIVED WITH, beside the counts. Two channels
    # are a counted number times a FIXED spectrum's mean, so the energy row
    # has to be that product -- which is the check that these are the
    # engine's own ledger rows and not a re-derivation. Sums over ticks in
    # the two orders agree to rounding, not to the bit.
    for pl_e_name, pl_e_count, pl_e_mean in (
        ("energy_birth_mesh_reemit", "birth_mesh_reemit",
         pl_sim._dvm.E_wall_mean),
        ("energy_birth_puff", "birth_puff", pl_sim._dvm.E_cold_mean),
    ):
        assert np.allclose(
            pl[pl_e_name], pl_e_mean * pl[pl_e_count],
            rtol=1.0e-12, atol=0.0,
        ), pl_e_name
    # An arriving stream carries energy exactly where it carries atoms.
    for pl_e_channel in _PL_E_BIRTH_CHANNELS:
        pl_e_row = pl[f"energy_birth_{pl_e_channel}"]
        assert np.all(pl_e_row >= 0.0), pl_e_channel
        assert np.all(
            (pl_e_row > 0.0) <= (pl[f"birth_{pl_e_channel}"] > 0.0)
        ), pl_e_channel
    assert pl["energy_birth_wall_accommodated"].sum() > 0.0

    with tempfile.TemporaryDirectory() as pl_dir:
        pl_mom_path = Path(pl_dir) / "dvm_particle_moment.h5"
        _save_result_hdf5_pl(pl_mom_path, pl_mom_result)
        with h5py.File(pl_mom_path, "r") as pl_mom_h5:
            assert "dvm_particle_ledger" not in pl_mom_h5
        assert not hasattr(
            load_result_hdf5(pl_mom_path), "dvm_particle_ledger"
        )

        pl_path = Path(pl_dir) / "dvm_particle.h5"
        _save_result_hdf5_pl(
            pl_path, pl_result, params=pl_params, flags=pl_flags
        )
        with h5py.File(pl_path, "r") as pl_h5:
            pl_group = pl_h5["dvm_particle_ledger"]
            assert set(pl_group) == set(_PL_FRAME_KEYS)
            # The group documents itself: three aligned string attributes in
            # the group's own row order, so the artifact is readable without
            # this repo.
            pl_channels = tuple(pl_group.attrs["channels"])
            assert pl_channels == tuple(_PL_FRAME_KEYS)
            for pl_attr, pl_index in (
                ("channel_units", 0),
                ("channel_meanings", 1),
            ):
                pl_doc = list(pl_group.attrs[pl_attr])
                assert pl_doc == [
                    _PL_ROW_DOC[pl_name][pl_index] for pl_name in pl_channels
                ], pl_attr
            pl_units = dict(zip(pl_channels, pl_group.attrs["channel_units"]))
            # Every flow row is a count of atoms; the two rows that are not
            # are the frame's own time and its tick count.
            assert all(
                pl_units[pl_name] == "atoms" for pl_name in _PL_FLOW_KEYS
            )
            assert all(
                pl_units[pl_name] == "erg" for pl_name in _PL_E_BIRTH_KEYS
            )
            assert pl_units["time"] == "s"
            assert pl_units["ticks"] == "ticks"

        pl_back = load_result_hdf5(pl_path).dvm_particle_ledger
        assert set(pl_back) == set(pl)
        for pl_name, pl_row in pl.items():
            assert np.array_equal(pl_back[pl_name], pl_row), pl_name

    # PRESENCE GATE, armed side: the bounded-chord flight transport is the
    # only closure that lands annulus atoms in the column cell by cell, and
    # it is the only one that carries the per-cell row. Same build otherwise.
    pl_fl_params = dict(pl_params)
    pl_fl_params["neutral_kinetic_dvm_annulus_flights"] = "bounded_chord"
    pl_fl_sim = LAPDSim1D(pl_fl_params, dict(pl_flags))
    assert pl_fl_sim._dvm.flights is not None
    for _ in range(8):
        pl_fl_sim.advance_one_step(dt=1.0e-9)
    assert pl_fl_sim._dvm_engaged
    pl_fl_result = pl_fl_sim.run(t_end=pl_fl_sim.time + 4.0e-8, dt=1.0e-9)
    pl_fl = pl_fl_result.dvm_particle_ledger
    assert set(pl_fl) == set(_PL_FRAME_KEYS) | {_PL_FLIGHT_KEY}
    pl_fl_row = pl_fl[_PL_FLIGHT_KEY]
    assert pl_fl_row.shape == (
        len(pl_fl_result.time), pl_fl_sim.geometry.cells
    )
    assert np.all(np.isfinite(pl_fl_row))
    assert np.all(pl_fl_row >= 0.0)
    assert pl_fl_row.sum() > 0.0
    with tempfile.TemporaryDirectory() as pl_fl_dir:
        pl_fl_path = Path(pl_fl_dir) / "dvm_particle_flight.h5"
        _save_result_hdf5_pl(
            pl_fl_path, pl_fl_result, params=pl_fl_params, flags=pl_flags
        )
        with h5py.File(pl_fl_path, "r") as pl_fl_h5:
            pl_fl_group = pl_fl_h5["dvm_particle_ledger"]
            # Appended LAST, so the row order an unarmed run writes is a
            # prefix of this one.
            assert tuple(pl_fl_group.attrs["channels"]) == (
                tuple(_PL_FRAME_KEYS) + (_PL_FLIGHT_KEY,)
            )
            assert pl_fl_group[_PL_FLIGHT_KEY].shape == pl_fl_row.shape
        assert np.array_equal(
            load_result_hdf5(pl_fl_path).dvm_particle_ledger[_PL_FLIGHT_KEY],
            pl_fl_row,
        )

    # The ARMING criterion's censored share, in ATOMS. A step COUNT cannot
    # stand in for it -- the censored steps are the low-current ones and
    # carry far less recycle each -- so the split is counted off the same
    # booking the jet is split from, on the same latch reading. The jet needs
    # a cathode solve for its launch energy, so this rides its own minimal
    # armed build rather than the coupling-free one above.
    pl_ja_params, pl_ja_flags = default_config()
    for pl_ja_space, pl_ja_key, pl_ja_value, _pl_ja_why in (
        KINETIC_DVM_INCOMPATIBLE_DEFAULTS
    ):
        (pl_ja_flags if pl_ja_space == "flags" else pl_ja_params)[
            pl_ja_key
        ] = pl_ja_value
    pl_ja_params["initial_neutral_state"] = "fill"
    pl_ja_params.update({
        "neutral_model": "kinetic_dvm",
        # Smoke-scale clock, and a velocity grid PINNED wide enough to carry
        # the launch band this jet's coefficients can produce -- the shipped
        # nvz/nvp are kept, because narrowing them narrows the grid-tied
        # launch smear and the band's low end stops projecting.
        "neutral_kinetic_dvm_cadence_s": 2.0e-9,
        "neutral_kinetic_dvm_vmax_cm_s": 3.0e7,
        "neutral_kinetic_dvm_cathode_jet": True,
        "neutral_jet_disarm_current_A": 0.0,
    })
    # Two runs that differ ONLY in the arm threshold: one that can never arm
    # and one that arms at once.
    pl_ja_split = {}
    for pl_ja_name, pl_ja_arm in (("censored", 1.0e30), ("launched", 1.0e-30)):
        pl_ja_sim = LAPDSim1D(
            dict(pl_ja_params, neutral_jet_arm_current_A=pl_ja_arm),
            dict(pl_ja_flags),
        )
        assert pl_ja_sim._jet_arming_active
        pl_ja_split[pl_ja_name] = pl_ja_sim.run(
            t_end=2.0e-8, dt=1.0e-9
        ).jet_arming
    for pl_ja_name, pl_ja in pl_ja_split.items():
        pl_ja_total = (
            pl_ja["recycle_launched_atoms"] + pl_ja["recycle_censored_atoms"]
        )
        # NON-VACUOUS: there was a recycle stream to split in the first place.
        assert pl_ja_total > 0.0, pl_ja_name
        assert np.isclose(
            pl_ja["recycle_censored_fraction"],
            pl_ja["recycle_censored_atoms"] / pl_ja_total,
            rtol=1.0e-12, atol=0.0,
        ), pl_ja_name
    # The share follows the LATCH, not the step count: an unreachable arm
    # threshold censors the whole stream, an immediate one censors none of
    # it -- and both runs book the SAME recycle, which is what makes the two
    # halves a partition of one quantity rather than two measurements.
    assert pl_ja_split["censored"]["recycle_launched_atoms"] == 0.0
    assert pl_ja_split["censored"]["recycle_censored_fraction"] == 1.0
    assert pl_ja_split["launched"]["recycle_censored_atoms"] == 0.0
    assert pl_ja_split["launched"]["recycle_censored_fraction"] == 0.0
    assert np.isclose(
        pl_ja_split["launched"]["recycle_launched_atoms"],
        pl_ja_split["censored"]["recycle_censored_atoms"],
        rtol=1.0e-9, atol=0.0,
    )
    # And the step count is NOT the share: the immediate-arm run censors a
    # step that carried no recycle at all, which is exactly the substitution
    # the atom counts exist to refuse.
    assert pl_ja_split["launched"]["censored_steps"] > 0
    # PRESENCE GATE: no counted recycle stream, no share to report. A moment
    # run with the same criterion carries the latch census and none of the
    # three atom rows, so absence says "nothing counted this" rather than a
    # zero saying "nothing was censored".
    pl_ja_mom_params = dict(pl_ja_params)
    pl_ja_mom_params["neutral_model"] = "moment"
    pl_ja_mom_params["neutral_jet_arm_current_A"] = 1.0e30
    pl_ja_mom_params["neutral_kinetic_dvm_cathode_jet"] = False
    pl_ja_mom = LAPDSim1D(pl_ja_mom_params, dict(pl_ja_flags)).run(
        t_end=1.0e-8, dt=1.0e-9
    ).jet_arming
    assert "censored_steps" in pl_ja_mom
    assert "recycle_censored_fraction" not in pl_ja_mom
    assert "recycle_launched_atoms" not in pl_ja_mom


# --------------------------------------------------------------------
# dvm-neutral-moment-export
# --------------------------------------------------------------------
@_case("dvm-neutral-moment-export", historical_stance=True)
def _case_dvm_neutral_moment_export(kd_flags, kd_params):
    # THE NEUTRAL GAS'S OWN FLOW, at save cadence. The only velocity-shaped
    # neutral row a saved DVM run carried was the transfer ledger's
    # ``sample_u_n_eff``, which is the mean velocity of the neutrals LOST to
    # collisions in the column -- not the gas flow -- so a z-t map of the
    # neutral flow could not be drawn from any saved run. Four statements:
    # the rows ARE the quadrature (an independent direct sum over the bins
    # reproduces the exported flux), a known drifting Maxwellian comes back
    # as its own drift and temperature inside the GRID's quadrature error,
    # every value is finite and is zero exactly where the floor convention
    # says, and the group round-trips while a file without it still loads.
    #
    # The UNSIGNED flux row rides all four. The gas is near-isotropic, so the
    # signed flux is a cancellation to the summation's roundoff floor over
    # most cells and a reader cannot tell a resolved drift from noise without
    # the traffic it cancelled out of: the unsigned row is that denominator,
    # so it is pinned against its own independent direct sum (which, having
    # no cancellation, admits the row-relative statement the signed flux
    # cannot support), it must bound the signed flux everywhere and coincide
    # with it on a one-sided distribution, it is non-negative and zero on an
    # empty zone, and a file written before it existed must still load.
    from cablp.solvers._sim1d.physics.kinetic_dvm import (
        NEUTRAL_MOMENT_QUANTITY_DOC as _NM_QUANTITY_DOC,
        NEUTRAL_MOMENT_ROW_DOC as _NM_ROW_DOC,
        NEUTRAL_MOMENT_SAVED_FRAME_KEYS as _NM_KEYS,
        NEUTRAL_MOMENT_ZONES as _NM_ZONES,
        _axial_moments as _nm_axial_moments,
        _directional_temperatures_eV as _nm_directional_T_eV,
    )
    from cablp.solvers._sim1d.physics.kinetic_neutrals import (
        EV as _NM_EV,
        M_HE as _NM_M_HE,
    )
    from cablp.solvers._sim1d.results.io import (
        save_result_hdf5 as _save_result_hdf5_nm,
    )

    nm_params = dict(kd_params)
    nm_params["dt_save"] = 5.0e-9
    nm_flags = dict(kd_flags)

    # The moment control, differing in neutral_model and nothing else: the
    # group is presence-gated on the arm, so a moment run must carry neither
    # the attribute nor the group.
    nm_mom_params = dict(nm_params)
    nm_mom_params["neutral_model"] = "moment"
    nm_mom_sim = LAPDSim1D(nm_mom_params, dict(nm_flags))
    assert nm_mom_sim._dvm is None
    nm_mom_result = nm_mom_sim.run(t_end=2.0e-8, dt=1.0e-9)
    assert not hasattr(nm_mom_result, "dvm_neutral_moments")

    nm_sim = LAPDSim1D(dict(nm_params), dict(nm_flags))
    nm_result = nm_sim.run(t_end=4.0e-8, dt=1.0e-9)
    assert nm_sim._dvm_engaged
    nm = nm_result.dvm_neutral_moments

    # Every declared row, and nothing else. Both zones, because the kinetic
    # path carries a distribution for each.
    assert set(nm) == set(_NM_KEYS)
    assert set(_NM_ROW_DOC) == set(_NM_KEYS)
    assert set(_NM_KEYS) == {"time"} | {
        f"{nm_zone}_{nm_q}"
        for nm_zone in _NM_ZONES
        for nm_q in _NM_QUANTITY_DOC
    }
    # The unsigned row sits immediately after the signed one it is the
    # denominator for, within each zone.
    for nm_zone in _NM_ZONES:
        assert _NM_KEYS.index(f"{nm_zone}_abs_flux_n_z") == (
            _NM_KEYS.index(f"{nm_zone}_flux_n_z") + 1
        ), nm_zone
    nm_frames = len(nm_result.time)
    nm_cells = nm_sim.geometry.cells
    assert nm_frames > 1
    # (iii) FINITE EVERYWHERE -- no NaN or inf reaches a saved file -- and
    # per cell on every row but the frame's own clock.
    assert nm["time"].shape == (nm_frames,)
    assert np.array_equal(nm["time"], nm_result.time)
    for nm_name in _NM_KEYS:
        if nm_name == "time":
            continue
        assert nm[nm_name].shape == (nm_frames, nm_cells), nm_name
        assert np.all(np.isfinite(nm[nm_name])), nm_name
    for nm_zone in _NM_ZONES:
        assert np.all(nm[f"{nm_zone}_n_n"] >= 0.0), nm_zone
        assert np.all(nm[f"{nm_zone}_T_n_par_eV"] >= 0.0), nm_zone
        assert np.all(nm[f"{nm_zone}_T_n_perp_eV"] >= 0.0), nm_zone
        assert np.all(nm[f"{nm_zone}_abs_flux_n_z"] >= 0.0), nm_zone
        # (ii) THE UNSIGNED ROW BOUNDS THE SIGNED ONE, on every cell of every
        # frame: a net drift cannot exceed the traffic it is the residue of,
        # so the ratio the row exists to supply lies in [0, 1] by
        # construction and a reader can read it as one.
        assert np.all(
            nm[f"{nm_zone}_abs_flux_n_z"]
            >= np.abs(nm[f"{nm_zone}_flux_n_z"])
        ), nm_zone
    # NON-VACUOUS: the column carries gas, and it is moving.
    assert nm["column_n_n"].max() > 0.0
    assert np.abs(nm["column_u_n"]).max() > 0.0
    assert nm["column_T_n_par_eV"].max() > 0.0
    # NON-VACUOUS for the unsigned row too: wherever the column holds gas
    # there is traffic to measure, and the bound above is not an identity --
    # somewhere the signed flux really has cancelled.
    nm_col_on = nm["column_n_n"] > 0.0
    assert nm_col_on.any()
    assert np.all(nm["column_abs_flux_n_z"][nm_col_on] > 0.0)
    assert np.any(
        nm["column_abs_flux_n_z"] > np.abs(nm["column_flux_n_z"])
    )

    # (i) THE ROWS ARE THE QUADRATURE. An independent direct sum over the
    # velocity bins -- flattened and contracted, a different summation order
    # from the export's own axis reduction -- reproduces the exported flux
    # to 1e-12 of the quadrature's OWN magnitude, the sum of its term
    # magnitudes. That normalization, not the flux itself, is the only
    # well-posed one here: the flux is a signed sum over a near-isotropic
    # distribution, so it cancels to the summation's roundoff floor in
    # almost every cell, and a flux-relative statement is undefined exactly
    # where the flux is smallest. The density and the velocity are pinned
    # against the same direct sum, and both are cancellation-free.
    nm_g = nm_sim._dvm.g
    nm_live = nm_sim._dvm.zone_velocity_moments()
    nm_vz_flat = np.broadcast_to(nm_g.VZ, (nm_g.nvz, nm_g.nvp)).reshape(-1)
    for nm_zone, nm_f in (
        ("column", nm_sim._dvm.f_c), ("annulus", nm_sim._dvm.f_a)
    ):
        nm_flat = nm_f.reshape(nm_cells, -1)
        nm_direct = nm_flat @ nm_vz_flat
        nm_got = nm_live[f"{nm_zone}_flux_n_z"]
        nm_scale = np.abs(nm_flat) @ np.abs(nm_vz_flat)
        nm_err = np.abs(nm_got - nm_direct)
        # NON-VACUOUS: there are terms to sum in every cell.
        assert np.all(nm_scale > 0.0), nm_zone
        assert np.all(nm_err <= 1.0e-12 * nm_scale), nm_zone
        # The exported density is that same sum, and the velocity their ratio.
        nm_n_direct = nm_flat @ np.ones(nm_g.nvz * nm_g.nvp)
        assert np.allclose(
            nm_live[f"{nm_zone}_n_n"], nm_n_direct, rtol=1.0e-12, atol=0.0
        ), nm_zone
        assert np.allclose(
            nm_live[f"{nm_zone}_u_n"] * nm_live[f"{nm_zone}_n_n"],
            nm_got, rtol=1.0e-12, atol=0.0,
        ), nm_zone
        # (i) THE UNSIGNED ROW IS THAT SAME QUADRATURE, UNCANCELLED. The
        # magnitude scale the signed flux was just judged against IS the
        # exported unsigned row, computed here by the independent
        # contraction rather than the export's axis reduction. This sum has
        # no cancellation, so the row's OWN magnitude is a well-posed
        # normalization for it -- 1e-12 relative to the row itself.
        nm_abs_got = nm_live[f"{nm_zone}_abs_flux_n_z"]
        assert np.allclose(
            nm_abs_got, nm_scale, rtol=1.0e-12, atol=0.0
        ), nm_zone
        # The save frame is the ENGINE's reading, not a re-derivation.
        assert np.array_equal(
            nm[f"{nm_zone}_flux_n_z"][-1], nm_got
        ), nm_zone
        assert np.array_equal(
            nm[f"{nm_zone}_abs_flux_n_z"][-1], nm_abs_got
        ), nm_zone

    # (ii) A KNOWN DRIFTING MAXWELLIAN COMES BACK. The grid's own projection
    # pins its DISCRETE drift and its discrete <v^2> to 1e-10 of the target
    # (that is the projection's own convergence criterion), so the drift and
    # the three-degree-of-freedom temperature are recovered to that. The
    # SPLIT between the parallel and perpendicular temperatures is not pinned
    # by the projection and carries the grid's midpoint quadrature error
    # instead: summing a second moment with the bin masses placed at the bin
    # CENTRES misses, per bin, at most the within-bin spread
    # ``dv^2/4 + |v - u| dv``, so the tolerance below is that bound summed
    # over the run's OWN grid with the projection's OWN bin masses -- a
    # number read off the grid, not chosen.
    for nm_T_eV, nm_u in ((0.30, 0.15 * nm_g.vz[-1]), (0.30, 0.0),
                          (1.0, 0.30 * nm_g.vz[-1])):
        nm_spec = nm_g.maxwellian(nm_T_eV, nm_u)
        nm_n0 = 3.0e12
        nm_fm = (nm_n0 * nm_spec)[None, :, :].copy()
        nm_n, nm_flux, nm_u_got = _nm_axial_moments(nm_fm, nm_g)
        nm_T_par, nm_T_perp = _nm_directional_T_eV(
            nm_fm, nm_g, nm_n, nm_u_got
        )
        nm_s = np.sqrt(nm_T_eV * _NM_EV / _NM_M_HE)
        assert np.isclose(nm_n[0], nm_n0, rtol=1.0e-12, atol=0.0)
        assert abs(nm_u_got[0] - nm_u) <= 1.0e-10 * max(abs(nm_u), nm_s)
        assert np.isclose(nm_flux[0], nm_n0 * nm_u_got[0],
                          rtol=1.0e-12, atol=0.0)
        assert abs(
            (nm_T_par[0] + 2.0 * nm_T_perp[0]) / 3.0 - nm_T_eV
        ) <= 1.0e-10 * nm_T_eV
        nm_wz = nm_spec.sum(axis=1)
        nm_wp = nm_spec.sum(axis=0)
        nm_dvz = np.diff(nm_g.vz_edges)
        nm_dvp = np.diff(nm_g.vp_edges)
        nm_spread_z = 0.25 * nm_dvz**2 + np.abs(nm_g.vz - nm_u) * nm_dvz
        nm_tol_par = (_NM_M_HE / _NM_EV) * float((nm_wz * nm_spread_z).sum())
        nm_tol_perp = (_NM_M_HE / (2.0 * _NM_EV)) * float(
            (nm_wp * (0.25 * nm_dvp**2 + nm_g.vp * nm_dvp)).sum()
        )
        assert abs(nm_T_par[0] - nm_T_eV) <= nm_tol_par, (nm_T_eV, nm_u)
        assert abs(nm_T_perp[0] - nm_T_eV) <= nm_tol_perp, (nm_T_eV, nm_u)

    # (ii) ON A ONE-SIDED DISTRIBUTION THE TWO COINCIDE. With every bin mass
    # at ``v_z > 0`` there is nothing to cancel, so the unsigned sum and the
    # signed one are the same sum and the ratio the row supplies is exactly
    # 1: the bound above is tight, not merely true.
    nm_os_spec = nm_g.maxwellian(0.30, 0.0).copy()
    nm_os_spec[nm_g.vz <= 0.0, :] = 0.0
    nm_os = (2.0e12 * nm_os_spec)[None, :, :].copy()
    nm_os_n, nm_os_flux, nm_os_u, nm_os_abs = _nm_axial_moments(
        nm_os, nm_g, with_abs_flux=True
    )
    # NON-VACUOUS: the one-sided construction kept mass, and it is moving.
    assert nm_os_n[0] > 0.0
    assert nm_os_flux[0] > 0.0
    assert nm_os_u[0] > 0.0
    assert np.isclose(
        nm_os_abs[0], abs(nm_os_flux[0]), rtol=1.0e-12, atol=0.0
    )

    # (iii) ZERO WHERE DOCUMENTED. An empty zone has no drift, no traffic and
    # no temperature to report, and the export says 0.0 rather than a NaN.
    nm_empty = np.zeros((2, nm_g.nvz, nm_g.nvp))
    nm_e_n, nm_e_flux, nm_e_u, nm_e_abs = _nm_axial_moments(
        nm_empty, nm_g, with_abs_flux=True
    )
    nm_e_par, nm_e_perp = _nm_directional_T_eV(
        nm_empty, nm_g, nm_e_n, nm_e_u
    )
    for nm_e_row in (
        nm_e_n, nm_e_flux, nm_e_u, nm_e_abs, nm_e_par, nm_e_perp
    ):
        assert np.all(nm_e_row == 0.0)
    # And on the run itself, wherever a zone is empty.
    for nm_zone in _NM_ZONES:
        nm_off = nm[f"{nm_zone}_n_n"] <= 0.0
        for nm_q in ("u_n", "T_n_par_eV", "T_n_perp_eV", "abs_flux_n_z"):
            assert np.all(nm[f"{nm_zone}_{nm_q}"][nm_off] == 0.0), nm_zone

    with tempfile.TemporaryDirectory() as nm_dir:
        # (iv) A FILE WITHOUT THE GROUP STILL LOADS -- the moment control,
        # which never wrote one.
        nm_mom_path = Path(nm_dir) / "dvm_moments_moment.h5"
        _save_result_hdf5_nm(nm_mom_path, nm_mom_result)
        with h5py.File(nm_mom_path, "r") as nm_mom_h5:
            assert "dvm_neutral_moments" not in nm_mom_h5
        assert not hasattr(
            load_result_hdf5(nm_mom_path), "dvm_neutral_moments"
        )

        nm_path = Path(nm_dir) / "dvm_moments.h5"
        _save_result_hdf5_nm(
            nm_path, nm_result, params=nm_params, flags=nm_flags
        )
        with h5py.File(nm_path, "r") as nm_h5:
            nm_group = nm_h5["dvm_neutral_moments"]
            assert set(nm_group) == set(_NM_KEYS)
            # The group documents itself, the way the particle ledger does.
            nm_channels = tuple(nm_group.attrs["channels"])
            assert nm_channels == tuple(_NM_KEYS)
            for nm_attr, nm_index in (
                ("channel_units", 0), ("channel_meanings", 1)
            ):
                assert list(nm_group.attrs[nm_attr]) == [
                    _NM_ROW_DOC[nm_name][nm_index] for nm_name in nm_channels
                ], nm_attr
            nm_units = dict(
                zip(nm_channels, nm_group.attrs["channel_units"])
            )
            assert nm_units["time"] == "s"
            for nm_zone in _NM_ZONES:
                assert nm_units[f"{nm_zone}_n_n"] == "cm^-3"
                assert nm_units[f"{nm_zone}_flux_n_z"] == "cm^-2 s^-1"
                assert nm_units[f"{nm_zone}_abs_flux_n_z"] == "cm^-2 s^-1"
                assert nm_units[f"{nm_zone}_u_n"] == "cm/s"
                assert nm_units[f"{nm_zone}_T_n_par_eV"] == "eV"
                assert nm_units[f"{nm_zone}_T_n_perp_eV"] == "eV"
        nm_back = load_result_hdf5(nm_path).dvm_neutral_moments
        assert set(nm_back) == set(nm)
        for nm_name, nm_row in nm.items():
            assert np.array_equal(nm_back[nm_name], nm_row), nm_name

        # (iv) again, on a DVM file in the OLD format: the same file with the
        # group removed loads with the attribute simply absent, and every
        # other group it carries is untouched.
        nm_old_path = Path(nm_dir) / "dvm_moments_old_format.h5"
        shutil.copyfile(nm_path, nm_old_path)
        with h5py.File(nm_old_path, "r+") as nm_old_h5:
            del nm_old_h5["dvm_neutral_moments"]
        nm_old = load_result_hdf5(nm_old_path)
        assert not hasattr(nm_old, "dvm_neutral_moments")
        assert set(nm_old.dvm_particle_ledger) == set(
            nm_result.dvm_particle_ledger
        )
        assert np.array_equal(nm_old.time, nm_result.time)

        # (iv) once more, on a file written by the layout that carried the
        # group WITHOUT the unsigned rows: the same file with just those two
        # datasets removed loads with every row it does carry and the two it
        # does not simply absent. The group is read by NAME, so adding a row
        # to it does not invalidate a file written before that row existed.
        nm_pre_path = Path(nm_dir) / "dvm_moments_pre_abs_flux.h5"
        shutil.copyfile(nm_path, nm_pre_path)
        nm_abs_names = {
            f"{nm_zone}_abs_flux_n_z" for nm_zone in _NM_ZONES
        }
        with h5py.File(nm_pre_path, "r+") as nm_pre_h5:
            nm_pre_group = nm_pre_h5["dvm_neutral_moments"]
            for nm_abs_name in nm_abs_names:
                del nm_pre_group[nm_abs_name]
        nm_pre = load_result_hdf5(nm_pre_path).dvm_neutral_moments
        assert set(nm_pre) == set(_NM_KEYS) - nm_abs_names
        for nm_name, nm_row in nm_pre.items():
            assert np.array_equal(nm_row, nm[nm_name]), nm_name


# --------------------------------------------------------------------
# dvm-jet-rn-interval-refusals
# --------------------------------------------------------------------
@_case("dvm-jet-rn-interval-refusals")
def _case_dvm_jet_rn_interval_refusals():
    # THE THREE SURFACE JETS REFUSE R_N = 0 THE SAME WAY. Each jet's launch
    # band is formed by dividing R_E by R_N, and the solver forms all three
    # bands before the engine exists. The end wall spec was already put
    # through the engine's validator at that point; the cathode and anode
    # specs were not, so their R_N = 0 reached the division and answered with
    # a ZeroDivisionError -- a Python arithmetic failure where the validators
    # promise an interval statement naming the key. All three now validate
    # before the band is formed, so the message a reader gets is the same
    # message whichever surface was misconfigured.
    for _jr_label, _jr_params, _jr_flags, _jr_says in (
        (
            "cathode",
            {
                "neutral_kinetic_dvm_cathode_jet": True,
                "neutral_kinetic_dvm_cathode_jet_R_N": 0.0,
                "neutral_kinetic_dvm_cathode_jet_R_E": 0.0,
                "cathode_neutral_jet": False,
            },
            {"cathode_coupling": True},
            "0 < R_E <= R_N < 1",
        ),
        (
            "anode",
            {
                "neutral_kinetic_dvm_anode_jet": True,
                "neutral_kinetic_dvm_anode_jet_R_N": 0.0,
                "neutral_kinetic_dvm_anode_jet_R_E": 0.0,
                "anode_neutral_jet": False,
            },
            {"cathode_coupling": True},
            "0 < R_E <= R_N < 1",
        ),
        (
            # The surface's name AS THE MESSAGE SPELLS IT: the validators name
            # the face in prose ("the DVM end wall jet's R_N ..."), so the
            # label checked below is the prose form, not the key's.
            "end wall",
            {
                "neutral_kinetic_dvm_end_wall_jet": True,
                "neutral_kinetic_dvm_end_wall_jet_R_N": 0.0,
                "neutral_kinetic_dvm_end_wall_jet_R_E": 0.0,
                "neutral_kinetic_dvm_end_wall_jet_sheath_Te_multiple": 3.0,
            },
            {},
            "0 < R_N <= 1",
        ),
    ):
        _jr_p, _jr_f = default_config()
        _jr_p["neutral_model"] = "kinetic_dvm"
        _jr_p.update(_jr_params)
        _jr_f.update(_jr_flags)
        try:
            LAPDSim1D(_jr_p, _jr_f)
        except ZeroDivisionError as _jr_zero:
            raise AssertionError(
                f"the DVM {_jr_label} jet answered R_N = 0 with a "
                f"ZeroDivisionError ({_jr_zero}) instead of the interval "
                "statement its validator promises"
            )
        except ValueError as _jr_error:
            assert _jr_says in str(_jr_error), (_jr_label, str(_jr_error))
            assert _jr_label in str(_jr_error).lower(), (
                _jr_label, str(_jr_error)
            )
        else:
            raise AssertionError(
                f"the DVM {_jr_label} jet accepted R_N = 0"
            )
