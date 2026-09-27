"""Cathode/anode/bank circuit physics, beam deposition, and the compiled kernels.

Members are imported by their own module path -- this package deliberately
re-exports nothing, so there is exactly one name for every symbol.

- :mod:`cablp.cathode.circuit_common` -- the primitives the circuit solves
  share: the device, plasma and result records, Richardson emission and its
  space-charge ceiling, the beam mean free path and gap bypass, and the
  electrode power expressions.
- :mod:`cablp.cathode.circuit_idriven` -- the current-driven sheath solve,
  with the inductor-integrated loop current as the independent variable.
- :mod:`cablp.cathode.circuit_prescribed` -- the prescribed-measured sheath
  solve, driven by a measured discharge current and voltage.
- :mod:`cablp.cathode.beam_deposition` -- CSDA beam-deposition marches for the
  emitted electron beam.
- :mod:`cablp.cathode.kernels` -- the ``CABLP_COMPILED_KERNELS`` opt-in
  selector, which binds the one compiled extension below and publishes its
  ``KERNEL_ID`` as artifact provenance.
- ``_cathode_kernels_cy.pyx`` -- that extension. Its basename keeps its leading
  underscore deliberately: renaming it would change ``KERNEL_ID`` and so the
  ``compiled_kernels`` provenance recorded in every artifact ever produced on
  the compiled path.
"""
