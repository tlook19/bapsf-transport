"""covcal pre-gate 3d: probe which kernel path is actually loaded.

Read-only. Correct attribute path per covdecide_memo.md anomaly 5:
``cablp.funcs._kernels.PROVENANCE`` (NOT ``_cathode_kernels.KERNEL_ID``).
"""

from cablp.funcs import _kernels as K

print("ENV_VAR                =", K.ENV_VAR)
print("compiled_requested     =", K.compiled_kernels_requested())
print("PROVENANCE             =", K.PROVENANCE)
print("PURE_PROVENANCE        =", K.PURE_PROVENANCE)
print("COMPILED_KERNELS       =", K.COMPILED_KERNELS)
mod = K.COMPILED_KERNELS
if mod is not None:
    print("module KERNEL_ID       =", getattr(mod, "KERNEL_ID", "<absent>"))
