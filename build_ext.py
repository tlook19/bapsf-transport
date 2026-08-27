"""Poetry build script: compile the optional Cython scalar kernels.

Scope (D4, extended to the Tier A cathode unit and then to the CSDA beam
march, 2026-08-02): ONE extension module,
``cablp.funcs._cathode_kernels_cy``. Nothing imports it unless
``CABLP_COMPILED_KERNELS`` opts in (``cablp.funcs._kernels``), so the pure
Python path is unaffected whether or not the extension is present.

Fresh-clone implications
------------------------
``pyproject.toml`` lists Cython and setuptools in ``build-system.requires``
and points ``[tool.poetry.build]`` here, so ``poetry install`` / ``pip
install .`` will compile the extension and a wheel built from this source
will carry it. A clone WITHOUT a C compiler still works: the build is
best-effort (see ``_cythonize`` below) and a failure leaves the package pure
Python, which is the default runtime path anyway. Opting in without a built
extension then fails loudly at import rather than silently running pure --
that is the intended behaviour.

How poetry-core invokes this
----------------------------
poetry-core 2.x with ``generate-setup-file = false`` runs the build script as
a plain subprocess -- ``python build_ext.py`` in the project root -- and then
packages whatever the script left in the source tree. So bare execution MUST
build the extension in place; there is no ``setup_kwargs`` handshake. The
legacy ``build(setup_kwargs)`` hook is kept below for older poetry-core
versions that do use it, and both routes share ``_cythonize``.

The same command is what you want for a spike or a benchmark, without
reinstalling::

    python build_ext.py
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
PYX = "cablp/funcs/_cathode_kernels_cy.pyx"

# Bit-exactness matters more than speed here. ``-ffp-contract=off`` DISABLES
# fused multiply-add contraction, which would otherwise let the compiler round
# ``1.0 + 2.0*psi`` once instead of twice and put the compiled kernel a ULP
# away from CPython's arithmetic. It is the opposite of a fastmath flag; no
# fastmath, reassociation, or unsafe-math option is used anywhere.
#
# It also has to be OFF for the one place the kernels must contract: numpy's
# own ``arr_interp`` is built WITH contraction, so its lerp is a single
# rounding, and ``_interp_scalar`` reproduces that with an explicit ``fma()``
# call. With contraction left on the compiler's discretion, "is this
# expression fused?" would stop being a property of the source.
SAFE_FP_ARGS = ["-ffp-contract=off"]


def _extensions():
    from setuptools import Extension

    return [
        Extension(
            "cablp.funcs._cathode_kernels_cy",
            [PYX],
            extra_compile_args=list(SAFE_FP_ARGS),
        )
    ]


def _cythonize():
    """Return cythonized ext_modules, or ``None`` if Cython is unavailable."""
    try:
        from Cython.Build import cythonize
    except ImportError:
        return None
    return cythonize(
        _extensions(),
        compiler_directives={"language_level": "3"},
        quiet=True,
    )


def build(setup_kwargs):
    """Legacy poetry-core hook: attach the extension to the generated setup().

    Unused by poetry-core 2.x, which runs this file as a script instead (see
    the module docstring). Kept so an older backend still gets the extension.
    """
    ext_modules = _cythonize()
    if ext_modules is None:
        # No Cython at build time: ship pure Python. The compiled path is
        # opt-in and its absence is detected loudly at opt-in time.
        return setup_kwargs
    setup_kwargs.update({"ext_modules": ext_modules, "zip_safe": False})
    return setup_kwargs


def build_inplace():
    """Compile the extension next to its ``.pyx``. Returns True if it built.

    Best-effort by design: a checkout with no Cython or no C compiler stays
    pure Python, which is the default runtime path anyway. Opting in to the
    compiled kernels without a built extension then fails loudly at import,
    so a skipped build can never masquerade as a compiled run.
    """
    ext_modules = _cythonize()
    if ext_modules is None:
        print("build_ext: Cython unavailable; skipping the optional extension")
        return False
    from setuptools import setup

    # `packages=[]` is not decoration: this call only compiles an extension in
    # place and installs nothing, and without it setuptools runs flat-layout
    # auto-discovery over the project root. Since the R2 flatten that root is
    # the repository root, where `cablp/` is not the only directory holding
    # Python, and auto-discovery then refuses to build at all ("Multiple
    # top-level packages discovered in a flat-layout").
    setup(
        name="cablp-kernels",
        packages=[],
        ext_modules=ext_modules,
        script_args=["build_ext", "--inplace"],
    )
    return True


def _main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="accepted for explicitness; in-place is the only mode",
    )
    parser.parse_args()
    try:
        build_inplace()
    except (Exception, SystemExit) as error:  # noqa: BLE001
        # A missing or unusable C toolchain must not break `poetry install`.
        # SystemExit is caught explicitly and is NOT paranoia: setuptools'
        # build_ext reports a failed compiler by calling sys.exit(), which is
        # a BaseException and would otherwise sail past `except Exception`
        # and fail the whole install on a machine with no compiler. Verified
        # 2026-08-02 with CC=/nonexistent/cc.
        print(f"build_ext: optional extension did not build ({error})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
