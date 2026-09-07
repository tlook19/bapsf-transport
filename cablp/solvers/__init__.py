"""Solvers.

``_sim1d`` (``LAPDSim1D``) is the only solver. Import it explicitly:

    from cablp.solvers._sim1d import LAPDSim1D, default_config

The 0D solvers that this package used to star-import (``_sim``, ``_rk``,
``_sim3``) are removed, and their reproducibility anchor was retired with
them: results from those solvers are deliberately not reproducible from this
repository.
"""

__all__ = []
