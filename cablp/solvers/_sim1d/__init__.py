"""Public API for the conservative axial 1D LAPD solver package.

The names in ``__all__`` are the declared surface: the configuration
templates and their loaders, the solver class with its progress and error
types, result and restart IO, and the trajectory health summary. Every other
member of this package remains importable by its own module path, and code
that needs an operator, a geometry helper or a state accessor reaches it that
way; new code targets the surface below.
"""

from .core.config import (
    config_manifest,
    default_config,
    input_dict_template_1d,
    input_flags_template_1d,
    load_config,
    resolve_config,
)
from .core.model_declarations import resolve_declaration_blocks
from .core.model_families import (
    DECLARED_FAMILIES,
    KINETIC_DVM_INCOMPATIBLE_DEFAULTS,
    KINETIC_TWO_MOMENT_INCOMPATIBLE_DEFAULTS,
    KINETIC_TWO_MOMENT_INTERNAL_MEMBERS,
    MODEL_FAMILIES,
    resolve_model_families,
)
from .results.io import save_result_hdf5
from .results.restart import (
    load_restart_state,
    save_restart_state,
)
from .solver import (
    BreakdownError,
    LAPDSim1D,
    ProgressPrinter1D,
    SimulationProgress1D,
    TimestepRejectionError,
    load_result_hdf5,
    summarize_result,
)

__all__ = [
    "BreakdownError",
    "DECLARED_FAMILIES",
    "KINETIC_DVM_INCOMPATIBLE_DEFAULTS",
    "KINETIC_TWO_MOMENT_INCOMPATIBLE_DEFAULTS",
    "KINETIC_TWO_MOMENT_INTERNAL_MEMBERS",
    "LAPDSim1D",
    "MODEL_FAMILIES",
    "ProgressPrinter1D",
    "SimulationProgress1D",
    "TimestepRejectionError",
    "config_manifest",
    "default_config",
    "input_dict_template_1d",
    "input_flags_template_1d",
    "load_config",
    "load_restart_state",
    "load_result_hdf5",
    "resolve_config",
    "resolve_declaration_blocks",
    "resolve_model_families",
    "save_restart_state",
    "save_result_hdf5",
    "summarize_result",
]
