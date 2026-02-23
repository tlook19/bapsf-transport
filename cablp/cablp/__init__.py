submodules = ["funcs", "vars", "solvers"]

__all__ = submodules + [
    "__version__",
]


def __dir__():
    return __all__


import importlib as _importlib


def __getattr__(name):
    if name in submodules:
        return _importlib.import_module(f"bapsfda.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'bapsfda' has no attribute '{name}'")
