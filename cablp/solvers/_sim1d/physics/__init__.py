"""Conservative physics operators for sim1d.

Members are imported by their own module path -- this package deliberately
re-exports nothing, so there is exactly one name for every symbol. No operator
appears on the package surface in ``cablp.solvers._sim1d``: the solver composes
them, and the audit and verification scripts that call one directly name the
module it lives in.
"""
