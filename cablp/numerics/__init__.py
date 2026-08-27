"""Numerical-method helpers with no physics content.

Members are imported by their own module path -- this package deliberately
re-exports nothing, so there is exactly one name for every symbol.

- :mod:`cablp.numerics.interp` -- the explicitly fused scalar interpolation
  (``math.fma``) that makes the pure and compiled paths bit-exact by
  construction of the source on both platforms.
"""
