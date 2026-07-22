"""Shared dimensionless math constants used across the codebase.

Kept as plain Python floats so they broadcast cleanly into both numpy
and torch expressions. Domain-specific constants (e.g. the Branin
coefficients, GP noise floors, Cholesky jitter) deliberately do *not*
live here — they belong with the objects they parameterise.
"""

import math

PI_SQUARED = math.pi**2
LOG_2PI = math.log(2.0 * math.pi)
HALF_LOG_2PI = 0.5 * LOG_2PI
