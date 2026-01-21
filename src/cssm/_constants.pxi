# _constants.pxi - Shared compile-time constants for CSSM modules
# This file is textually included by Cython .pyx files using: include "_constants.pxi"
#
# IMPORTANT: This is the single source of truth for these constants.
# Do NOT define MAX_THREADS elsewhere.

# Maximum threads supported for parallel execution (compile-time limit)
# This determines the size of static arrays for per-thread RNG states.
# Requesting more threads at runtime will raise ValueError.
DEF MAX_THREADS = 256
