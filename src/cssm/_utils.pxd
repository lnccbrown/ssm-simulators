# Header file for _utils.pyx
# Allows C-level imports of utility functions across CSSM modules

cimport numpy as np

# Random number generation functions (updated to NumPy-based implementations)
cdef set_seed(random_state)
cdef float random_uniform()
cdef float random_exponential()
cdef float[:] draw_random_stable(int n, float alpha)
cdef float[:] draw_gaussian(int n)

# Helper functions
cdef int sign(float x)
cdef float csum(float[:] x)
