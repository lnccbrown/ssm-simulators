# Header file for _utils.pyx
# Allows C-level imports of utility functions across CSSM modules

cimport numpy as np

# Random number generation functions (updated to NumPy-based implementations)
cpdef void set_seed(random_state)
cpdef float random_uniform()
cpdef float random_exponential()
cpdef float[:] draw_random_stable(int n, float alpha)
cpdef float[:] draw_gaussian(int n)
cpdef float[:] draw_uniform(int n)

# Helper functions
cpdef int sign(float x)
cpdef float csum(float[:] x)
