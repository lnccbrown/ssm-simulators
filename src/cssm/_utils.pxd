# Header file for _utils.pyx
# Allows C-level imports of utility functions across CSSM modules

cimport numpy as np

# Random number generation functions
cdef set_seed(random_state)
cdef float random_uniform()
cdef float random_exponential()
cdef float random_stable(float alpha)
cdef float[:] draw_random_stable(int n, float alpha)
cdef float random_gaussian()

# Helper functions
cdef int sign(float x)
cdef float csum(float[:] x)
cdef void assign_random_gaussian_pair(float[:] out, int assign_ix)
cdef float[:] draw_gaussian(int n)
