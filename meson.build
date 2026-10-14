# meson-python build for ssm-simulators (replaces setup.py).
#
# Package metadata stays in pyproject.toml. This file builds the Cython
# extensions in src/cssm/ into the `cssm` package and installs the pure-Python
# `ssms` package.
#
# Optional features (see meson.options), replacing SSMS_FORCE_GSL/OPENMP:
#   -Dgsl=auto|enabled|disabled     (HAVE_GSL; also gates cssm._c_rng)
#   -Dopenmp=auto|enabled|disabled  (HAVE_OPENMP + -fopenmp on parallel modules)
# e.g.  uv sync -C setup-args=-Dgsl=disabled   /   uv build -C setup-args=-Dopenmp=disabled

project(
  'ssm-simulators',
  'c', 'cpp', 'cython',
  meson_version: '>= 1.5.0',
  default_options: [
    'buildtype=release',   # -O3 (gcc/clang) or /O2 (msvc)
    'b_ndebug=if-release',
    'cython_language=cpp', # setup.py: language="c++"
  ],
)

py = import('python').find_installation(pure: false)
py_dep = py.dependency()
numpy_dep = dependency('numpy')  # numpy >= 2 ships numpy-config
cpp = meson.get_compiler('cpp')
system = host_machine.system()

cssm_inc = include_directories('src/cssm')
cython_args = ['-3']  # compiler_directives={"language_level": "3"}

# --------------------------------------------------------------------------
# Optimization flags (setup.py get_openmp_flags base_flags, applied to every
# module, OpenMP or not).
# --------------------------------------------------------------------------
opt_cpp_args = []
opt_link_args = []
if system == 'darwin'
  # setup.py: -O3 -ffast-math, no -march for portability
  opt_cpp_args += ['-ffast-math']
elif system == 'windows'
  # setup.py: /O2 (already implied by buildtype=release with MSVC)
else
  # Linux and others (gcc). -march must stay portable for manylinux wheels
  # (PR #303): x86-64-v2 on x86_64 only, compiler default elsewhere.
  opt_cpp_args += ['-flto', '-ffast-math', '-funroll-loops']
  opt_link_args += ['-flto']
  if host_machine.cpu_family() == 'x86_64'
    opt_cpp_args += ['-march=x86-64-v2']
  endif
endif

# --------------------------------------------------------------------------
# GSL: pkg-config first, then gsl-config (what setup.py used).
# --------------------------------------------------------------------------
gsl_dep = dependency('', required: false)
if not get_option('gsl').disabled()
  gsl_dep = dependency('gsl', method: 'pkg-config', required: false)
endif
if not get_option('gsl').disabled() and not gsl_dep.found()
  gsl_config = find_program('gsl-config', required: false)
  if gsl_config.found()
    gsl_dep = declare_dependency(
      compile_args: run_command(gsl_config, '--cflags', check: true).stdout().strip().split(),
      link_args: run_command(gsl_config, '--libs', check: true).stdout().strip().split(),
    )
  endif
endif
if get_option('gsl').enabled() and not gsl_dep.found()
  error('-Dgsl=enabled but GSL was not found (pkg-config gsl / gsl-config)')
endif
have_gsl = gsl_dep.found()
gsl_args = have_gsl ? ['-DHAVE_GSL=1'] : []

# --------------------------------------------------------------------------
# OpenMP: meson's openmp dependency handles gcc/clang/msvc. Apple clang needs
# Homebrew libomp via -Xpreprocessor (setup.py's darwin branch).
# --------------------------------------------------------------------------
omp_dep = dependency('', required: false)
if not get_option('openmp').disabled()
  omp_dep = dependency('openmp', language: 'cpp', required: false)
  if not omp_dep.found() and system == 'darwin'
    brew = find_program('brew', required: false)
    if brew.found()
      r = run_command(brew, '--prefix', 'libomp', check: false)
      if r.returncode() == 0
        libomp = r.stdout().strip()
        omp_dep = declare_dependency(
          compile_args: ['-Xpreprocessor', '-fopenmp', '-I' + libomp / 'include'],
          link_args: ['-L' + libomp / 'lib', '-lomp'],
        )
      endif
    endif
  endif
  if get_option('openmp').enabled() and not omp_dep.found()
    error('-Dopenmp=enabled but no working OpenMP was found')
  endif
endif
have_openmp = omp_dep.found()
omp_args = have_openmp ? ['-DHAVE_OPENMP=1'] : []

# --------------------------------------------------------------------------
# Extension modules (same split as setup.py)
# --------------------------------------------------------------------------
# No OpenMP / GSL
cython_modules = ['_utils', 'lba_models']

# OpenMP (+ GSL when available)
openmp_modules = [
  '_openmp_status',
  'addm_models',
  'race_multistage_models',
  'ddm_models',
  'levy_models',
  'ornstein_models',
  'race_models',
  'parallel_models',
  'sequential_models',
  'poisson_race_models',
]

# Only built when GSL is available
gsl_modules = ['_c_rng']

foreach mod : cython_modules
  py.extension_module(mod, 'src/cssm' / mod + '.pyx',
    cython_args: cython_args,
    cpp_args: opt_cpp_args,
    link_args: opt_link_args,
    include_directories: cssm_inc,
    dependencies: [numpy_dep],
    install: true,
    subdir: 'cssm',
  )
endforeach

foreach mod : openmp_modules + (have_gsl ? gsl_modules : [])
  py.extension_module(mod, 'src/cssm' / mod + '.pyx',
    cython_args: cython_args,
    cpp_args: opt_cpp_args + omp_args + gsl_args,
    link_args: opt_link_args,
    include_directories: cssm_inc,
    dependencies: [numpy_dep, omp_dep, gsl_dep],
    install: true,
    subdir: 'cssm',
  )
endforeach

# cssm package: __init__.py plus the sources setuptools shipped as package
# data ("*.pyx", "*.pxd", "*.pxi", "*.h"); all .pyx are shipped, as before,
# even when _c_rng is not compiled.
cssm_pyx = []
foreach mod : cython_modules + openmp_modules + gsl_modules
  cssm_pyx += 'src/cssm' / mod + '.pyx'
endforeach
py.install_sources(
  'src/cssm/__init__.py',
  cssm_pyx,
  'src/cssm/_c_rng.pxd',
  'src/cssm/_utils.pxd',
  'src/cssm/_constants.pxi',
  'src/cssm/_rng_wrappers.pxi',
  'src/cssm/_xoshiro_rng.pxi',
  'src/cssm/gsl_rng.h',
  subdir: 'cssm',
)

# Pure-Python ssms package (incl. ssms/cli/*.yaml package data).
install_subdir('ssms', install_dir: py.get_install_dir())

summary({
  'OpenMP': have_openmp,
  'GSL': have_gsl,
  'extra C++ flags': ' '.join(opt_cpp_args),
}, section: 'ssm-simulators')
