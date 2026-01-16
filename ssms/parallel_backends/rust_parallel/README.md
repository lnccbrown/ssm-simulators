# Rust Parallel Backend for SSM Simulators

This directory contains the Rust implementation of DDM simulators using
Rayon for parallelism and PyO3 for Python bindings.

## Prerequisites

1. **Rust toolchain**: Install from https://rustup.rs/
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **maturin**: Python build tool for Rust extensions
   ```bash
   pip install maturin
   ```

## Building

### Development (fast, debug build)

```bash
cd ssms/parallel_backends/rust_parallel
maturin develop
```

### Release (optimized build)

```bash
cd ssms/parallel_backends/rust_parallel
maturin develop --release
```

### Building a wheel

```bash
maturin build --release
```

## Usage

After building, you can use the Rust backend:

```python
from ssms.parallel_backends.rust_parallel import ddm_rust

result = ddm_rust(
    v=np.array([0.5] * 100, dtype=np.float32),
    a=np.array([1.5] * 100, dtype=np.float32),
    z=np.array([0.5] * 100, dtype=np.float32),
    t=np.array([0.3] * 100, dtype=np.float32),
    n_samples=10000,
)
```

## Performance Notes

The Rust implementation uses:
- **Rayon** for work-stealing parallelism
- **XorShift** RNG for fast, reproducible random numbers
- **Zero-copy** NumPy integration via PyO3

Expected speedup: 2-10x over Cython depending on workload and thread count.

## Troubleshooting

If you get import errors:
1. Make sure Rust is installed: `rustc --version`
2. Make sure maturin is installed: `maturin --version`
3. Rebuild with: `maturin develop --release`
