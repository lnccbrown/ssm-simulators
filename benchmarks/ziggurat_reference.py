"""
Ziggurat sampler for N(0,1) with table generation (Marsaglia & Tsang style).

This is a self-contained implementation:
 - generates the x / wn / kn / fn tables programmatically
 - uses 24-bit x-coordinates (32-bit random word with 7-bit index + 1 sign bit)
 - implements the tail sampler exactly as Marsaglia recommends
 - includes a small correctness test

References:
 - Marsaglia & Tsang, "The Ziggurat Method for Generating Random Variables" (2000).
 - GSL implementations and tables (Voss / GSL).
"""

from __future__ import annotations
import math
import random
import numpy as np
from typing import Tuple, List

# ---------- Parameters ----------
N_LAYERS = 128  # number of layers/rectangles
R = 3.442619855899  # tail cutoff used by standard implementations
V = 9.91256303526217e-3  # rectangle area constant used by many refs
BITS = 24  # number of bits used for j (so 32 - 8 index bits)
TWO_POW_BITS = float(1 << BITS)


# ---------- Table generation ----------
def make_ziggurat_tables(
    n: int = N_LAYERS, r: float = R, v: float = V, bits: int = BITS
) -> Tuple[List[float], List[float], List[int], List[float]]:
    """
    Generate x, wn, kn, fn arrays consistent with Marsaglia & Tsang.
    - x: length n+1, x[0]=r, x[n]=0
    - wn: length n, wn[i] = x[i] / 2^bits
    - kn: length n, kn[i] = floor(2^bits * x[i+1] / x[i])  (integer thresholds)
    - fn: length n+1, fn[i] = exp(-x[i]^2 / 2)
    """
    x = [0.0] * (n + 1)
    fn = [0.0] * (n + 1)

    x[0] = r
    fn[0] = math.exp(-0.5 * r * r)

    # recurrence from Marsaglia-Tsang: x[i+1] = sqrt(-2 * ln( f(x[i]) + v / x[i] ))
    for i in range(0, n - 1):
        term = fn[i] + v / x[i]
        # Guard against tiny rounding issues that might make term >= 1
        if term >= 1.0:
            x[i + 1] = 0.0
            fn[i + 1] = 1.0
        else:
            x[i + 1] = math.sqrt(-2.0 * math.log(term))
            fn[i + 1] = math.exp(-0.5 * x[i + 1] * x[i + 1])

    # ensure last x[n] is exactly 0
    x[n] = 0.0
    fn[n] = 1.0

    # wn and kn
    wn = [0.0] * n
    kn = [0] * n
    for i in range(n):
        wn[i] = x[i] / float(1 << bits)  # x[i] / 2^bits
        # compute k = floor(2^bits * x[i+1] / x[i])
        if x[i] == 0.0:
            k = 0
        else:
            ratio = (x[i + 1] / x[i]) * (1 << bits)
            k = int(math.floor(ratio))
            # clamp
            if k < 0:
                k = 0
            elif k > (1 << bits) - 1:
                k = (1 << bits) - 1
        kn[i] = k

    return x, wn, kn, fn


# ---------- Tail sampler ----------
def tail_sample(rng_random=random.random) -> float:
    """
    Sample the tail (x > R) for standard normal using Marsaglia tail method.
    Returns positive x >= R.
    """
    while True:
        u1 = rng_random()
        u2 = rng_random()
        # guard: avoid log(0)
        if u1 <= 0.0:
            u1 = 1e-300
        if u2 <= 0.0:
            u2 = 1e-300
        x = -math.log(u1) / R
        y = -math.log(u2)
        if 2.0 * y >= x * x:
            return R + x


# ---------- Ziggurat Gaussian sampler factory ----------
def make_ziggurat_gaussian(
    rng: random.Random = random,
    n: int = N_LAYERS,
    r: float = R,
    v: float = V,
    bits: int = BITS,
):
    """
    Return a function that draws standard normal variates using the
    ziggurat tables generated with the same parameters.
    The returned function draws one sample per call.
    """

    x_tbl, wn, kn, fn = make_ziggurat_tables(n=n, r=r, v=v, bits=bits)

    def ziggurat_one() -> float:
        while True:
            u32 = rng.getrandbits(32)  # 32-bit random word
            i = u32 & 0x7F  # bottom 7 bits -> layer index 0..127
            sign = (u32 >> 7) & 1  # bit 7 -> sign (0 -> negative, 1 -> positive)
            j = u32 >> 8  # upper 24 bits -> j in [0, 2^24)

            # Fast path
            if j < kn[i]:
                x = j * wn[i]
                return x if sign else -x

            # Tail layer
            if i == 0:
                x = tail_sample(rng.random)
                return x if sign else -x

            # Innermost layer (i == n-1): always accept
            if i == n - 1:
                x = j * wn[i]
                return x if sign else -x

            # Wedge test (layers 1..n-2)
            x = j * wn[i]
            # y uniform in [fn[i], fn[i+1]]
            y = fn[i] + rng.random() * (fn[i + 1] - fn[i])
            if y < math.exp(-0.5 * x * x):
                return x if sign else -x

            # otherwise loop and try again

    # supply tables in case caller wants to inspect them
    ziggurat_one.tables = {
        "x": x_tbl,
        "wn": wn,
        "kn": kn,
        "fn": fn,
        "n": n,
        "r": r,
        "v": v,
        "bits": bits,
    }
    return ziggurat_one


# ---------- Quick validation ----------
def test_ziggurat(samples: int = 2_000_000, seed: int = 123456789):
    rng = random.Random(seed)
    zigg = make_ziggurat_gaussian(rng)

    # warm up a bit
    _ = [zigg() for _ in range(1000)]

    # accumulate mean and variance with Welford's algorithm (numerically stable)
    mean = 0.0
    m2 = 0.0
    n = 0
    for _ in range(samples):
        v = zigg()
        n += 1
        delta = v - mean
        mean += delta / n
        m2 += delta * (v - mean)

    var = m2 / n  # population variance
    print(f"Samples: {n:,}")
    print(f"Mean   : {mean:.6e}")
    print(f"Var    : {var:.12f}")

    # optional simple histogram check (coarse)
    return mean, var


if __name__ == "__main__":
    # Run a quick test (can lower samples for faster check)
    for i in range(10):
        seed = np.random.randint(0, 1000000)
        test_ziggurat(samples=500_000, seed=seed)
