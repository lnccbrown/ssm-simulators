"""Seeded wrapper around ``cssm.lba_vanilla``.

``cssm.lba_vanilla`` draws from NumPy's *global* legacy RNG and never consumes
the ``random_state`` that ``ssms`` passes to every model function, so the
built-in ``lba2``/``lba3``/``lba4`` are not reproducible through the public
``random_state`` argument::

    >>> a = simulator(theta=..., model="lba3", n_samples=5, random_state=42)
    >>> b = simulator(theta=..., model="lba3", n_samples=5, random_state=42)
    >>> # a["rts"] != b["rts"]

Nothing in the MLBA requires fixing that, but a model whose seed is silently
ignored is of little use for generating training data, so the registered
``mlba2``/``mlba3`` go through this wrapper, which seeds the generator
``cssm.lba_vanilla`` actually uses and then delegates.  The simulation itself
is unchanged -- it is the same engine, with the same draw order.

If ``cssm.lba_vanilla`` is ever fixed upstream to honour ``random_state``,
this wrapper should be dropped and the configs pointed back at
``cssm.lba_vanilla`` directly.
"""

from __future__ import annotations

import numpy as np

import cssm

__all__ = ["lba_vanilla_seeded"]

_SEED_MODULUS = 2**32  # np.random.seed accepts [0, 2**32); ssms allows negatives


def lba_vanilla_seeded(random_state: int | None = None, **kwargs) -> dict:
    """Seed the global NumPy RNG, then call ``cssm.lba_vanilla``.

    Parameters
    ----------
    random_state
        Seed forwarded by ``ssms``.  ``None`` leaves the RNG untouched.
    **kwargs
        Everything ``cssm.lba_vanilla`` takes (``v``, ``a``, ``z``,
        ``deadline``, ``sd``, ``t``, ``nact``, ``n_samples``, ...), plus the
        simulation-config entries ``ssms`` splats in and ``lba_vanilla``
        absorbs through its own ``**kwargs``.
    """
    if random_state is not None:
        np.random.seed(int(random_state) % _SEED_MODULUS)
    return cssm.lba_vanilla(**kwargs)
