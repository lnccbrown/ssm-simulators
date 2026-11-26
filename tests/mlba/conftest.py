"""Pytest configuration for the MLBA tests (Trueblood, Brown & Heathcote, 2014).

In the contribution bundle this file put the bundle root on ``sys.path`` and
imported ``ssms_mlba`` so that ``mlba2``/``mlba3`` were registered at runtime.
In the library the models are registered statically in
``ssms/config/_modelconfig/__init__.py`` and the implementation is importable
from ``ssms.basic_simulators.mlba*``, so no path or registration glue is needed.
"""
