"""Offline cross-pollination forecasting experiment."""

"""Offline peer-update experiment over resolved questions with archived research.

The cost gate requires ``--i-accept-spend`` for model calls and checks the uncached
estimate against the cap. ``--build-manifest`` performs Metaculus reads only;
``--dry-run``, ``--rescore`` and ``--analyze`` are free.

Run with ``make cross_pollination_bench ARGS='--build-manifest --perf-json <path>'``
to freeze inputs, then pass that run directory to ``--dry-run`` or a separately
approved ``--i-accept-spend`` run.
"""
