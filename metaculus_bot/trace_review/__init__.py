"""Offline trace review: an index of every resolved forecast and a Streamlit app to read it.

``python -m metaculus_bot.trace_review`` (``make trace_index``) joins the newest residual round, a
free read-only pull of the current tournaments, the telemetry archive and the research archive into
the parquet bundle under ``backtests/trace_index/``; ``make review`` opens the app over that bundle.
Nothing here calls an LLM or a research provider, and nothing writes to Metaculus or Mantic.
Docs: ``docs/trace_review.md``.
"""
