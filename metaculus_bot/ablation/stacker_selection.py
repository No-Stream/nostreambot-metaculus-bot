"""Which stacker LLM an ablation run uses, and how its batch kwargs are built.

``--plain-llm`` and ``--lineup`` together decide the stacker model, and that choice has
to be visible in two places at once: the on-disk cache slug (so an opus-4.5 free-tier
run and an opus-5.5 prod run never overwrite each other) and the kwargs handed to
``run_stacker_batch``. Both derivations live here so they cannot drift apart.
"""

from __future__ import annotations

import argparse
from typing import Any

from forecasting_tools import GeneralLlm

from metaculus_bot.ablation.cache import model_slug_to_filename
from metaculus_bot.ablation.run_stacker import (
    _OPENAI_STACKER_KWARGS,
    _OPUS_STACKER_KWARGS,
    _PROD_STACKER_KWARGS,
    DEFAULT_STACKER_FALLBACK_MODEL,
    DEFAULT_STACKER_MODEL,
    PROD_STACKER_MODEL,
)


def _active_stacker_slug(args: argparse.Namespace) -> str:
    """Filesystem slug for the stacker this run uses, for per-stacker cache keying.

    Only the LLM-stacker arms (stack / stack_aug) are slugged so a stacker swap
    (e.g. opus-4.5 free-tier vs opus-5.5 prod) never overwrites another stacker's
    results, while deterministic arms (median / mean / pdf_*) stay shared. The
    selection mirrors the stacker construction in ``_stage_llm_stacker``:

    * ``--plain-llm --lineup prod`` → opus-5.5 (``PROD_STACKER_MODEL``).
    * ``--plain-llm`` other lineups → opus-4.5 (``DEFAULT_STACKER_MODEL``).
    * No ``--plain-llm`` → the default donated-key wrapper, whose primary is also
      ``DEFAULT_STACKER_MODEL`` (opus-4.5).

    All three callers (``_stage_llm_stacker`` write/read, ``_stage_score`` read,
    ``_hydrate_working_set_from_cache`` read) route through here so they agree.
    """
    use_prod_stacker = args.plain_llm and args.lineup == "prod"
    model = PROD_STACKER_MODEL if use_prod_stacker else DEFAULT_STACKER_MODEL
    return model_slug_to_filename(model)


def _stacker_batch_kwargs(
    args: argparse.Namespace,
    *,
    stacker_slug: str | None,
    force: bool,
) -> dict[str, Any]:
    """Wire --plain-llm and --no-stacker-fallback into ``run_stacker_batch``'s kwargs.

    An omitted ``stacker_llm`` / ``fallback_stacker_llm`` key leaves the runner on its
    own donated-key-wrapped defaults, which is why the keys are added conditionally
    rather than passed as None: an explicit None means "no fallback chain".
    ``--lineup prod`` uses the prod stacker (mirroring the prod-ish forecaster posture);
    other lineups keep the opus default.
    """
    stacker_llm_kwarg: GeneralLlm | None = None
    fallback_llm_kwarg: GeneralLlm | None = None
    if args.plain_llm:
        if args.lineup == "prod":
            stacker_llm_kwarg = GeneralLlm(model=PROD_STACKER_MODEL, **_PROD_STACKER_KWARGS)
        else:
            stacker_llm_kwarg = GeneralLlm(model=DEFAULT_STACKER_MODEL, **_OPUS_STACKER_KWARGS)
        if not args.no_stacker_fallback:
            fallback_llm_kwarg = GeneralLlm(model=DEFAULT_STACKER_FALLBACK_MODEL, **_OPENAI_STACKER_KWARGS)

    batch_kwargs: dict[str, Any] = {
        "stacker_slug": stacker_slug,
        "force": force,
        "concurrency": args.concurrency,
    }
    if stacker_llm_kwarg is not None:
        batch_kwargs["stacker_llm"] = stacker_llm_kwarg
    if args.no_stacker_fallback:
        batch_kwargs["fallback_stacker_llm"] = None
    elif fallback_llm_kwarg is not None:
        batch_kwargs["fallback_stacker_llm"] = fallback_llm_kwarg
    return batch_kwargs
