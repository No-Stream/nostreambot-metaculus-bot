"""Gap-fill v2 seam — wires the bounded agentic research loop into orchestration.

Keeps ``ResearchOrchestrator`` thin: this module owns prompt/tool/config
construction and the soft-fail boundary, and the orchestrator just gathers the
returned section. Contract mirrors v1 (``research/targeted.py``
``run_gap_fill_pass``): never raises, returns ``""`` when disabled,
benchmarking, unsupported question type, or on any failure.

Two callbacks mirror the orchestrator's ``research_sink`` pattern. ``archive_sink``
receives the loop transcript, telemetry and ghost when the loop actually ran (the
findings string alone is not enough for the research-archive trace requirement).
``ghost_context_sink`` receives the loop's :class:`GhostContext` when the plain
ghost ran, so the stage can issue the v1 ghost (``run_gap_fill_v2_ghost_v1``)
once gap-fill v1's section has landed: v1 and v2 run concurrently, so the loop
itself never sees that section (docs/agentic_gap_fill.md "The ghost forecast").
"""

import dataclasses
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forecasting_tools.data_models.questions import MetaculusQuestion

from metaculus_bot.constants import (
    GAP_FILL_V2_CONCLUDE_THRESHOLD,
    GAP_FILL_V2_DRIVER_EFFORT,
    GAP_FILL_V2_DRIVER_MODEL,
    GAP_FILL_V2_ENABLED_ENV,
    GAP_FILL_V2_MAX_GAPS,
    GAP_FILL_V2_MAX_TOOL_CALLS,
    GAP_FILL_V2_WALL_DEADLINE,
    env_flag_enabled,
)
from metaculus_bot.research.agentic import (
    GhostContext,
    GhostForecast,
    LoopConfig,
    build_gap_fill_tools,
    run_agentic_loop,
    run_ghost_v1,
)
from metaculus_bot.research.agentic.driver_prompt import (
    SupportedQuestion as _SupportedQuestion,
)
from metaculus_bot.research.agentic.driver_prompt import (
    build_ghost_prompt,
    build_ghost_v1_prompt,
    build_system_prompt,
    build_user_brief,
)
from metaculus_bot.research.agentic.tools import question_ladder_context
from metaculus_bot.research.fetch_ladder import guard
from metaculus_bot.research.image_persistence import persist_image_views

__all__ = ["run_gap_fill_v2", "run_gap_fill_v2_ghost_v1"]

logger = logging.getLogger(__name__)

# Broad by design, v1's policy (targeted.py _GAP_FILL_SOFT_FAIL_EXCEPTIONS): an enrichment layer never costs the forecast.
_GAP_FILL_V2_SOFT_FAIL_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


def _question_ref(question: MetaculusQuestion) -> str:
    return question.page_url or str(question.id_of_question)


async def run_gap_fill_v2(
    question: MetaculusQuestion,
    bundle_markdown: str,
    *,
    is_benchmarking: bool,
    archive_sink: Callable[[dict[str, Any]], None] | None = None,
    ghost_context_sink: Callable[[GhostContext], None] | None = None,
    on_error: Callable[[BaseException], None] | None = None,
    image_output_dir: Path | str = "research_outputs",
) -> str:
    """Run the agentic gap-fill v2 loop and return its findings section.

    Returns ``""`` with zero LLM calls when ``GAP_FILL_V2_ENABLED`` is off, when benchmarking
    (live search sees post-resolution information, the prediction-market provider's leakage
    rule), or for a question type the dry-run scaffold has no template for; soft-fails to
    ``""`` on any error. ``archive_sink`` receives ``{"transcript", "telemetry", "ghost"}`` when
    the loop ran, empty-findings runs included (``ghost`` is None when the ghost phase did not
    run); ``ghost_context_sink`` receives the loop's ``GhostContext`` when the plain ghost ran.
    ``on_error`` fires only when prompt or tool CONSTRUCTION crashes, the one crash path with
    no marker and no payload, never on the flag-off, benchmarking or unsupported-type skips.
    """
    if not env_flag_enabled(GAP_FILL_V2_ENABLED_ENV):
        return ""
    if is_benchmarking:
        return ""
    if not isinstance(question, _SupportedQuestion):
        # Every type the bot forecasts has a template, so only a conditional question reaches this branch.
        logger.info(
            "Gap-fill v2 skipped: unsupported question type %s",
            type(question).__name__,
        )
        return ""
    try:
        # UTC, so the driver's "today" matches the forecaster bundle's "Today:" line regardless of host timezone.
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        system_prompt = build_system_prompt(today)
        user_brief = build_user_brief(question, bundle_markdown)
        # ONE fetch-ladder session and context per question, so known-API tools and rung 0 share
        # both the connector and the Kalshi detail-GET budget across the driver's tool calls.
        async with guard._get_session() as session:
            tools = build_gap_fill_tools(
                question.question_text,
                ctx=question_ladder_context(session=session),
            )
            question_ref = _question_ref(question)
            config = LoopConfig(
                model=GAP_FILL_V2_DRIVER_MODEL,
                reasoning_effort=GAP_FILL_V2_DRIVER_EFFORT,
                max_tool_calls=GAP_FILL_V2_MAX_TOOL_CALLS,
                wall_deadline_s=GAP_FILL_V2_WALL_DEADLINE,
                conclude_threshold_s=GAP_FILL_V2_CONCLUDE_THRESHOLD,
                max_gaps=GAP_FILL_V2_MAX_GAPS,
                question_ref=question_ref,
            )
            result = await run_agentic_loop(
                system_prompt,
                user_brief,
                tools,
                config,
                ghost_prompt=build_ghost_prompt(),
                log_prefix=f"question={question_ref} ",
            )
        if archive_sink is not None:
            archive_payload: dict[str, Any] = {
                "transcript": result.transcript,
                "telemetry": dataclasses.asdict(result.telemetry),
                "ghost": result.ghost.model_dump() if result.ghost is not None else None,
            }
            if result.image_views:
                archive_payload["images"] = persist_image_views(
                    result.image_views,
                    result.image_sources,
                    output_dir=image_output_dir,
                    image_observations=result.image_observations,
                )
            archive_sink(archive_payload)
        if ghost_context_sink is not None and result.ghost_context is not None:
            ghost_context_sink(result.ghost_context)
        return result.findings_markdown
    except _GAP_FILL_V2_SOFT_FAIL_EXCEPTIONS as exc:
        logger.exception("Gap-fill v2 seam failed; continuing without v2 findings")
        if on_error is not None:
            on_error(exc)
        return ""


async def run_gap_fill_v2_ghost_v1(
    question: MetaculusQuestion, context: GhostContext, v1_addendum: str
) -> GhostForecast | None:
    """The v1 ghost: the plain ghost's brief plus gap-fill v1's section, on the loop's cached prefix.

    Issued by the stage once both passes have landed, because v1 and v2 run concurrently and
    the loop never sees v1's section. Telemetry only, never published, never raises; paired
    against the plain ghost it measures v1's marginal value on the driver.
    """
    return await run_ghost_v1(
        context, build_ghost_v1_prompt(v1_addendum), log_prefix=f"question={_question_ref(question)} "
    )
