"""Aggregation pipeline: stacking fallback chain and simple combine logic.

Extracted from main.py to keep TemplateForecaster focused on orchestration.
Owns all per-question stacking state (outcomes, meta reasoning, counters).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from forecasting_tools import (
    BinaryQuestion,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.data_models.data_organizer import PredictionTypes
from forecasting_tools.data_models.questions import DateQuestion

from metaculus_bot import calibration, stacking
from metaculus_bot.aggregation_strategies import (
    AggregationStrategy,
    combine_binary_predictions,
    combine_multiple_choice_predictions,
    combine_numeric_predictions,
)
from metaculus_bot.constants import STACKER_FALLBACK_SOFT_DEADLINE, STACKER_SOFT_DEADLINE
from metaculus_bot.exceptions import UnitMismatchError
from metaculus_bot.llm_configs import STACKER_FALLBACK_LLM
from metaculus_bot.member_forecast import (
    MEMBER_FORECAST_ROLE_STACKER,
    NUMERIC_COMBINE_METHOD_SINGLE,
    NUMERIC_COMBINE_METHOD_STACKED,
    format_member_forecast_marker,
    out_of_range_mass,
    percentile_pairs,
)
from metaculus_bot.numeric.config import elicit_per_bin
from metaculus_bot.numeric.date_axis import numeric_qtype, numeric_view
from metaculus_bot.numeric.diagnostics import log_final_prediction, log_open_bound_piling_diagnostics
from metaculus_bot.numeric.pipeline import build_numeric_distribution, sanitize_percentiles
from metaculus_bot.numeric.utils import bound_messages
from metaculus_bot.numeric.validation import detect_unit_mismatch
from metaculus_bot.post_processing import apply_platt_calibration, apply_thin_publish_floor, maybe_snap_to_integers

logger = logging.getLogger(__name__)


@dataclass
class AggregationCounters:
    stacking_expected_combine_count: int = 0
    stacking_unexpected_combine_count: int = 0
    stacking_fallback_count: int = 0
    stacker_primary_failed_count: int = 0
    stacker_fallback_used_count: int = 0
    stacker_fallback_failed_count: int = 0
    conditional_stacking_triggered_count: int = 0
    conditional_stacking_skipped_count: int = 0
    conditional_stacking_skipped_single_forecaster_count: int = 0
    conditional_stacking_crux_failures: int = 0
    conditional_stacking_search_failures: int = 0


@dataclass
class AggregationPipeline:
    strategy: AggregationStrategy
    stacker_llm: GeneralLlm | None
    parser_llm: GeneralLlm
    stacking_fallback_on_failure: bool = True
    stacking_randomize_order: bool = True
    stacking_spread_thresholds: dict[str, float] = field(default_factory=dict)
    discrete_integer_votes: defaultdict[int, list[bool]] = field(default_factory=lambda: defaultdict(list))

    # Per-question state
    meta_reasoning: dict[int, str] = field(default_factory=dict)
    outcomes: dict[int, str] = field(default_factory=dict)
    # Why a "skipped"/"skipped_config_off" outcome skipped: "spread_below_threshold",
    # "config_off", or "single_forecaster". Only the skip paths in stacking_route
    # write it; it rides the comment as the additive STACKER_SKIP_REASON marker, so
    # the single-forecaster short-circuit (which computes no spread at all) stops
    # reading identically to a spread-below-threshold skip in the published record.
    skip_reasons: dict[int, str] = field(default_factory=dict)
    # The NUMERIC_AGGREGATE ``method`` per question (tokens in member_forecast.py); popped by the forecaster's seam.
    numeric_combine_methods: dict[int, str] = field(default_factory=dict)
    expected_base_combines: set[int] = field(default_factory=set)
    counters: AggregationCounters = field(default_factory=AggregationCounters)

    def get_threshold_for_question(self, question: MetaculusQuestion) -> float:
        """The spread threshold for the question's type; a date question's spread is numeric (range-normalised)."""
        if isinstance(question, BinaryQuestion):
            return self.stacking_spread_thresholds["binary"]
        if isinstance(question, MultipleChoiceQuestion):
            return self.stacking_spread_thresholds["mc"]
        if isinstance(question, (NumericQuestion, DateQuestion)):
            return self.stacking_spread_thresholds["numeric"]
        raise ValueError(f"No spread threshold for question type: {type(question).__name__}")

    def register_expected_base_combine(self, question: MetaculusQuestion) -> None:
        qid = question.id_of_question
        assert qid is not None, "register_expected_base_combine requires question.id_of_question"
        self.expected_base_combines.add(qid)

    async def run_stacking(
        self,
        question: MetaculusQuestion,
        research: str,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]],
        *,
        stacker_llm_override: GeneralLlm | None = None,
        aggregated_tool_output: str | None = None,
        stacker_wall_timeout: float = STACKER_SOFT_DEADLINE,
    ) -> PredictionTypes:
        """Dispatch stacker LLM per question type.

        ``stacker_wall_timeout`` is the hard wall-clock cap handed to the
        per-type stacker invoke (via ``invoke_with_transient_retry``). Callers
        pass the deadline matching the attempt: STACKER_SOFT_DEADLINE for the
        primary stacker, STACKER_FALLBACK_SOFT_DEADLINE for the fallback.
        """
        if stacker_llm_override is not None:
            stacker_llm = stacker_llm_override
        else:
            if self.stacker_llm is None:
                raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
            stacker_llm = self.stacker_llm

        page_url = question.page_url or "<unknown>"
        qid = question.id_of_question
        assert qid is not None, "run_stacking requires question.id_of_question"

        base_predictions = [stacking.strip_model_tag(pred.reasoning) for pred in reasoned_predictions]

        if self.stacking_randomize_order:
            combined = list(zip(base_predictions, reasoned_predictions, strict=True))
            random.shuffle(combined)
            base_predictions = [bp for bp, _ in combined]
            reasoned_predictions = [rp for _, rp in combined]

        if isinstance(question, BinaryQuestion):
            value, meta_text = await stacking.run_stacking_binary(
                stacker_llm,
                self.parser_llm,
                question,
                research=research,
                base_texts=base_predictions,
                aggregated_tool_output=aggregated_tool_output,
                stacker_wall_timeout=stacker_wall_timeout,
            )
            self.meta_reasoning[qid] = meta_text
            logger.info(f"Stacked binary prediction for {page_url}: {value}")
            return value
        if isinstance(question, MultipleChoiceQuestion):
            pol, meta_text = await stacking.run_stacking_mc(
                stacker_llm,
                self.parser_llm,
                question,
                research=research,
                base_texts=base_predictions,
                aggregated_tool_output=aggregated_tool_output,
                stacker_wall_timeout=stacker_wall_timeout,
            )
            self.meta_reasoning[qid] = meta_text
            logger.info(f"Stacked multiple choice prediction for {page_url}: {pol}")
            return pol
        if isinstance(question, (NumericQuestion, DateQuestion)):
            # A date question stacks on its epoch-seconds view. Reachable only with
            # NUMERIC_STACKING_ENABLED on (stacking_route gates dates on the numeric flag,
            # off in prod); the stacker's numeric prompt and extraction are float-based, so a
            # date stack that fails to parse degrades to MEDIAN through the ladder above.
            return await self._run_stacking_numeric(
                numeric_view(question),
                research,
                base_predictions,
                stacker_llm=stacker_llm,
                qid=qid,
                page_url=page_url,
                aggregated_tool_output=aggregated_tool_output,
                stacker_wall_timeout=stacker_wall_timeout,
            )
        raise ValueError(f"Unsupported question type for stacking: {type(question)}")

    async def _run_stacking_numeric(
        self,
        question: NumericQuestion,
        research: str,
        base_predictions: list[str],
        *,
        stacker_llm: GeneralLlm,
        qid: int,
        page_url: str,
        aggregated_tool_output: str | None,
        stacker_wall_timeout: float,
    ) -> PredictionTypes:
        """Stack a numeric question: percentiles -> sanitize -> PCHIP CDF -> unit guard.

        The CDF is built before the unit-mismatch guard so the stacker's MEMBER_FORECAST line
        can carry the built CDF's out-of-range mass; the guard still withholds the same
        distributions, and a withheld stacker still leaves its line.
        """
        upper_msg, lower_msg = bound_messages(question)
        perc_list, meta_text = await stacking.run_stacking_numeric(
            stacker_llm,
            self.parser_llm,
            question,
            research=research,
            base_texts=base_predictions,
            lower_bound_message=lower_msg,
            upper_bound_message=upper_msg,
            aggregated_tool_output=aggregated_tool_output,
            stacker_wall_timeout=stacker_wall_timeout,
        )
        self.meta_reasoning[qid] = meta_text

        percentile_list, zero_point = sanitize_percentiles(list(perc_list), question, model_name=stacker_llm.model)
        prediction = build_numeric_distribution(percentile_list, question, zero_point, model_name=stacker_llm.model)
        logger.info(
            format_member_forecast_marker(
                question_id=qid,
                model=stacker_llm.model,
                role=MEMBER_FORECAST_ROLE_STACKER,
                qtype=numeric_qtype(question),
                raw=percentile_pairs(perc_list),
                published=percentile_pairs(percentile_list),
                out_of_range=out_of_range_mass(prediction),
            )
        )

        mismatch, reason = detect_unit_mismatch(percentile_list, question)  # type: ignore[arg-type]
        if mismatch:
            logger.error(
                f"Unit mismatch likely for Q {qid} | URL {page_url} | reason={reason}. Withholding prediction."
            )
            raise UnitMismatchError(
                f"Unit mismatch likely; {reason}. Values: {[float(p.value) for p in percentile_list]}"
            )

        log_open_bound_piling_diagnostics(prediction, question, stacker_llm.model, percentile_list)
        log_final_prediction(prediction, question)
        logger.info(f"Stacked numeric prediction for {page_url}")
        return prediction

    def base_combine(  # noqa: PLR0912  # explicit type/state branches preserve the base-combine lifecycle
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        """Combine report outputs after routing has selected the base path."""
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        qkey = question.id_of_question

        expected = qkey in self.expected_base_combines
        if expected:
            self.expected_base_combines.discard(qkey)
            self.counters.stacking_expected_combine_count += 1
        else:
            self.counters.stacking_unexpected_combine_count += 1

        if len(predictions) == 1:
            if expected:
                logger.info("STACKING base combine: single pre-stacked output; returning as-is")
            else:
                logger.warning("Unexpected STACKING combine: single input without stacking context; returning as-is")
            lone = predictions[0]
            # Single-survivor binary publish floor. This branch serves two different
            # objects — the lone RAW member the single-forecaster short-circuit hands
            # through, and the single PRE-STACKED output of a fired stacker — so the
            # count alone is not the trigger: the skip reason route_after_forecasts
            # writes for exactly the first event is, and the stacked path never writes
            # one. Read, not popped: the comment builder pops it to publish
            # STACKER_SKIP_REASON. Binary only — a lone numeric survivor keeps its snap
            # path below and a lone MC survivor is returned as is.
            #
            # Three asymmetries are accepted here rather than papered over with a second
            # wiring. (1) skip_reasons is written only under STACKING /
            # CONDITIONAL_STACKING, so a single survivor under plain MEAN/MEDIAN routes
            # through simple_combine and is NOT floored; prod and the code default
            # both run CONDITIONAL_STACKING. (2) outcomes[qid] is overwritten by every
            # routing path, while skip_reasons[qid] is written only by the skip paths and
            # cleared only by the comment builder — so a stale reason could reach this
            # read only if a run died between routing and comment-building AND the same
            # qid were routed again on the same pipeline instance, which no entrypoint
            # does. Clearing it on the stacked path would just trade that for the mirror
            # case: with research_reports_per_question > 1 the reports share this
            # instance, so one report's failed stack attempt would erase the reason a
            # sibling report's genuine lone survivor wrote, un-flooring the value the
            # framework then publishes and losing its STACKER_SKIP_REASON marker.
            # (3) This branch returns above the Platt tail at the bottom of the method, so
            # a lone raw binary member is the only published binary path that never
            # receives Platt calibration — the exact mirror of simple_combine, which
            # calibrates a single prediction but never floors it. Inert today: no workflow
            # sets PLATT_CALIBRATION_ENABLED, and the standing operator decision is that
            # fitted calibration layers are not a lever, so this is recorded rather than
            # wired. Whoever turns Platt on decides whether a floored single-survivor value
            # should also be calibrated and in which order (see backlog decision-1 "Calibration and post-hoc layers"
            # entry, which carries the same note).
            if (
                isinstance(question, BinaryQuestion)
                and qkey is not None
                and self.skip_reasons.get(qkey) == "single_forecaster"
            ):
                return self._floor_single_survivor_binary(cast(float, lone), qkey)
            # setdefault: a pre-stacked output re-enters here after stack_predictions already recorded it.
            if isinstance(question, (NumericQuestion, DateQuestion)) and qkey is not None:
                self.numeric_combine_methods.setdefault(qkey, NUMERIC_COMBINE_METHOD_SINGLE)
            # Snap-to-integers for a lone numeric prediction — the
            # min-forecasters=1 single-survivor path (forecaster.py short-circuits
            # spread + stacking and hands the raw prediction through). No-op for
            # binary/MC and for the pre-stacked STACKING output (whose discrete
            # votes were already consumed in stack_predictions, so the vote list
            # is empty and majority_votes_discrete([]) is False).
            return self._maybe_snap_to_integers(lone, question)

        # CONDITIONAL_STACKING uses MEDIAN; regular STACKING uses MEAN; per-bin members are pooled
        base_combine_strategy = self._numeric_combine_strategy(
            question,
            AggregationStrategy.MEDIAN
            if self.strategy == AggregationStrategy.CONDITIONAL_STACKING
            else AggregationStrategy.MEAN,
        )
        strategy_name = base_combine_strategy.value
        self._log_base_combine_strategy(len(predictions), strategy_name, expected=expected)

        apply_platt_after_combine = self.strategy == AggregationStrategy.CONDITIONAL_STACKING

        first = predictions[0]
        combined = self._combine_by_type(
            predictions, question, base_combine_strategy, error_context="STACKING base combine"
        )
        if isinstance(first, (int, float)):
            values = [float(p) for p in predictions if isinstance(p, (int, float))]
            logger.info("STACKING base combine: binary %s of %s = %.3f", strategy_name, values, combined)
        elif isinstance(combined, PredictedOptionList):
            summary = {o.option_name: round(o.probability, 4) for o in combined.predicted_options}
            logger.info("STACKING base combine: MC %s aggregation | %s", strategy_name, summary)
        else:
            logger.info(
                "STACKING base combine: numeric %s aggregation | CDF points=%d",
                strategy_name,
                len(getattr(combined, "cdf", [])),
            )
            combined = self._maybe_snap_to_integers(combined, question)

        if apply_platt_after_combine:
            return self._apply_platt_calibration(combined, question)
        return combined

    @staticmethod
    def _floor_single_survivor_binary(raw: float, qid: int) -> float:
        """Apply the k=1 publish floor and log the move, if any, as THIN_PUBLISH_FLOOR.

        The value is a float by construction on this path: a BinaryQuestion's members
        come out of run_binary_forecast as ``ReasonedPrediction[float]``. No line when the
        lone value already sits inside the band — the single-survivor EVENT is already
        observable via FORECASTERS_SURVIVED and the skip reason, so silence here means
        exactly "nothing moved".
        """
        clamped = apply_thin_publish_floor(raw, survivors=1)
        if clamped != raw:
            logger.warning(
                "THIN_PUBLISH_FLOOR: question=%s raw=%.4f clamped=%.4f survivors=1",
                qid,
                raw,
                clamped,
            )
        return clamped

    def _log_base_combine_strategy(self, n_predictions: int, strategy_name: str, *, expected: bool) -> None:
        """One line naming how many pre-stacked outputs are being combined, and by what.

        An UNEXPECTED combine is the interesting case: it means the parent class re-entered
        aggregate without stacking context, so it logs at WARNING.
        """
        if expected:
            logger.info(
                "STACKING base combine: %d pre-stacked outputs; aggregating by %s for final output",
                n_predictions,
                strategy_name,
            )
        else:
            logger.warning(
                "Unexpected STACKING combine: %d inputs without stacking context; aggregating by %s",
                n_predictions,
                strategy_name,
            )

    async def stack_predictions(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        *,
        research: str | None,
        reasoned_predictions: list[ReasonedPrediction[PredictionTypes]] | None,
        aggregated_tool_output: str | None = None,
    ) -> PredictionTypes:
        """Run the primary/fallback stacker ladder for one research report."""
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        if self.stacker_llm is None:
            raise ValueError("STACKING aggregation strategy requires a stacker LLM to be configured")
        if reasoned_predictions is None:
            raise ValueError("STACKING aggregation strategy requires reasoned predictions")
        if research is None:
            raise ValueError("STACKING aggregation strategy requires research context")

        qid_for_outcome = question.id_of_question
        assert qid_for_outcome is not None

        try:
            stacked = await asyncio.wait_for(
                self.run_stacking(
                    question,
                    research,
                    reasoned_predictions,
                    aggregated_tool_output=aggregated_tool_output,
                    stacker_wall_timeout=STACKER_SOFT_DEADLINE,
                ),
                timeout=STACKER_SOFT_DEADLINE,
            )
            self.outcomes[qid_for_outcome] = "primary"
            return self._adopt_stacked(stacked, question)
        # Deliberate fallback ladder: ANY primary-stacker failure (timeout, API, parse)
        # degrades to the fallback LLM rather than dropping the question.
        except Exception as primary_exc:  # HARNESS-SCAN-EXEMPT-broad-except
            if not self.stacking_fallback_on_failure:
                raise

            self.counters.stacker_primary_failed_count += 1
            logger.warning(
                "STACKER_PRIMARY_FAILED: primary stacker failed on Q %s (%s: %s); trying fallback model",
                question.id_of_question,
                type(primary_exc).__name__,
                primary_exc,
            )

            try:
                self.counters.stacker_fallback_used_count += 1
                stacked = await asyncio.wait_for(
                    self.run_stacking(
                        question,
                        research,
                        reasoned_predictions,
                        stacker_llm_override=STACKER_FALLBACK_LLM,
                        aggregated_tool_output=aggregated_tool_output,
                        stacker_wall_timeout=STACKER_FALLBACK_SOFT_DEADLINE,
                    ),
                    timeout=STACKER_FALLBACK_SOFT_DEADLINE,
                )
                logger.info(
                    "STACKER_FALLBACK_SUCCEEDED: fallback stacker succeeded on Q %s",
                    question.id_of_question,
                )
                self.outcomes[qid_for_outcome] = "fallback_llm"
                return self._adopt_stacked(stacked, question)
            # Boundary, last rung of the ladder: fallback-stacker failure degrades to MEDIAN,
            # never drops the question. Narrowing here would turn an unexpected stacker
            # failure into a lost forecast.
            except Exception as fallback_exc:  # noqa: BLE001  # HARNESS-SCAN-EXEMPT-broad-except
                self.counters.stacker_fallback_failed_count += 1
                self.counters.stacking_fallback_count += 1
                logger.error(
                    "STACKER_FALLBACK_FAILED: fallback stacker also failed on Q %s (%s: %s); "
                    "falling back to MEDIAN aggregation",
                    question.id_of_question,
                    type(fallback_exc).__name__,
                    fallback_exc,
                )
                self.outcomes[qid_for_outcome] = "fallback_median"
                return self._median_fallback(predictions, question)

    def _median_fallback(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        combined = self._combine_by_type(
            predictions,
            question,
            self._numeric_combine_strategy(question, AggregationStrategy.MEDIAN),
            error_context="MEDIAN fallback",
        )
        return self._apply_platt_calibration(self._maybe_snap_to_integers(combined, question), question)

    def _adopt_stacked(self, stacked: PredictionTypes, question: MetaculusQuestion) -> PredictionTypes:
        """The stacker's output becomes the report's prediction: record the method, then post-process."""
        self._record_numeric_combine_method(question, NUMERIC_COMBINE_METHOD_STACKED)
        return self._apply_platt_calibration(self._maybe_snap_to_integers(stacked, question), question)

    @staticmethod
    def _numeric_combine_strategy(question: MetaculusQuestion, strategy: AggregationStrategy) -> AggregationStrategy:
        """The linear opinion pool for per-bin members; ``strategy`` unchanged for everything else.

        A member elicited per bin (``elicit_per_bin``: a Mantic enumerable grid) can be sharp, 0.99
        on one bin, and the pointwise MEDIAN of three sharp CDFs that disagree is the middle member's
        CDF outright: 0.99 on its bin and the platform floor on the bins the other two believed,
        the cliff a log score in the resolved bin punishes hardest (about -230 Series 1 points on
        question 651's 12-bin grid). The pointwise MEAN of the CDFs is the CDF of the mixture PMF,
        so every believed bin keeps a third of the mass a member gave it (+69), and
        ``aggregate_numeric(..., "mean")`` already computes it. The elicitation is decided per
        question, so an ensemble never mixes the two kinds of member; percentile members keep the
        median on both platforms, and binary and MC questions are untouched.
        """
        if isinstance(question, (NumericQuestion, DateQuestion)) and elicit_per_bin(numeric_view(question)):
            return AggregationStrategy.MEAN
        return strategy

    def _record_numeric_combine_method(self, question: MetaculusQuestion, method: str) -> None:
        """Remember how a numeric or date question's members were combined; the aggregation seam pops it."""
        qid = question.id_of_question
        if isinstance(question, (NumericQuestion, DateQuestion)) and qid is not None:
            self.numeric_combine_methods[qid] = method

    def simple_combine(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
    ) -> PredictionTypes:
        """Combine raw member predictions under a non-stacking strategy."""
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        first_prediction = predictions[0]
        logger.info(
            "Aggregating %s predictions with %s", self._prediction_type_label(first_prediction), self.strategy.value
        )

        effective_strategy = self._numeric_combine_strategy(
            question,
            AggregationStrategy.MEDIAN if self.strategy == AggregationStrategy.CONDITIONAL_STACKING else self.strategy,
        )

        combined = self._combine_by_type(predictions, question, effective_strategy, error_context="aggregation")
        if isinstance(first_prediction, (int, float)):
            float_preds = [float(p) for p in predictions if isinstance(p, (int, float))]
            if effective_strategy == AggregationStrategy.MEAN:
                logger.info("Binary question ensembling: mean of %s = %.3f (rounded)", float_preds, combined)
            elif effective_strategy == AggregationStrategy.MEDIAN:
                logger.info("Binary question ensembling: median of %s = %.3f", float_preds, combined)
            else:
                logger.info(
                    "Binary question ensembling: %s of %s = %.3f", effective_strategy.value, float_preds, combined
                )
            return self._apply_platt_calibration(combined, question)

        if isinstance(combined, PredictedOptionList):
            summary = {o.option_name: round(o.probability, 4) for o in combined.predicted_options}
            logger.info("MC %s aggregation; renormalized to 1.0 | %s", effective_strategy.value, summary)
            return self._apply_platt_calibration(combined, question)

        logger.info(
            "Numeric aggregation=%s | preserved bounds [%s, %s] | CDF points=%d",
            effective_strategy.value,
            getattr(question, "lower_bound", None),
            getattr(question, "upper_bound", None),
            len(getattr(combined, "cdf", [])),
        )
        return self._apply_platt_calibration(self._maybe_snap_to_integers(combined, question), question)

    @staticmethod
    def _prediction_type_label(prediction: PredictionTypes) -> str:
        if isinstance(prediction, (int, float)):
            return "binary"
        if isinstance(prediction, NumericDistribution):
            return "numeric"
        if isinstance(prediction, PredictedOptionList):
            return "multiple-choice"
        return type(prediction).__name__

    def _combine_by_type(
        self,
        predictions: list[PredictionTypes],
        question: MetaculusQuestion,
        strategy: AggregationStrategy,
        error_context: str,
    ) -> PredictionTypes:
        """Filter predictions to the first one's type and run the matching combiner.

        Shared dispatch core for ``base_combine`` / ``_median_fallback`` / ``simple_combine``;
        callers layer their own logging and post-processing (snap/platt) around the result.
        """
        first = predictions[0]
        if isinstance(first, (int, float)):
            values = [float(p) for p in predictions if isinstance(p, (int, float))]
            return combine_binary_predictions(values, strategy)  # type: ignore[return-value]
        if isinstance(first, NumericDistribution) and isinstance(question, (NumericQuestion, DateQuestion)):
            numeric_preds = [p for p in predictions if isinstance(p, NumericDistribution)]
            self._record_numeric_combine_method(question, strategy.value)
            # A date question's members live on the epoch-seconds axis; the combiner reads the
            # bounds and grid off the same view, so the ensemble CDF carries is_date too.
            return combine_numeric_predictions(numeric_preds, numeric_view(question), strategy)  # type: ignore[return-value]
        if isinstance(first, PredictedOptionList):
            mc_preds = [p for p in predictions if isinstance(p, PredictedOptionList)]
            return combine_multiple_choice_predictions(mc_preds, strategy)  # type: ignore[return-value]
        raise ValueError(f"Unsupported prediction type for {error_context}: {type(first)}")

    def _apply_platt_calibration(self, prediction: PredictionTypes, question: MetaculusQuestion) -> PredictionTypes:
        # Params read via the module attribute (not from-imported) so test monkeypatching
        # of metaculus_bot.calibration.{BINARY,MC}_PLATT_PARAMS is observed.
        return apply_platt_calibration(
            prediction, question, calibration.BINARY_PLATT_PARAMS, calibration.MC_PLATT_PARAMS
        )

    def _maybe_snap_to_integers(self, prediction: PredictionTypes, question: MetaculusQuestion) -> PredictionTypes:
        if not isinstance(prediction, NumericDistribution) or not isinstance(question, NumericQuestion):
            return prediction
        qid = question.id_of_question
        if qid is None:
            return prediction
        votes = self.discrete_integer_votes.pop(qid, [])
        return maybe_snap_to_integers(prediction, question, votes)
