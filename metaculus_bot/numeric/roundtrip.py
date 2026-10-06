"""Compare original percentile declarations with built member grid CDFs."""

from collections.abc import Sequence

import numpy as np
from forecasting_tools import NumericDistribution, NumericQuestion, Percentile
from forecasting_tools.data_models.questions import DiscreteQuestion

from metaculus_bot.numeric.date_axis import numeric_qtype
from metaculus_bot.performance_analysis.analysis import pit_on_grid
from metaculus_bot.question_platform import question_platform


def format_percentile_roundtrip_marker(
    prediction: NumericDistribution,
    stated: Sequence[Percentile],
    question: NumericQuestion,
    *,
    model: str,
) -> str:
    """Compare stated probabilities with point CDFs or discrete bin probability intervals."""
    cdf = prediction.get_cdf()
    grid = np.asarray([point.value for point in cdf], dtype=float)
    heights = np.asarray([point.percentile for point in cdf], dtype=float)
    if isinstance(question, DiscreteQuestion):
        evaluated: list[tuple[Percentile, float]] = []
        for point in stated:
            upper_edge_index = int(np.searchsorted(grid, point.value, side="right"))
            lower_height = 0.0 if upper_edge_index == 0 else float(heights[upper_edge_index - 1])
            upper_height = 1.0 if upper_edge_index == len(grid) else float(heights[upper_edge_index])
            evaluated.append((point, min(max(point.percentile, lower_height), upper_height)))
    else:
        evaluated = [(point, pit_on_grid(point.value, grid, heights, None)[0]) for point in stated]
    worst, cdf_at_v = max(evaluated, key=lambda item: abs(item[1] - item[0].percentile))
    qtype = "discrete" if isinstance(question, DiscreteQuestion) else numeric_qtype(question)
    return (
        f"PERCENTILE_ROUNDTRIP: question={question.id_of_question} model={model} "
        f"qtype={qtype} platform={question_platform(question)} "
        f"max_abs_drift={abs(cdf_at_v - worst.percentile):.9f} p={worst.percentile:.9f} "
        f"v={worst.value:.17g} cdf_at_v={cdf_at_v:.9f} point_count={len(stated)}"
    )
