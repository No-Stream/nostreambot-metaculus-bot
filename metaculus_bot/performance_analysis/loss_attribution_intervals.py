"""Resolution-day interval settings for loss-attribution diagnostics."""

import numpy as np

from metaculus_bot.performance_analysis.era_gap import ClusterSums

DEFAULT_SEED = 20261125
DEFAULT_DRAWS = 2000
MIN_N_FOR_INTERVAL = 10
MIN_CLUSTERS_FOR_INTERVAL = 5


def interval_allowed(n: int, clusters: list[str] | set[str]) -> bool:
    return n >= MIN_N_FOR_INTERVAL and len(set(clusters)) >= MIN_CLUSTERS_FOR_INTERVAL


def cluster_interval(
    values: list[float], clusters: list[str], *, seed: int = DEFAULT_SEED, draws: int = DEFAULT_DRAWS
) -> tuple[float, float] | None:
    """Return a resolution-day bootstrap interval when the sample is large enough."""
    if not interval_allowed(len(values), clusters):
        return None
    if draws <= 0:
        raise ValueError("draws must be positive")
    means = ClusterSums.from_values(values, clusters).resampled_means(np.random.default_rng(seed), draws)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])
