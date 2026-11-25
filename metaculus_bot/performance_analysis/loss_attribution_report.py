"""Markdown rendering for the loss-attribution report."""

from __future__ import annotations

from metaculus_bot.performance_analysis.markdown import markdown_table


def _interval(value: list[float] | tuple[float, float] | None) -> str:
    return "(n too small)" if value is None else f"[{_signed(value[0])}, {_signed(value[1])}]"


def _signed(value: float) -> str:
    return f"{0.0 if round(value, 2) == 0 else value:+.2f}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{0.0 if round(value, 3) == 0 else value:.3f}"


def _accounting(era: dict) -> list[str]:
    lines = ["### Points lost", ""]
    for name, rows in era["accounting"].items():
        lines += [
            f"#### {name}",
            "",
            f"Eligible: {sum(row['n'] for row in rows) + sum(era['exclusions'][name].values())}; excluded: {era['exclusions'][name]}; excluded spot peer: {_number(era['excluded_spot_peer'][name])}.",
            "",
        ]
        lines += markdown_table(
            [
                "bin",
                "n",
                "sum spot peer",
                "mean spot peer",
                "95% interval",
                "share of negative spot peer",
                "fraction negative",
            ],
            [
                [
                    row["label"],
                    str(row["n"]),
                    _number(row["sum_spot_peer"]),
                    _number(row["mean_spot_peer"]),
                    _interval(row["interval"]),
                    _number(row["negative_share"]),
                    _number(row["fraction_negative"]),
                ]
                for row in rows
            ],
        )
        lines.append("")
    return lines


def _calibration(era: dict) -> list[str]:
    lines = [
        "### CORP decomposition and reliability",
        "",
        f"Excluded binary records: {era['exclusions']['binary: outcome diagnostics']}; excluded MC records: {era['exclusions']['multiple_choice: outcome diagnostics']}.",
        "",
    ]
    for report in era["calibration"]:
        lines += [f"#### {report['name']}", "", f"Tied-top published ballots: {report['tied_top']}.", ""]
        lines += markdown_table(
            ["score", "n", "mean loss", "MCB", "DSC", "UNC"],
            [
                [
                    name,
                    str(report[name]["n"]),
                    *[_number(report[name][key]) for key in ("mean_score", "mcb", "dsc", "unc")],
                ]
                for name in ("brier", "log")
            ],
        )
        lines += [
            "",
            *markdown_table(
                ["bin", "n", "mean forecast", "observed frequency", "Jeffreys 95%", "observed minus forecast"],
                [
                    [
                        row["label"],
                        str(row["n"]),
                        _number(row["mean_forecast"]),
                        _number(row["observed"]),
                        _interval(row["jeffreys_interval"]),
                        _number(row["observed_minus_forecast"]),
                    ]
                    for row in report["reliability"]
                ],
            ),
            "",
        ]
    return lines


def _floor_comparison(era: dict) -> list[str]:
    floor = era["floor"]
    return [
        "### Floor calibration comparison",
        "",
        f"Excluded MC records: {era['exclusions']['multiple_choice: outcome diagnostics']}.",
        "",
        f"Options at or below 0.0101: {floor['options']} on {floor['questions']} questions; "
        f"{floor['hits']} resolved on those options, with {floor['expected_hits']:.2f} expected from their probabilities; "
        f"Jeffreys 95% {_interval(floor['interval'])}.",
        "",
        "",
    ]


def _discrimination(era: dict) -> list[str]:
    rows = era["discrimination"]
    return [
        "### Multiple-choice discrimination",
        "",
        f"Excluded MC outcome records: {era['exclusions']['multiple_choice: outcome diagnostics']}.",
        "not run: no --telemetry-archive" if era["ghost"] is None else "",
        "",
        *markdown_table(
            [
                "forecaster",
                "n",
                "incomplete",
                "zero probability",
                "tied top",
                "scored n",
                "paired n",
                "top accuracy",
                "Jeffreys 95%",
                "mean normalized rank",
                "mean winner probability",
                "mean counterfactual spot peer",
                "mean delta",
                "delta 95%",
                "member only top",
                "published only top",
            ],
            [
                [
                    row["name"],
                    str(row["n"]),
                    str(row["incomplete"]),
                    str(row["zero_probability"]),
                    str(row["tied_top"]),
                    str(row["scored_n"]),
                    str(row["paired_n"]),
                    _number(row["top_accuracy"]),
                    _interval(row["top_interval"]),
                    _number(row["mean_rank"]),
                    _number(row["mean_winner_probability"]),
                    _number(row["mean_counterfactual_spot"]),
                    _number(row["mean_delta"]),
                    _interval(row["delta_interval"]),
                    str(row["member_only_top"]),
                    str(row["published_only_top"]),
                ]
                for row in rows
            ],
        ),
        "",
    ]


def _ghosts(era: dict) -> list[str]:
    ghost = era["ghost"]
    if ghost is None:
        return ["### Ghost comparison", "", "not run: no --telemetry-archive", ""]
    return [
        "### Ghost comparison",
        "",
        f"Unjoined records: {ghost['unjoinable']}; unscoreable joined ghosts: {ghost['unscoreable']}, including {ghost['zero_probability']} with non-positive outcome probability.",
        "",
        *markdown_table(
            ["type", "status counts", "excluded spot peer"],
            [
                [row["question_type"], str(row["statuses"]), _number(row["excluded_spot_peer"])]
                for row in ghost["counts_by_type"]
            ],
        ),
        "",
        *markdown_table(
            ["type", "distance", "n", "mean spot-peer change", "95% interval"],
            [
                [
                    row["question_type"],
                    row["label"],
                    str(row["n"]),
                    _number(row["mean_delta"]),
                    _interval(row["interval"]),
                ]
                for row in ghost["rows"]
            ],
        ),
        "",
    ]


def _slices(era: dict) -> list[str]:
    return [
        "### Descriptive slice ranking",
        "",
        "These comparisons have no multiplicity correction.",
        f"Excluded: {era['exclusions']['slices and worst decile']}; research slices additionally exclude {era['exclusions']['research slices']} with spot peer {_number(era['excluded_spot_peer']['research slices'])}.",
        "",
        *markdown_table(
            ["dimension", "value", "n", "mean spot peer", "95% interval"],
            [
                [
                    row["dimension"],
                    row["value"],
                    str(row["n"]),
                    _number(row["mean_spot_peer"]),
                    _interval(row["interval"]),
                ]
                for row in era["slices"]
            ],
        ),
        "",
        "### Worst decile",
        "",
        *markdown_table(
            ["question", "type", "era", "spot peer", "title"],
            [
                [
                    f"[{row['question_id']}]({row['url']})",
                    row["type"],
                    row["era"],
                    _number(row["spot_peer"]),
                    str(row["title"]).replace("|", "\\|"),
                ]
                for row in era["worst"]
            ],
        ),
        "",
    ]


def render_report(payload: dict) -> str:
    """Render the computed results without recalculating statistics."""
    header = payload["header"]
    lines = [
        "# Loss attribution",
        "",
        f"- Dataset: `{header['dataset']}`",
        f"- Counts before exclusions: {header['before']}",
        f"- Counts after exclusions: {header['after']}; Mantic excluded: {header['mantic_excluded']}; cohort counts: {header['cohorts']}; total question exclusions: {header['excluded']}",
        f"- Era field: {header['era_field']}; bootstrap seed {header['seed']}, draws {header['draws']}.",
        f"- Records missing spot peer, excluded from score tables: {header['missing_spot_peer']}.",
        f"- Archived ghosts with no included record, by ghost type: {header['unjoined_archived_ghosts']}. Their eras are unknown.",
        "- Cluster intervals resample UTC resolution days and are shown only for at least 10 records and 5 days.",
        "- Alternatives assigning zero or negative probability to the outcome are excluded from score changes and counted.",
        "- A winner tied for maximum probability ranks top in accuracy and dissent counts; rank averages tied positions. MC top-option CORP uses the maximum probability and whether any tied top option won.",
        f"- MC records: {header['mc_all']} all eras, {header['mc_fall']} in triple_fall_config, {header['mc_joined_ghost']} with a joined ghost.",
        "- Per-member rows are diagnostic only, not a member ranking (decision-15). Dissenter counts cover both directions (decision-11).",
        f"- Decision-31 rejected the per-member MC audit because stored comments lacked vectors; complete member vectors appear on {header['mc_complete_member_vectors_before']} of {header['before'].get('multiple_choice', 0)} MC records before exclusions and {header['mc_complete_member_vectors']} of {header['mc_all']} afterward.",
        "- Decision-2's MC 0-5% bucket and the clip sweep are not re-run here. Any resulting change still needs an out-of-sample era test.",
        "",
    ]
    for era in payload["eras"]:
        lines += [
            f"## Era: {era['era']}",
            "",
            *_accounting(era),
            *_calibration(era),
            *_floor_comparison(era),
            *_discrimination(era),
            *_ghosts(era),
            *_slices(era),
        ]
    return "\n".join(lines)
