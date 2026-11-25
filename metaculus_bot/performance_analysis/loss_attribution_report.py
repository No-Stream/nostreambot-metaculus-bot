"""Markdown rendering for the loss-attribution report."""

from __future__ import annotations

from metaculus_bot.performance_analysis.markdown import markdown_table

ACCOUNTING_SECTIONS: dict[str, tuple[str, str, str]] = {
    "binary: where the points went": (
        "Binary: where the points went (uses the outcome)",
        "p on outcome",
        "Binned by the probability we gave the outcome that happened; describes where points were won and lost.",
    ),
    "binary: by our confidence": (
        "Binary: by our confidence (outcome-blind)",
        "max(p, 1 - p)",
        "The same questions binned by our confidence, max(p, 1 - p).",
    ),
    "multiple_choice: where the points went": (
        "Multiple choice: where the points went (uses the outcome)",
        "p on winning option",
        "Binned by the probability we gave the option that won; describes where points were won and lost.",
    ),
    "multiple_choice: by our confidence": (
        "Multiple choice: by our confidence (outcome-blind)",
        "top-option p",
        "The same questions binned by the probability of our top option.",
    ),
    "binary: by p_yes": (
        "Binary: by probability of Yes (outcome-blind)",
        "p(Yes)",
        "Binary questions binned by our probability of Yes.",
    ),
    "numeric and discrete: placement in the distribution": (
        "Numeric and discrete: placement in the distribution (uses the outcome)",
        "PIT of resolved value",
        "Binned by our forecast CDF at the resolved value (PIT); about 80% should fall in [0.1, 0.9). "
        "Rows such as 'low, open' are resolutions outside the range.",
    ),
}

CALIBRATION_SECTIONS: dict[str, tuple[str, bool]] = {
    "binary p_yes": ("Binary: probability of Yes", False),
    "MC top option": ("Multiple choice: top option, one observation per question", True),
    "MC secondary: options within a ballot are dependent": (
        "Multiple choice, every option pooled (secondary: options on one question are not independent)",
        True,
    ),
}


def _how_to_read(header: dict) -> list[str]:
    """The reading guide, with the interval gate taken from the run header."""
    return [
        "## How to read this report",
        "",
        "- Scores are spot peer points per question: positive means we beat the average other forecaster on that "
        "question. Sums and means in the score tables are in these units.",
        "- Intervals on means are 95% bootstrap intervals that resample UTC resolution days. Intervals on rates are "
        f"95% Jeffreys intervals. '(n too small)' means fewer than {header['min_n_for_interval']} questions or fewer "
        f"than {header['min_clusters_for_interval']} distinct resolution "
        "days, so only the point value is shown.",
        "- Each era section repeats the same tables on that era's questions. 'all' pools every era, and '(none)' holds "
        "questions with no value in the era field.",
        "- Score tables by forecast probability: tables marked 'uses the outcome' are accounting. They bin by the "
        "probability we gave the outcome that happened, so they show where points were won and lost but not why: "
        "a low probability on the winner usually loses spot peer, though not always, since spot peer is measured "
        "against the other forecasters. Tables marked 'outcome-blind' bin by a quantity known before resolution; "
        "a bin with negative mean spot peer there shows a problem at that confidence level.",
        "- Score table columns: 'total' and 'mean' are spot peer; 'share of negative' is the bin's share of the "
        "table's summed negative spot peer (positive scores ignored); 'fraction negative' is the share of the bin's "
        "questions that scored below zero.",
        "- CORP decomposition (Dimitriadis, Gneiting and Jordan 2021) splits a mean loss as MCB - DSC + UNC. "
        "MCB (miscalibration) is the loss that recalibrating our probabilities would remove; lower is better. "
        "DSC (discrimination) is how well our forecasts separate outcomes that happened from ones that did not; "
        "higher is better. UNC (uncertainty) is the loss of always forecasting the base rate, a property of the "
        "questions. These are Brier and log losses (lower is better), not spot peer.",
        "- Multiple-choice discrimination columns: 'top option won' is the share of questions where the forecaster's "
        "highest-probability option won; 'mean winner rank' runs from 0 (winner ranked first) to 1 (ranked last), "
        "averaging tied positions; 'spot peer if published' prices that forecaster's vector on the realized option; "
        "'change vs published' is that minus our published spot peer on the same questions; the last two columns "
        "count questions where only that forecaster, or only the published forecast, had the winner on top.",
        "- Member rows in the discrimination table are diagnostic, not a ranking: each member covers a different "
        'set of questions and eras (backlog decision-15 "Per-model peer ranking as an action"). The dissent '
        'counts cover both directions on every question (backlog decision-11 "Median drowns the correct '
        'dissenter").',
        "- Ties at the top: top-option accuracy and the multiple-choice top-option CORP score the first tied option "
        "in option order, chosen before the outcome is read. The two 'only ... had winner on top' counts treat a "
        "winner tied for the maximum as on top, the same rule in both directions.",
        "- The ghost is the gap-fill driver's single-model forecast. The ghost table bins questions by its distance "
        "from the published forecast, which does not use the outcome: binary |p - q|, multiple choice total "
        "variation distance, numeric and discrete mean absolute CDF difference.",
        "- Slice tables count each question once per dimension. For anchor_present and gfv2_loop_ran the value "
        "reads 'research source class / value', so the classes are never pooled.",
        "- Alternatives that give the outcome zero or negative probability are left out of score changes and counted.",
        "- Backlog decision-2 (the multiple-choice 0-5% bucket) and the clip sweep are not re-run here. Any change "
        "this report suggests still needs an out-of-sample era test before it ships.",
        "",
    ]


def _interval(value: list[float] | tuple[float, float] | None) -> str:
    return "(n too small)" if value is None else f"[{_signed(value[0])}, {_signed(value[1])}]"


def _rate_interval(value: list[float] | tuple[float, float] | None) -> str:
    return "(n too small)" if value is None else f"[{value[0]:.2f}, {value[1]:.2f}]"


def _signed(value: float) -> str:
    return f"{0.0 if round(value, 2) == 0 else value:+.2f}"


def _number(value: float | None) -> str:
    return "n/a" if value is None else f"{0.0 if round(value, 3) == 0 else value:.3f}"


def _counts(counts: dict[str, int] | None) -> str:
    if counts is None:
        return "not run: no --telemetry-archive"
    return ", ".join(f"{name}: {count}" for name, count in counts.items()) or "none"


def _accounting(era: dict) -> list[str]:
    lines = ["### Spot peer by forecast probability", ""]
    for name, rows in era["accounting"].items():
        heading, bin_header, lead_in = ACCOUNTING_SECTIONS[name]
        excluded = era["exclusions"][name]
        eligible = sum(row["n"] for row in rows) + sum(excluded.values())
        exclusion_note = f"excluded: {_counts(excluded)}"
        if excluded:
            exclusion_note += f", with total spot peer {_number(era['excluded_spot_peer'][name])}"
        lines += [f"#### {heading}", "", f"{lead_in} Questions: {eligible}; {exclusion_note}.", ""]
        lines += markdown_table(
            [
                bin_header,
                "n",
                "total spot peer",
                "mean spot peer",
                "95% interval",
                "share of negative",
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
        "### Miscalibration and discrimination (CORP decomposition)",
        "",
        f"Excluded binary questions: {_counts(era['exclusions']['binary: outcome diagnostics'])}; "
        f"excluded multiple-choice questions: {_counts(era['exclusions']['multiple_choice: outcome diagnostics'])}.",
        "",
    ]
    for report in era["calibration"]:
        heading, is_multiple_choice = CALIBRATION_SECTIONS[report["name"]]
        lines += [f"#### {heading}", ""]
        if is_multiple_choice:
            lines += [f"Questions where the published forecast tied for top option: {report['tied_top']}.", ""]
        lines += markdown_table(
            ["loss", "n", "mean loss", "MCB (miscalibration)", "DSC (discrimination)", "UNC (uncertainty)"],
            [
                [
                    label,
                    str(report[name]["n"]),
                    *[_number(report[name][key]) for key in ("mean_score", "mcb", "dsc", "unc")],
                ]
                for name, label in (("brier", "Brier"), ("log", "log"))
            ],
        )
        if report["name"] == "binary p_yes" and report["log"]["mcb"] is not None:
            lines += [
                "",
                f"Binary log miscalibration corresponds to {_number(100 * report['log']['mcb'])} spot peer points "
                "per question. This in-sample upper bound overstates the gain expected out of sample.",
            ]
        lines += [
            "",
            "Reliability by forecast bin; a negative 'observed minus forecast' means our probabilities were too high. "
            "The count difference interval requires at least five expected outcomes inside and outside the bin, "
            "in addition to the question and resolution-day gates.",
            "",
            *markdown_table(
                [
                    "forecast bin",
                    "n",
                    "questions",
                    "mean forecast",
                    "observed frequency",
                    "95% interval",
                    "observed minus forecast",
                    "observed count",
                    "expected count",
                    "observed minus expected",
                    "95% interval on count difference",
                ],
                [
                    [
                        row["label"],
                        str(row["n"]),
                        str(row["n_questions"]),
                        _number(row["mean_forecast"]),
                        _number(row["observed"]),
                        _rate_interval(row["jeffreys_interval"]),
                        _number(row["observed_minus_forecast"]),
                        str(row["observed_count"]),
                        _number(row["expected_count"]),
                        _number(row["observed_minus_expected"]),
                        _interval(row["cluster_interval"]),
                    ]
                    for row in report["reliability"]
                ],
            ),
            "",
        ]
    return lines


def _floor_comparison(era: dict, floor_threshold: float) -> list[str]:
    floor = era["floor"]
    return [
        "### Multiple-choice options at the probability floor",
        "",
        f"Options we gave {floor_threshold} or less. If the floor is set right, the number that resolved should be close to the "
        "number expected from their probabilities. "
        f"Excluded multiple-choice questions: {_counts(era['exclusions']['multiple_choice: outcome diagnostics'])}.",
        "",
        f"{floor['options']} options on {floor['questions']} questions; {floor['hits']} resolved, against "
        f"{floor['expected_hits']:.2f} expected. 95% interval on the resolved rate: {_rate_interval(floor['interval'])}.",
        "",
    ]


def _discrimination(era: dict) -> list[str]:
    rows = era["discrimination"]
    ghost_note = " Ghost row not run: no --telemetry-archive." if era["ghost"] is None else ""
    return [
        "### Multiple-choice discrimination by forecaster",
        "",
        "How each forecaster ranked the option that won. 'published' is what we submitted; member and ghost rows "
        f"each cover their own questions and are diagnostic, not a ranking. Excluded questions: {_counts(era['exclusions']['multiple_choice: outcome diagnostics'])}."
        f"{ghost_note}",
        "",
        *markdown_table(
            [
                "forecaster",
                "n",
                "incomplete ballots",
                "zero p on winner",
                "tied top",
                "n with spot peer",
                "n comparable to published",
                "top option won",
                "95% interval",
                "mean winner rank",
                "mean p on winner",
                "spot peer if published",
                "change vs published",
                "change 95% interval",
                "only this forecaster had winner on top",
                "only published had winner on top",
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
                    _rate_interval(row["top_interval"]),
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
    heading = "### Ghost forecast compared with the published forecast"
    if ghost is None:
        return [heading, "", "not run: no --telemetry-archive", ""]
    lines = [
        heading,
        "",
        "Spot peer change from publishing the ghost instead, binned by its distance from the published forecast; "
        "positive means the ghost would have scored higher.",
        "",
        f"Questions with no archived ghost: {ghost['unjoinable']}; joined ghosts that could not be scored: "
        f"{ghost['unscoreable']}, of which {ghost['zero_probability']} gave the outcome zero or negative probability.",
        "",
        *markdown_table(
            ["type", "questions by status", "total spot peer of questions not scored"],
            [
                [row["question_type"], _counts(row["statuses"]), _number(row["excluded_spot_peer"])]
                for row in ghost["counts_by_type"]
            ],
        ),
        "",
        *markdown_table(
            ["type", "distance from published", "n", "mean spot peer change", "95% interval"],
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
    if era["era"] == "all":
        questions = [
            (row, question)
            for row in ghost["rows"]
            if row["distance_lower"] is not None and row["distance_lower"] >= 0.05
            for question in row["questions"]
        ]
        if questions:
            lines += [
                "Questions in distance bins at or above 0.05, for trace review.",
                "",
                *markdown_table(
                    ["type", "distance bin", "question", "distance", "spot peer change"],
                    [
                        [
                            row["question_type"],
                            row["label"],
                            f"[{question['post_id']}]({question['url']})",
                            _number(question["distance"]),
                            _number(question["spot_peer_change"]),
                        ]
                        for row, question in questions
                    ],
                ),
                "",
            ]
    return lines


def _slices(era: dict) -> list[str]:
    return [
        "### Mean spot peer by slice (descriptive)",
        "",
        "Sorted from lowest mean; descriptive, with no multiple-comparison correction.",
        "",
        f"Excluded: {_counts(era['exclusions']['slices and worst decile'])}. The anchor_present and gfv2_loop_ran "
        f"rows also exclude {era['exclusions']['research slices']['no research source class']} questions with no "
        f"research source class, with total spot peer "
        f"{_number(era['excluded_spot_peer']['research slices'])}.",
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
    ]


def _worst_table(rows: list[dict], era_of_question: dict[int, str]) -> list[str]:
    return markdown_table(
        ["question", "type", "spot peer", "also in worst decile of era", "title"],
        [
            [
                f"[{row['question_id']}]({row['url']})",
                row["type"],
                _number(row["spot_peer"]),
                era_of_question.get(row["question_id"], ""),
                str(row["title"]).replace("|", "\\|"),
            ]
            for row in rows
        ],
    )


def _worst_decile(era: dict, all_era_question_ids: set[int], era_of_question: dict[int, str]) -> list[str]:
    worst = era["worst"]
    lines = ["### Worst-decile questions", ""]
    if era["era"] == "all":
        return [
            *lines,
            f"The lowest-scoring tenth of questions across all eras ({len(worst)}), for trace review. The era column "
            "is filled when the question is also in the lowest tenth of its own era. Other era sections list only "
            "the questions missing from this table.",
            "",
            *_worst_table(worst, era_of_question),
            "",
        ]
    if not worst:
        return [*lines, "No scored questions in this era.", ""]
    missing = [row for row in worst if row["question_id"] not in all_era_question_ids]
    lines.append(
        f"The lowest-scoring tenth of this era: {len(worst)} questions, spot peer at or below "
        f"{_number(worst[-1]['spot_peer'])}. {len(worst) - len(missing)} of them are in the all-era list."
    )
    if missing:
        lines += ["", *_worst_table(missing, era_of_question)]
    return [*lines, ""]


def _recalibration(result: dict) -> list[str]:
    return [
        "## Out-of-sample recalibration",
        "",
        "Each fit uses all earlier eras of the same question type. Positive spot peer change means the fitted "
        "layer would have helped; any layer still needs a decisive out-of-sample era test before shipping. "
        "Earlier-era outcomes are used retrospectively, even if they had not resolved when a later forecast was made.",
        "",
        f"Chronological eras: {', '.join(result['era_order'])}. Excluded: {_counts(result['exclusions'])}.",
        "",
        *markdown_table(
            [
                "type",
                "method",
                "test era",
                "train n",
                "test n",
                "status",
                "parameters",
                "mean spot peer change",
                "95% interval",
            ],
            [
                [
                    row["question_type"],
                    row["method"],
                    row["era"],
                    str(row["n_train"]),
                    str(row["n_test"]),
                    row["status"],
                    ", ".join(f"{key}: {_number(value)}" for key, value in (row["parameters"] or {}).items()) or "n/a",
                    _number(row["mean_delta"]),
                    _interval(row["interval"]),
                ]
                for row in result["rows"]
            ],
        ),
        "",
        "Pooled held-out questions, with skipped test eras and their question counts:",
        "",
        *markdown_table(
            ["type", "method", "test n", "mean spot peer change", "95% interval", "skipped eras", "skipped questions"],
            [
                [
                    row["question_type"],
                    row["method"],
                    str(row["n_test"]),
                    _number(row["mean_delta"]),
                    _interval(row["interval"]),
                    str(row["skipped_eras"]),
                    str(row["skipped_questions"]),
                ]
                for row in result["pooled"]
            ],
        ),
        "",
    ]


def render_report(payload: dict) -> str:
    """Render the computed results without recalculating statistics."""
    header = payload["header"]
    eras = payload["eras"]
    era_of_question = {row["question_id"]: era["era"] for era in eras if era["era"] != "all" for row in era["worst"]}
    all_era_question_ids = {row["question_id"] for era in eras if era["era"] == "all" for row in era["worst"]}
    lines = [
        "# Loss attribution",
        "",
        "Where the bot gains and loses spot peer points, by forecast probability, era and slice, with "
        "miscalibration and discrimination diagnostics.",
        "",
        "## Data",
        "",
        f"- Dataset: `{header['dataset']}`. Eras come from the field `{header['era_field']}`; bootstrap seed "
        f"{header['seed']}, {header['draws']} draws.",
        f"- Questions before exclusions: {_counts(header['before'])}.",
        f"- Questions after exclusions: {_counts(header['after'])}. Excluded by cohort: {_counts(header['cohorts'])} "
        f"({header['excluded']} questions in total); Mantic questions excluded: {header['mantic_excluded']}.",
        f"- Questions without a spot peer score, left out of score tables: {_counts(header['missing_spot_peer'])}.",
        f"- Multiple-choice questions: {header['mc_all']} in all eras, {header['mc_fall']} in triple_fall_config, "
        f"{'not run: no --telemetry-archive' if header['mc_joined_ghost'] is None else header['mc_joined_ghost']} "
        "with a joined ghost.",
        f"- Complete per-member multiple-choice vectors: {header['mc_complete_member_vectors_before']} of "
        f"{header['before'].get('multiple_choice', 0)} questions before exclusions and "
        f'{header["mc_complete_member_vectors"]} of {header["mc_all"]} after. Backlog decision-31 ("Time-of-tournament '
        'Brier averages, nr_forecasters stratification, per-model MC audit") dropped the per-member audit because '
        "stored comments lacked these vectors.",
        f"- Archived ghosts with no included question, by ghost type: {_counts(header['unjoined_archived_ghosts'])}. "
        "Their eras are unknown.",
        "",
        *_how_to_read(header),
    ]
    for era in eras:
        lines += [
            f"## Era: {era['era']}",
            "",
            *_accounting(era),
            *_calibration(era),
            *_floor_comparison(era, header["floor_threshold"]),
            *_discrimination(era),
            *_ghosts(era),
            *_slices(era),
            *_worst_decile(era, all_era_question_ids, era_of_question),
        ]
    lines += _recalibration(payload["recalibration"])
    return "\n".join(lines)
