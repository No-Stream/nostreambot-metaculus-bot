from __future__ import annotations

import pytest
from pydantic import ValidationError

from metaculus_bot.research.agentic.loop import run_agentic_loop
from metaculus_bot.research.agentic.types import EvidencePassage, ToolOutcome
from tests.agentic_fakes import FakeLlm, plan_call, response, tool_call, tool_messages, tool_spec
from tests.agentic_fakes import loop_config as config


@pytest.mark.asyncio
async def test_delivered_reference_binds_excerpt_to_adapter_source_and_cannot_be_restamped() -> None:
    source_a = "https://agency.example/a.pdf"
    source_b = "https://agency.example/b"

    async def search_web(query: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown=f"Results for {query}: {source_a} and {source_b}",
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_a,
                    text="The rate was 5.2% in 2025.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
            ],
        )

    async def fetch(url: str) -> ToolOutcome:
        return ToolOutcome(content_markdown=f"Cookies only. {url}", method="plain")

    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("search", "search_web", {"query": "rate"})]),
            response(
                tool_calls=[
                    tool_call(
                        "wrong",
                        "record_findings",
                        {
                            "findings": [
                                {"claim": "The rate was 5.2%.", "source_url": source_b, "evidence_refs": ["ev_1"]}
                            ]
                        },
                    )
                ]
            ),
            response(
                tool_calls=[
                    tool_call(
                        "right",
                        "record_findings",
                        {"findings": [{"claim": "The rate was 5.2%.", "evidence_refs": ["ev_1"]}]},
                    )
                ]
            ),
            response(tool_calls=[tool_call("fetch", "fetch", {"url": source_a})]),
            response(
                tool_calls=[
                    tool_call(
                        "repeat",
                        "conclude",
                        {"final_findings": [{"claim": "The rate was 5.2%.", "evidence_refs": ["ev_1"]}]},
                    )
                ]
            ),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("search_web", search_web), tool_spec("fetch", fetch)],
        config(max_steps=8, max_conclude_gate_rejections=0),
        llm_call=driver,
    )

    assert len(result.delivered_evidence) == 1
    assert result.delivered_evidence[0].text == "The rate was 5.2% in 2025."
    assert "provider summary" in result.findings_markdown.lower()
    assert result.findings_markdown.count("Claim: The rate was 5.2%.") == 1
    assert "Cookies only" not in result.findings_markdown
    assert "source_url disagrees" in tool_messages(result)[2]["content"]


@pytest.mark.asyncio
async def test_unknown_and_oversize_references_are_rejected() -> None:
    async def fetch(url: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="source page",
            method="plain",
            evidence_passages=[
                EvidencePassage(source_url=url, text="A" * 2500, retrieval_method="plain", content_origin="source_text")
            ],
        )

    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("fetch", "fetch", {"url": "https://agency.example/report"})]),
            response(
                tool_calls=[
                    tool_call(
                        "record",
                        "record_findings",
                        {
                            "findings": [
                                {"claim": "A", "evidence_refs": ["missing"]},
                                {"claim": "B", "evidence_refs": ["ev_1"]},
                            ]
                        },
                    )
                ]
            ),
            response(tool_calls=[tool_call("done", "conclude")]),
        ]
    )
    result = await run_agentic_loop(
        "system", "briefing", [tool_spec("fetch", fetch)], config(max_conclude_gate_rejections=0), llm_call=driver
    )
    assert result.telemetry.findings_count == 0
    assert "unknown or undelivered" in tool_messages(result)[2]["content"]


@pytest.mark.asyncio
async def test_multisource_support_and_explicit_replacement_preserve_origin_and_identity() -> None:
    source_a = "https://agency.example/report"
    source_b = "https://publisher.example/article"
    source_c = "https://archive.example/reader"

    async def search_web(query: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown=query,
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=source_a,
                    text="A: reported count 120.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                ),
                EvidencePassage(
                    source_url=source_b,
                    text="B: reported count 100.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                ),
            ],
        )

    async def fetch(url: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="fetched report",
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url=url,
                    text="A: confirmed count 121.",
                    retrieval_method="plain",
                    content_origin="source_text",
                    clipped=True,
                )
            ],
        )

    async def read_document(url: str, ask: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown=f"Answer to {ask}",
            method="document",
            evidence_passages=[
                EvidencePassage(
                    source_url=url,
                    text="Reader answer: year 2025.",
                    retrieval_method="document",
                    content_origin="generated_answer",
                )
            ],
        )

    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("search", "search_web", {"query": "counts"})]),
            response(
                tool_calls=[
                    tool_call(
                        "first",
                        "record_findings",
                        {"findings": [{"claim": "A reported 120.", "evidence_refs": ["ev_1"]}]},
                    )
                ]
            ),
            response(tool_calls=[tool_call("fetch", "fetch", {"url": source_a})]),
            response(tool_calls=[tool_call("reader", "read_document", {"url": source_c, "ask": "Which year?"})]),
            response(
                tool_calls=[
                    tool_call(
                        "revision",
                        "record_findings",
                        {
                            "findings": [
                                {
                                    "claim": "A confirmed 121.",
                                    "evidence_refs": ["ev_3"],
                                    "replaces_finding_id": "finding_1",
                                },
                                {
                                    "claim": "A and B reported different counts in 2025.",
                                    "evidence_refs": ["ev_3", "ev_2", "ev_4"],
                                    "discrepancy": True,
                                },
                            ]
                        },
                    )
                ]
            ),
            response(
                tool_calls=[
                    tool_call(
                        "done",
                        "conclude",
                        {
                            "final_findings": [
                                {"claim": "A confirmed 121.", "evidence_refs": ["ev_3"]},
                            ]
                        },
                    )
                ]
            ),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("search_web", search_web), tool_spec("fetch", fetch), tool_spec("read_document", read_document)],
        config(max_steps=9, max_conclude_gate_rejections=0),
        llm_call=driver,
    )
    assert result.telemetry.findings_count == 2
    assert result.findings_markdown.count("Claim: A confirmed 121.") == 1
    assert "A reported 120." not in result.findings_markdown
    assert "A: confirmed count 121." in result.findings_markdown
    assert "B: reported count 100." in result.findings_markdown
    assert "Reader answer: year 2025." in result.findings_markdown
    assert source_a in result.findings_markdown
    assert source_b in result.findings_markdown
    assert "provider summary" in result.findings_markdown
    assert "source text" in result.findings_markdown
    assert "clipped visible excerpt" in result.findings_markdown
    assert result.delivered_evidence[2].clipped is True
    assert "generated answer" in result.findings_markdown
    assert "limited evidence" in result.findings_markdown.lower()
    assert "supersede the corresponding briefing" not in result.findings_markdown
    assert "A reported 120." in str(result.transcript)


@pytest.mark.asyncio
async def test_repeated_external_delivery_keeps_each_result_bounded() -> None:
    async def fetch(url: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="Context " * 1000,
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url=url, text="Published count 121.", retrieval_method="plain", content_origin="source_text"
                )
            ],
        )

    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("first", "fetch", {"url": "https://agency.example/report"})]),
            response(tool_calls=[tool_call("repeat", "fetch", {"url": "https://agency.example/report"})]),
            response(tool_calls=[tool_call("done", "conclude")]),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("fetch", fetch)],
        config(max_result_chars=900, max_conclude_gate_rejections=0),
        llm_call=driver,
    )
    messages = tool_messages(result)
    assert all(len(message["content"]) <= 900 for message in messages)
    assert len(result.delivered_evidence) == 2
    assert all(
        item.ref in messages[index + 1]["content"] and item.text in messages[index + 1]["content"]
        for index, item in enumerate(result.delivered_evidence)
    )

    with pytest.raises(ValidationError):
        result.delivered_evidence[0].text = "changed"


@pytest.mark.asyncio
async def test_combined_cited_support_over_limit_is_rejected_without_shortening() -> None:
    async def fetch(url: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="Two passages",
            method="plain",
            evidence_passages=[
                EvidencePassage(
                    source_url=url,
                    text=f"Passage {index}: " + "A" * 1510,
                    retrieval_method="plain",
                    content_origin="source_text",
                )
                for index in (1, 2)
            ],
        )

    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("fetch", "fetch", {"url": "https://agency.example/report"})]),
            response(
                tool_calls=[
                    tool_call(
                        "record",
                        "record_findings",
                        {
                            "findings": [
                                {"claim": "Two long passages.", "evidence_refs": ["ev_1", "ev_2"]},
                            ]
                        },
                    )
                ]
            ),
            response(tool_calls=[tool_call("done", "conclude")]),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("fetch", fetch)],
        config(max_result_chars=5000, max_conclude_gate_rejections=0),
        llm_call=driver,
    )
    assert result.telemetry.findings_count == 0
    assert len(result.delivered_evidence) == 2
    assert "choose a smaller adequate set" in tool_messages(result)[2]["content"]


@pytest.mark.asyncio
async def test_replacement_cannot_take_another_current_findings_content_identity() -> None:
    async def search_web(query: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown=query,
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=f"https://agency.example/{name}",
                    text=f"{name} count.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
                for name in ("alpha", "beta")
            ],
        )

    original = [
        {"claim": "Alpha count.", "evidence_refs": ["ev_1"]},
        {"claim": "Beta count.", "evidence_refs": ["ev_2"]},
    ]
    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("search", "search_web", {"query": "counts"})]),
            response(tool_calls=[tool_call("record", "record_findings", {"findings": original})]),
            response(
                tool_calls=[
                    tool_call(
                        "collision",
                        "record_findings",
                        {
                            "findings": [
                                {**original[1], "replaces_finding_id": "finding_1"},
                            ]
                        },
                    )
                ]
            ),
            response(tool_calls=[tool_call("done", "conclude", {"final_findings": [original[1]]})]),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("search_web", search_web)],
        config(max_steps=7, max_conclude_gate_rejections=0),
        llm_call=driver,
    )
    assert result.telemetry.findings_count == 2
    assert result.findings_markdown.count("Claim: Alpha count.") == 1
    assert result.findings_markdown.count("Claim: Beta count.") == 1
    assert "replacement content already belongs" in tool_messages(result)[3]["content"]
    assert "replaces_finding_id" in str(result.transcript)


@pytest.mark.asyncio
async def test_two_replacements_in_one_batch_cannot_claim_the_same_content_identity() -> None:
    async def search_web(query: str) -> ToolOutcome:
        return ToolOutcome(
            content_markdown=query,
            method="search",
            evidence_passages=[
                EvidencePassage(
                    source_url=f"https://agency.example/{name}",
                    text=f"{name} count.",
                    retrieval_method="search",
                    content_origin="provider_summary",
                )
                for name in ("alpha", "beta", "gamma")
            ],
        )

    original = [
        {"claim": "Alpha count.", "evidence_refs": ["ev_1"]},
        {"claim": "Gamma count.", "evidence_refs": ["ev_3"]},
    ]
    replacement = {"claim": "Beta count.", "evidence_refs": ["ev_2"]}
    driver = FakeLlm(
        [
            response(tool_calls=[plan_call()]),
            response(tool_calls=[tool_call("search", "search_web", {"query": "counts"})]),
            response(tool_calls=[tool_call("record", "record_findings", {"findings": original})]),
            response(
                tool_calls=[
                    tool_call(
                        "collision",
                        "record_findings",
                        {
                            "findings": [
                                {**replacement, "replaces_finding_id": "finding_1"},
                                {**replacement, "replaces_finding_id": "finding_2"},
                            ]
                        },
                    )
                ]
            ),
            response(tool_calls=[tool_call("done", "conclude")]),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "briefing",
        [tool_spec("search_web", search_web)],
        config(max_steps=7, max_conclude_gate_rejections=0),
        llm_call=driver,
    )
    assert result.telemetry.findings_count == 2
    assert result.findings_markdown.count("Claim: Beta count.") == 1
    assert result.findings_markdown.count("Claim: Gamma count.") == 1
    assert "replacement content already belongs" in tool_messages(result)[3]["content"]
