from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from metaculus_bot.research.agentic import llm as agentic_llm
from metaculus_bot.research.agentic.artifact import detachment_lint
from metaculus_bot.research.agentic.image_messages import bind_image_materialization, image_reference_message
from metaculus_bot.research.agentic.loop import run_agentic_loop, run_ghost_v1
from metaculus_bot.research.agentic.provenance import _method_to_tier
from metaculus_bot.research.agentic.types import Finding, ToolOutcome
from metaculus_bot.research.image_assets import ImageView
from tests.agentic_fakes import FakeLlm
from tests.agentic_fakes import gap_accounting as _accounting
from tests.agentic_fakes import loop_config as _config
from tests.agentic_fakes import plan_call as _plan_call
from tests.agentic_fakes import response as _response
from tests.agentic_fakes import tool_call as _tool_call
from tests.agentic_fakes import tool_spec as _tool_spec


def _image(
    source_url: str,
    *,
    final_url: str | None = None,
    image_id: str = "pixel-sha",
    parent_page_urls: tuple[str, ...] = (),
    source_sha256: str = "source-sha",
    original_width: int = 640,
    original_height: int = 480,
    crop: tuple[int, int, int, int] | None = None,
) -> ImageView:
    return ImageView(
        image_id=image_id,
        source_url=source_url,
        final_url=final_url or source_url,
        source_sha256=source_sha256,
        original_width=original_width,
        original_height=original_height,
        width=640,
        height=480,
        crop=crop,
        png_bytes=b"\x89PNG\r\n\x1a\nactual-pixels",
        parent_page_urls=parent_page_urls,
    )


def _image_blocks(call: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        block
        for message in call["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "image_url"
    ]


@pytest.mark.asyncio
async def test_image_follows_all_tool_messages_and_reaches_same_driver_next_turn() -> None:
    image = _image("https://charts.example/a.png")

    async def view_image(**_: Any) -> ToolOutcome:
        return ToolOutcome(
            content_markdown="image_id=pixel-sha source=https://charts.example/a.png",
            method="image_local",
            image_views=[image],
        )

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("view1", "view_image", {"url": image.source_url}),
                    _tool_call("search1", "search_web", {"query": "chart context"}),
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude", {"gap_accounting": _accounting("g1")})]),
        ]
    )

    async def search_web(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="context", method="search")

    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image), _tool_spec("search_web", search_web)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    roles_after_batch = [message["role"] for message in result.transcript[5:8]]
    assert roles_after_batch == ["tool", "tool", "user"]
    reference_message = result.transcript[7]
    assert reference_message["content"][-1] == {"type": "image_reference", "image_id": image.image_id}
    assert _image_blocks(fake_llm.calls[2]) == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgphY3R1YWwtcGl4ZWxz"}}
    ]
    assert fake_llm.calls[2]["tools"] == fake_llm.calls[1]["tools"]
    assert b"actual-pixels" not in repr(result.transcript).encode()
    assert "data:image" not in repr(result.transcript)


@pytest.mark.asyncio
async def test_duplicate_pixels_attach_once_and_retain_all_source_aliases() -> None:
    first = _image(
        "https://charts.example/a.png",
        final_url="https://cdn.example/chart.png",
        parent_page_urls=("https://agency.example/report",),
    )
    alias = _image(
        "https://mirror.example/a.png",
        image_id=first.image_id,
        parent_page_urls=("https://mirror.example/report",),
        source_sha256="different-source-sha",
        original_width=1280,
        original_height=960,
        crop=(0, 0, 640, 480),
    )

    async def view_image(url: str) -> ToolOutcome:
        image = first if "charts.example" in url else alias
        return ToolOutcome(
            content_markdown=f"image_id={image.image_id} source={url}", method="image_local", image_views=[image]
        )

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("view1", "view_image", {"url": first.source_url}),
                    _tool_call("view2", "view_image", {"url": alias.source_url}),
                    _tool_call("view3", "view_image", {"url": first.source_url}),
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert [view.image_id for view in result.image_views] == [first.image_id]
    assert result.image_sources == {
        first.image_id: sorted({first.source_url, first.final_url, alias.source_url, alias.final_url})
    }
    assert result.image_views[0].parent_page_urls == (
        "https://agency.example/report",
        "https://mirror.example/report",
    )
    assert [observation["source_sha256"] for observation in result.image_observations[first.image_id]] == [
        "source-sha",
        "different-source-sha",
    ]
    assert [observation["crop"] for observation in result.image_observations[first.image_id]] == [
        None,
        (0, 0, 640, 480),
    ]
    assert len(_image_blocks(fake_llm.calls[2])) == 1


@pytest.mark.asyncio
async def test_visual_finding_requires_image_from_preceding_turn_and_registered_source() -> None:
    image = _image("https://charts.example/a.png", final_url="https://cdn.example/a.png")
    visual_finding = {
        "claim": "The plotted series ends at 42.",
        "source_url": image.final_url,
        "evidence_kind": "image",
        "image_id": image.image_id,
        "visual_observation": "Transcribed: the final labeled point is 42.",
        "retrieved_how": "view_image",
        "topic": "chart",
    }

    async def view_image(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="image metadata", method="image_local", image_views=[image])

    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("view", "view_image", {"url": image.source_url})]),
            _response(tool_calls=[_tool_call("record", "record_findings", {"findings": [visual_finding]})]),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert result.telemetry.findings_count == 1
    assert result.telemetry.provenance_rejections == 0
    assert "Visual observation" in result.findings_markdown
    assert visual_finding["visual_observation"] in result.findings_markdown
    assert "Quote:" not in result.findings_markdown


@pytest.mark.asyncio
async def test_coissued_view_and_visual_finding_is_rejected_as_premature() -> None:
    image = _image("https://charts.example/a.png")

    async def view_image(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="image metadata", method="image_local", image_views=[image])

    finding = {
        "claim": "The plotted series ends at 42.",
        "source_url": image.source_url,
        "evidence_kind": "image",
        "image_id": image.image_id,
        "visual_observation": "Transcribed: the final labeled point is 42.",
    }
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(
                tool_calls=[
                    _tool_call("view", "view_image", {"url": image.source_url}),
                    _tool_call("record", "record_findings", {"findings": [finding]}),
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert result.telemetry.findings_count == 0
    assert result.telemetry.provenance_rejections == 1
    assert "was not delivered" in next(m["content"] for m in result.transcript if m.get("tool_call_id") == "record")


@pytest.mark.asyncio
async def test_visual_finding_rejects_unknown_image_id_and_unregistered_source_alias() -> None:
    image = _image("https://charts.example/a.png")

    async def view_image(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="image metadata", method="image_local", image_views=[image])

    base_finding = {
        "claim": "The plotted series ends at 42.",
        "source_url": image.source_url,
        "evidence_kind": "image",
        "image_id": image.image_id,
        "visual_observation": "Transcribed: the final labeled point is 42.",
    }
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("view", "view_image", {"url": image.source_url})]),
            _response(
                tool_calls=[
                    _tool_call(
                        "record",
                        "record_findings",
                        {
                            "findings": [
                                base_finding | {"image_id": "invented-id"},
                                base_finding | {"source_url": "https://invented.example/chart.png"},
                            ]
                        },
                    )
                ]
            ),
            _response(tool_calls=[_tool_call("done", "conclude")]),
        ]
    )

    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
    )

    assert result.telemetry.findings_count == 0
    assert result.telemetry.provenance_rejections == 2
    record_message = next(m["content"] for m in result.transcript if m.get("tool_call_id") == "record")
    assert "invented-id" in record_message
    assert "not a registered source or final URL" in record_message


def test_text_finding_contract_remains_quote_grounded() -> None:
    finding = Finding(claim="The report says 42.", source_url="https://example.com", quote="The value was 42.")

    assert finding.evidence_kind == "text"
    assert finding.image_id is None
    assert finding.visual_observation is None

    with pytest.raises(ValueError, match="text evidence requires quote"):
        Finding(claim="The report says 42.", source_url="https://example.com")


def test_only_real_local_content_methods_grant_fetched_tier() -> None:
    assert _method_to_tier("image_local") == "fetched"
    assert _method_to_tier("local") == "fetched"
    assert _method_to_tier("local_navigation") is None


def test_visual_observation_keeps_detachment_lint() -> None:
    finding = Finding(
        claim="The line ends above the threshold.",
        source_url="https://example.com/chart.png",
        evidence_kind="image",
        image_id="pixels",
        visual_observation="The line likely remains above 42.",
    )

    assert detachment_lint(finding) == ["visual_observation contains banned register 'likely'"]


@pytest.mark.asyncio
async def test_default_transport_receives_materialized_pixels_without_mutating_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = _image("https://charts.example/a.png")
    acompletion = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(agentic_llm, "acompletion", acompletion)
    monkeypatch.setattr(agentic_llm, "should_route_via_donated_key", lambda model: False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    references = [image_reference_message([image.image_id])]
    call = bind_image_materialization(
        agentic_llm.build_default_llm_call(_config()),
        {image.image_id: image},
        {image.image_id: {image.source_url}},
    )

    await call(references, None)

    assert acompletion.await_args is not None
    acompletion.assert_awaited_once()
    assert acompletion.await_args.kwargs["model"] == "openrouter/openai/gpt-5.6-luna"
    provider_messages = acompletion.await_args.kwargs["messages"]
    assert provider_messages[0]["content"][-1]["type"] == "image_url"
    assert provider_messages[0]["content"][-1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert references[0]["content"][-1] == {"type": "image_reference", "image_id": image.image_id}


@pytest.mark.asyncio
async def test_ghost_context_keeps_refs_while_plain_and_v1_ghost_requests_receive_pixels() -> None:
    image = _image("https://charts.example/a.png")

    async def view_image(**_: Any) -> ToolOutcome:
        return ToolOutcome(content_markdown="image metadata", method="image_local", image_views=[image])

    ghost_text = '```json\n{"question_type":"binary","posterior_prob":0.5}\n```'
    fake_llm = FakeLlm(
        [
            _response(tool_calls=[_plan_call()]),
            _response(tool_calls=[_tool_call("view", "view_image", {"url": image.source_url})]),
            _response(tool_calls=[_tool_call("done", "conclude")]),
            _response(content=ghost_text),
            _response(content=ghost_text),
        ]
    )
    result = await run_agentic_loop(
        "system",
        "brief",
        [_tool_spec("view_image", view_image)],
        _config(max_conclude_gate_rejections=0),
        llm_call=fake_llm,
        ghost_prompt="plain ghost",
    )
    assert result.ghost_context is not None

    await run_ghost_v1(result.ghost_context, "v1 ghost")

    assert len(_image_blocks(fake_llm.calls[3])) == 1
    assert len(_image_blocks(fake_llm.calls[4])) == 1
    assert "image_reference" in repr(result.ghost_context.messages)
    assert "data:image" not in repr(result.ghost_context.messages)
    assert "data:image" not in repr(result.transcript)
