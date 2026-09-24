"""Log-noise levels ``cli._configure_process`` sets on third-party loggers."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from metaculus_bot.cli import _configure_process
from tests.cli_test_helpers import _configure_process_stubs

_COST_MANAGER_LOGGER = "forecasting_tools.ai_models.resource_managers.monetary_cost_manager"


@pytest.fixture
def restore_cost_manager_level() -> Iterator[None]:
    cost_logger = logging.getLogger(_COST_MANAGER_LOGGER)
    original_level = cost_logger.level
    yield
    cost_logger.setLevel(original_level)


class TestCostManagerNoise:
    def test_cost_mismatch_warning_is_silenced_but_errors_still_surface(
        self, restore_cost_manager_level: None, caplog: pytest.LogCaptureFixture
    ) -> None:
        with _configure_process_stubs():
            _configure_process("test_questions")

        cost_logger = logging.getLogger(_COST_MANAGER_LOGGER)
        with caplog.at_level(logging.DEBUG):
            cost_logger.warning(
                "Litellm hidden param cost 0.0 and response object cost 0.24 are different by more than 5 cents."
            )
            cost_logger.error("a real cost-manager failure")

        messages = [record.getMessage() for record in caplog.records if record.name == _COST_MANAGER_LOGGER]
        assert messages == ["a real cost-manager failure"]
