"""The autouse ``_stop_litellm_logging_worker`` fixture leaves no litellm worker task behind.

``forecasting_tools`` applies ``nest_asyncio``, whose ``asyncio.run`` drives the coroutine on the
current loop WITHOUT cancelling leftover tasks or closing the loop. A sync test that runs the CLI
therefore strands litellm's global ``_worker_loop`` task on a loop nobody finishes; the loop is
closed later, and the garbage-collected coroutine raises ``RuntimeError: Event loop is closed``
into whichever test happens to be running (a CI ``##[error]`` annotation).
"""

import asyncio

from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER


async def _start_logging_worker() -> None:
    GLOBAL_LOGGING_WORKER.start()
    await asyncio.sleep(0)


class TestLoggingWorkerTeardown:
    """Ordered pair: the first test strands the worker, the second sees what teardown left."""

    def test_strand_worker_on_a_loop_that_is_never_finished(self) -> None:
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_start_logging_worker())
        worker_task = GLOBAL_LOGGING_WORKER._worker_task
        assert worker_task is not None
        assert not worker_task.done()

    def test_previous_test_worker_was_stopped(self) -> None:
        assert GLOBAL_LOGGING_WORKER._worker_task is None
