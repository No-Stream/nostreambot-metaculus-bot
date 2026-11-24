"""Question stubs and prompt renderers shared by the ``tests/prompts/`` modules.

Every module in that package renders the same three base prompts over the same throwaway
question stubs, so the builders live here once. A plain module rather than a ``conftest.py``,
following ``tests/resolution_source_fakes.py``: none of these is a pytest fixture, nothing in
the prompt tests needs autouse setup, and an ordinary import is easier to follow than fixture
injection for a function that takes arguments.

Scope: these are the helpers used by MORE THAN ONE module. A one-off render stays inline in the
test that needs it, and the roughly twenty readable one-line ``binary_prompt(_binary_q(),
research="r")`` calls elsewhere are deliberately left as they are.
"""

import re
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from forecasting_tools.data_models.questions import DateQuestion, DiscreteQuestion

from metaculus_bot.constants import MANTIC_HOST, METACULUS_HOST
from metaculus_bot.numeric.date_axis import as_epoch_question
from metaculus_bot.numeric.utils import bound_messages, pmf_bound_messages
from metaculus_bot.prompts import (
    MARKET_SNAPSHOT_SECTION_HEADER,
    asknews_summarizer_prompt,
    binary_prompt,
    date_prompt,
    multiple_choice_prompt,
    numeric_prompt,
    pmf_prompt,
    stacking_binary_prompt,
    stacking_multiple_choice_prompt,
    stacking_numeric_prompt,
)
from tests.mantic_fakes import load_preseason_date_question

# One page URL per platform, so a test can build the same stub as a Metaculus or a Mantic question.
METACULUS_PAGE_URL = f"https://www.{METACULUS_HOST}/questions/1/"
MANTIC_PAGE_URL = f"https://{MANTIC_HOST}/questions/1/"

# Research that carries a rendered prediction-market section. The market clause in the three
# forecaster prompts is gated on this header (see ``tests/prompts/test_research_clauses.py``),
# so tests that assert on the clause must hand the prompt research that would actually have carried a snapshot.
_RESEARCH_WITH_MARKETS = f"Some news.\n\n{MARKET_SNAPSHOT_SECTION_HEADER}\n| venue | market | prob |\n| k | m | 0.4 |"


def _binary_q(
    open_time: datetime | None = None,
    resolve_time: datetime | None = None,
) -> MagicMock:
    """Minimal question stub with the attributes the prompts read."""
    q = MagicMock()
    q.question_text = "Will X occur by 2030?"
    q.background_info = "bg"
    q.resolution_criteria = "rc"
    q.fine_print = "fp"
    q.open_time = open_time if open_time is not None else datetime.now() - timedelta(days=30)
    q.scheduled_resolution_time = resolve_time if resolve_time is not None else datetime.now() + timedelta(days=365)
    # A Metaculus question with no platform-specific flags. Both are read by the prompts: the
    # platform off ``page_url``'s host, and Mantic's per-question flags (``multi_resolution``,
    # ``precision``, ``date_granularity``) off ``api_json["question"]``. Left as MagicMocks, the
    # first would raise inside ``urlparse`` and the second would render every Mantic clause,
    # since a MagicMock attribute chain is truthy.
    q.page_url = METACULUS_PAGE_URL
    q.api_json = {"question": {}}
    return q


def _mc_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.options = ["A", "B", "C"]
    return q


def _numeric_q(**kwargs) -> MagicMock:
    q = _binary_q(**kwargs)
    q.unit_of_measure = "widgets"
    q.lower_bound = 0
    q.upper_bound = 1000
    # The Metaculus default grid, typed as the model carries it: the scoring-grid clause reads
    # the bin count off ``cdf_size`` and the geometry off ``zero_point``, and a MagicMock attribute
    # is neither an int nor None.
    q.cdf_size = 201
    q.zero_point = None
    return q


def _date_q() -> DateQuestion:
    """The recorded Mantic post 651: a closed 12-bin, day-granularity date question, as the API sent it.

    A real ``DateQuestion`` rather than a MagicMock because ``date_prompt`` builds the epoch view
    through ``numeric.date_axis.as_epoch_question``, which reads the API's ``scaling`` block.
    """
    return load_preseason_date_question()


def _open_upper_date_q() -> DateQuestion:
    """The modal live Mantic date shape: a legacy fine-grid "when will X happen" question, open above.

    Granularity empty (the platform sends ``""`` on 50 of 51 live date questions), 201-point CDF,
    closed lower bound at the question open and an OPEN upper bound, which is where 101 of 188
    Series 1 date questions resolved.
    """
    lower = datetime(2026, 9, 1, tzinfo=UTC)
    upper = datetime(2027, 3, 1, tzinfo=UTC)
    return DateQuestion(
        id_of_question=777,
        id_of_post=777,
        page_url=MANTIC_PAGE_URL,
        question_text="When will X happen?",
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        published_time=None,
        close_time=None,
        open_time=datetime.now(UTC) - timedelta(days=30),
        scheduled_resolution_time=datetime.now(UTC) + timedelta(days=365),
        lower_bound=lower,
        upper_bound=upper,
        open_lower_bound=False,
        open_upper_bound=True,
        zero_point=None,
        cdf_size=201,
        api_json={
            "question": {
                "date_granularity": "",
                "inbound_outcome_count": 200,
                "scaling": {"nominal_min": lower.timestamp(), "nominal_max": upper.timestamp()},
            }
        },
    )


def _pmf_q(
    *,
    cdf_size: int = 22,
    lower_bound: float = -0.5,
    upper_bound: float = 20.5,
    nominal_lower: float = 0.0,
    nominal_upper: float = 20.0,
    open_lower: bool = False,
    open_upper: bool = True,
    unit_of_measure: str = "releases",
    page_url: str = MANTIC_PAGE_URL,
    zero_point: float | None = None,
) -> DiscreteQuestion:
    """A coarse Mantic discrete question in the shape of Series 1 post 643: counts 0 to 20, open ceiling.

    Twenty-one centre-aligned bins (``range_min = nominal_min - step / 2``), the platform's discrete
    convention, so the per-bin prompt labels them ``0`` through ``20`` and adds ``above_range``. A
    real ``DiscreteQuestion`` rather than a MagicMock because ``pmf_prompt`` builds the labelled grid
    from the typed bounds and reads the nominal bounds through ``getattr``, which a MagicMock would
    answer with another MagicMock. The defaults are overridable so a test can render the other
    corpus shapes (a closed 3-bin count, a 30-bin numeric grid whose nominal bounds are its edges, a
    log-spaced grid with a ``zero_point``).
    """
    return DiscreteQuestion(
        id_of_question=643,
        id_of_post=643,
        page_url=page_url,
        question_text="How many public releases will U.S. Central Command publish?",
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        published_time=None,
        close_time=None,
        open_time=datetime.now(UTC) - timedelta(days=30),
        scheduled_resolution_time=datetime.now(UTC) + timedelta(days=365),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        open_lower_bound=open_lower,
        open_upper_bound=open_upper,
        unit_of_measure=unit_of_measure,
        zero_point=zero_point,
        cdf_size=cdf_size,
        nominal_lower_bound=nominal_lower,
        nominal_upper_bound=nominal_upper,
        api_json={"question": {}},
    )


def _pmf_prompt_text(question: DiscreteQuestion | DateQuestion | None = None, research: str = "r") -> str:
    """``pmf_prompt`` on ``question`` (default the post-643 count grid) with its real, per-bin-worded bound messages."""
    q = question if question is not None else _pmf_q()
    view = as_epoch_question(q) if isinstance(q, DateQuestion) else q
    upper_message, lower_message = pmf_bound_messages(view)
    return pmf_prompt(q, research=research, lower_bound_message=lower_message, upper_bound_message=upper_message)


def _date_prompt_text(question: DateQuestion | None = None, research: str = "r") -> str:
    """``date_prompt`` on ``question`` (default post 651) with its real, date-rendered bound messages."""
    q = question if question is not None else _date_q()
    upper_message, lower_message = bound_messages(as_epoch_question(q), stated_tail_fields=True)
    return date_prompt(q, research=research, lower_bound_message=lower_message, upper_bound_message=upper_message)


def _flat(text: str) -> str:
    """Lowercase and collapse all whitespace.

    The prompt constants are pre-indented for ``clean_indents``, so an assertion on their
    wording must not depend on where the lines happen to wrap.
    """
    return " ".join(text.lower().split())


def _binary_prompt_text() -> str:
    return binary_prompt(_binary_q(), research="r")


def _mc_prompt_text() -> str:
    return multiple_choice_prompt(_mc_q(), research="r")


def _numeric_prompt_text() -> str:
    return numeric_prompt(_numeric_q(), research="r", lower_bound_message="lbm", upper_bound_message="ubm")


def _stacked_prompt_texts() -> list[str]:
    """All three stacking prompts, for the scope guard every base-prompt rule carries."""
    return [
        stacking_binary_prompt(_binary_q(), research="r", base_predictions=["a1", "a2"]),
        stacking_multiple_choice_prompt(_mc_q(), research="r", base_predictions=["a1", "a2"]),
        stacking_numeric_prompt(
            _numeric_q(),
            research="r",
            base_predictions=["a1", "a2"],
            lower_bound_message="lbm",
            upper_bound_message="ubm",
        ),
    ]


def _extract_last_json_block(prompt: str) -> str:
    """Return the body of the LAST fenced ```json block in a prompt.

    Every forecasting prompt ends on its STRUCTURED FORECAST example, and the ladder
    (``value_extraction``) reads that block as the authoritative forecast, so the example
    is what teaches the model the schema. Shared because two modules read it: the schema-block
    suite parses the example as real JSON, and the base-rule suite asserts that neither Phase 1
    rule added a field to it.
    """
    blocks = re.findall(r"```json\s*\n(.*?)\n\s*```", prompt, re.DOTALL)
    assert blocks, "no fenced json block found in prompt"
    return blocks[-1]


def _summarizer_prompt(**overrides) -> str:
    """Build the AskNews summarizer prompt with representative defaults."""
    kwargs = {
        "question_text": "Will X happen by 2027?",
        "resolution_criteria": "Resolves YES if X happens",
        "fine_print": "fp",
        "open_date": "2026-03-15",
        "research": "raw asknews articles",
    }
    kwargs.update(overrides)
    return asknews_summarizer_prompt(**kwargs)
