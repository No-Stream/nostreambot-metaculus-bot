"""Flag source attributions Gemini's verified search-link record cannot support.

Gemini writes self-invented source-tier tags into its search-link output —
``[A: NASA]``, ``[B: Reuters]``, ``[C: Time and Date]`` — while the only provenance we
hold is the ``### Sources`` list our formatter renders from resolved, cited-link domains.
Across the 323 archived Gemini sections, 681 outlet-named tier
attributions reach this check and **70% of them name an outlet absent from that same
response's own verified-domain list** (q44953 claims ``[A: NASA]`` for the eclipse path
over a source list of perlan.is / guidetoiceland.is / timeanddate.com; q45401 names 19
institutions over one verified domain). The cited-link floor cannot
see this — it fires only when no cited link verifies — and the forecaster prompts
instruct weighting by source tier, so an unbacked tier tag is an authority claim we
manufactured. Receipts: ``scratch/residual_2026-08-31/gemini_search_audit/cutB_pattern.md``
§3.2 and ``VERDICT.md`` §2 (the embellishment channel).

What this module does NOT claim: that the FACT is wrong. An outlet missing from the
verified domains can still be the true origin — a search redirect may name an
aggregator while the text names the original wire. So the rewrite replaces only the
attribution decoration, never a word of the sentence, and it says exactly what we know:
``unverified attribution``. Matching is deliberately loose in the KEEP direction (six
rules, any one of which credits the name), because a false strip discards real
provenance while a false keep merely leaves a tag standing.

Scope is the tier-tag surface only: a bracket group with at least one ``<Letter>: <name>``
item. Bare-name brackets (``[Forbes]``, ``[Consumer Reports, NPR]``) and prose
attributions ("according to Reuters") are deliberately out of scope — the corpus shows
the bare-bracket surface is ~5% the size of the tier surface and shares its syntax with
markdown link text and editorial insertions (``[the states]``), where a rewrite would
corrupt content. Validation, counts and the false-strip review:
``scratch/next_season_bundle_2026-09/item4_attribution_check/VALIDATION.md``.

A tier tag that names no outlet at all (``[A: official]``, ``[A: peer-reviewed journal]``)
is rewritten to the same marker. The research prompts ask for the outlet name, and a
class description lets the model claim tier A with nothing to check it against (Q14333
smoke, 2026-09-24: 10 of 12 tags were descriptions). Replay over the archive:
``scratch/attribution_named_tags_2026-09-24/``.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from metaculus_bot.research.bracket_groups import (
    BRACKET_GROUP_RE,
    iter_group_items,
    join_group_items,
    rebuild_group,
)
from metaculus_bot.research.public_suffix import registrable_domain

__all__ = ["AttributionCheck", "rewrite_unsupported_attributions"]

# What we render in place of an attribution the grounding record cannot back. Wording is
# load-bearing: "unverified" is what we can defend, "false" is not.
UNVERIFIED_ATTRIBUTION_MARKER = "unverified attribution"

# Words that describe a CLASS of source rather than name an outlet. A tag whose every
# identity token is one of these (``official``, ``peer-reviewed journal``, ``aggregator /
# crowd platform``) names nothing the grounding record could back, so it is rewritten to
# the marker like an unmatched name. Seeded from the archived corpus's descriptive tags
# (``official`` 443, ``aggregator`` 217, ``primary`` 46 of its 790 tier items), the
# 2026-09-22 probe controls and the 2026-09-24 Q14333 smoke (10 of 12 tags descriptive).
# Census: ``scratch/attribution_named_tags_2026-09-24/tag_census.txt``. ``site`` stays out
# because "The News Site" is a real outlet in that corpus. The token-intersection support
# rule ignores these too, so a shared "research" cannot credit a name on its own.
_DESCRIPTOR_TOKENS = frozenset(
    {
        "a",
        "academic",
        "aggregated",
        "aggregator",
        "aggregators",
        "analysis",
        "analyst",
        "analysts",
        "attributed",
        "authority",
        "crowd",
        "data",
        "database",
        "expert",
        "experts",
        "financial",
        "forecast",
        "forecasting",
        "gov",
        "government",
        "institute",
        "journal",
        "local",
        "market",
        "markets",
        "media",
        # ``n/a`` splits into ``n`` and ``a``.
        "n",
        "named",
        "news",
        "newspaper",
        "official",
        "original",
        "outlet",
        "outlets",
        "peer",
        "platform",
        "platforms",
        "policy",
        "prediction",
        "preprint",
        "primary",
        "projections",
        "record",
        "records",
        "registry",
        "report",
        "reports",
        "research",
        "reviewed",
        "secondary",
        "service",
        "services",
        "single",
        "social",
        "source",
        "sources",
        "state",
        "statistics",
        "study",
        "studies",
        "tertiary",
        "translated",
        "unknown",
        "updates",
        "validation",
        "validations",
        "wire",
    }
)

# Tokens carrying no outlet identity: English function words plus the hostname parts every
# domain shares. Kept close to the audit's stop list so the measurement stays comparable.
_STOP_TOKENS = frozenset(
    {
        "the",
        "of",
        "and",
        "an",
        "for",
        "in",
        "on",
        "at",
        "to",
        "com",
        "org",
        "net",
        "co",
        "uk",
        "gov",
        "edu",
        # A TLD, not an identity: crediting the ``.ai`` in orcarouter.ai to every name
        # carrying "AI" made three unrelated outlets read as grounded.
        "ai",
        "www",
        "inc",
        "ltd",
        "llc",
        "news",
    }
)

# A one- or two-letter domain core (``ap``, ``ft``, ``mk``) is a substring of far too many
# outlet names to credit one in the reverse direction.
_MIN_DOMAIN_CORE_CHARS = 3

# The bracket-group and item-split grammar comes from ``research/bracket_groups.py``, which
# the citation-index strip reads the same text through immediately before this pass (see
# ``gemini_search._strip_model_citation_indices``). Shared rather than restated so the two
# cannot come to disagree about what one string's groups and items are; that module is a
# leaf below both, so importing it keeps this module's own leaf position (the provider
# imports it, not the other way round).
# ``A: NASA`` -> tier grade + outlet. An item with no grade is a continuation name under
# the previous one (``[D: GrackerAI, siberX]``).
_TIER_ITEM_RE = re.compile(r"^([A-Z]):\s*(.+)$")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_HAS_LETTER_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True, slots=True)
class AttributionCheck:
    """Rewritten text plus the per-response counts the telemetry marker reports.

    ``tagged`` counts outlet-named tier attributions; ``unsupported`` how many of them no
    grounded label backs; ``generic`` the tier items that name no outlet at all (a class
    description such as ``official``), which are rewritten too; ``groups_rewritten`` how
    many bracket groups changed for either reason, which is the marker's render footprint,
    since several rewritten items in one group collapse to a single marker.
    """

    text: str
    tagged: int
    unsupported: int
    generic: int
    groups_rewritten: int


def _squash(text: str) -> str:
    """Lowercase alnum-only form, so ``Golf Channel`` meets ``golfchannel.com``."""
    return _NON_ALNUM_RE.sub("", text.lower())


def _identity_tokens(text: str) -> list[str]:
    return [token for token in _NON_ALNUM_RE.split(text.lower()) if token and token not in _STOP_TOKENS]


def _domain_cores(label: str) -> list[str]:
    """Every name-bearing label of a domain left of its public suffix: ``["lse"]`` for
    ``lse.ac.uk``, ``["nhc", "noaa"]`` for ``nhc.noaa.gov``, ``["grg"]`` for
    ``grg-supercentenarians.org``.

    A label renders as ``<title> — <domain>`` when the chunk carries both, so the domain
    is the last dash-separated segment. Self-cited sources list full hostnames, and the
    identity can sit in either part (``tropical.colostate.edu``, ``nhc.noaa.gov``), so each
    label counts. Stop and class words (``www``, ``news``) and cores shorter than
    ``_MIN_DOMAIN_CORE_CHARS`` never do: they sit inside far too many outlet names.
    """
    host = label.rsplit(" — ", 1)[-1].strip().lower()
    domain = registrable_domain(host) or host
    name_labels = [*host.removesuffix(domain).split("."), domain.split(".", 1)[0]]
    # A hyphenated label reads by its first run: ``grg`` for grg-supercentenarians.org.
    cores = [next(iter(_NON_ALNUM_RE.split(name_label)), "") for name_label in name_labels]
    return [
        core
        for core in cores
        if len(core) >= _MIN_DOMAIN_CORE_CHARS and core not in _STOP_TOKENS and core not in _DESCRIPTOR_TOKENS
    ]


def _is_subsequence(needle: str, haystack: str) -> bool:
    haystack_chars = iter(haystack)
    return all(char in haystack_chars for char in needle)


def _is_prefix_concatenation(needle: str, tokens: Sequence[str]) -> bool:
    """Whether ``needle`` reads as ``tokens`` abbreviated in order from the first.

    ``timesca`` is ``times`` + ``c``(entral) + ``a``(sia), which is how The Times of
    Central Asia's own domain is built. Anchored at the first token and requiring at least
    two of them, so an unrelated core cannot walk in from the middle: ``dailystar`` is not
    ``daily`` + a prefix of ``express``.
    """

    def walk(position: int, token_index: int, consumed: int) -> bool:
        if position == len(needle):
            return consumed >= 2
        if token_index >= len(tokens):
            return False
        token = tokens[token_index]
        for take in range(min(len(token), len(needle) - position), 0, -1):
            if token.startswith(needle[position : position + take]) and walk(
                position + take, token_index + 1, consumed + 1
            ):
                return True
        return False

    return bool(needle) and walk(0, 0, 0)


def _name_matches_label(name: str, label: str) -> bool:
    """Whether one grounded label backs one outlet name, under any of six rules.

    Each rule closes a shape the archive actually contains, and each only ever ADDS a
    keep:

    1. the name concatenates into the domain (``Golf Channel`` / golfchannel.com);
    2. every identity token appears in the domain (``The Guardian`` / guardian.co.uk);
    3. the token sets intersect on a token that is not a class word (``LSE Blogs`` /
       lse.ac.uk, but not ``Research Institute of Foo`` / demographic-research.org);
    4. a domain core (any label left of the public suffix) sits inside the name — the sub-brand shape
       (``Chosunbiz`` / chosun.com, ``iHeartRadio`` / iheart.com);
    5. a single-token name is a subsequence of the label — the name-abbreviates-the-outlet
       shape (``WaPo`` and ``WashPost`` / washingtonpost.com, ``RCP`` /
       realclearpolling.com, ``GEF`` / global-energy-flow.com). Restricted to single-token
       names because a subsequence test over a multiword name credits almost anything;
    6. a domain core abbreviates the name — the same relation the other way round
       (``Times of Central Asia`` / timesca.com).
    """
    name_squashed = _squash(name)
    label_squashed = _squash(label)
    if not name_squashed or not label_squashed:
        return False
    if name_squashed in label_squashed:
        return True
    name_tokens = _identity_tokens(name)
    if name_tokens and all(token in label_squashed for token in name_tokens):
        return True
    if (set(name_tokens) - _DESCRIPTOR_TOKENS) & set(_identity_tokens(label)):
        return True
    cores = _domain_cores(label)
    if any(core in name_squashed for core in cores):
        return True
    if len(name_tokens) == 1 and len(name_squashed) >= 2 and _is_subsequence(name_squashed, label_squashed):
        return True
    return any(_is_prefix_concatenation(core, name_tokens) for core in cores)


def _is_generic_descriptor(name: str) -> bool:
    """Whether ``name`` describes a class of source rather than naming one."""
    return all(token in _DESCRIPTOR_TOKENS for token in _identity_tokens(name))


def _attribution_alternatives(name: str) -> list[str]:
    """Named halves of a slash-joined attribution (``Reuters/AP``, ``official / GRG``).

    Returns ``[]`` when every half describes a class rather than naming an outlet
    (``official``, ``official/wire``), which is what makes a tag generic.
    """
    halves = [half.strip() for half in name.split("/") if half.strip()]
    return [half for half in halves if _HAS_LETTER_RE.search(half) and not _is_generic_descriptor(half)]


def _is_supported(name: str, labels: Sequence[str]) -> bool:
    """Supported when ANY named half matches ANY grounded label.

    Any-half rather than every-half: ``Reuters/AP`` on a record holding apnews.com is a
    correctly attributed wire story with a second outlet mentioned, not a fabrication.
    """
    return any(
        _name_matches_label(alternative, label) for alternative in _attribution_alternatives(name) for label in labels
    )


def _split_group_items(inner: str) -> list[tuple[str, str]]:
    """One ``(introducing separator, stripped item text)`` pair per non-empty item.

    The split itself is the shared grammar; what this adds is THIS pass's reading of an
    item — stripped, and dropped when nothing is left. (The citation-index strip drops an
    item with no alphanumeric character instead, which is why the shared iterator hands
    items over raw.)
    """
    return [(separator, item.strip()) for separator, item in iter_group_items(inner) if item.strip()]


def _rewrite_group(inner: str, labels: Sequence[str]) -> tuple[str | None, int, int, int]:
    """Rewrite one bracket group's inner text: ``(new inner or None, tagged, unsupported, generic)``.

    ``None`` means leave the group exactly as the model wrote it: it holds no tier grade at
    all (a bare-name bracket, or one of our own spliced ``[N]`` markers), or every item in
    it names an outlet the record backs.
    """
    items = _split_group_items(inner)
    if not any(_TIER_ITEM_RE.match(item) for _separator, item in items):
        return None, 0, 0, 0

    rendered: list[tuple[str, str]] = []
    tagged = 0
    unsupported = 0
    generic = 0
    for separator, item in items:
        tier_match = _TIER_ITEM_RE.match(item)
        name = tier_match.group(2).strip() if tier_match else item
        if name == UNVERIFIED_ATTRIBUTION_MARKER or not _HAS_LETTER_RE.search(name):
            rendered.append((separator, item))
            continue
        if not _attribution_alternatives(name):
            generic += 1
        else:
            tagged += 1
            if _is_supported(name, labels):
                rendered.append((separator, item))
                continue
            unsupported += 1
        # The tier GRADE goes with the outlet: the grade is an authority claim read off
        # that outlet, so it cannot outlive it, and a tag naming no outlet backs no grade.
        # Only the first rewritten item of a group renders a marker; a second "we could
        # not verify this" adds nothing.
        if unsupported + generic == 1:
            rendered.append((separator, UNVERIFIED_ATTRIBUTION_MARKER))
    if not unsupported and not generic:
        return None, tagged, 0, 0
    return join_group_items(rendered), tagged, unsupported, generic


def rewrite_unsupported_attributions(text: str, labels: Sequence[str]) -> AttributionCheck:
    """Replace tier-tag attributions no grounded ``label`` backs, and tags naming no outlet, with the marker.

    ``labels`` are the rendered grounding labels the forecaster is shown, so the check
    and the ``### Sources`` block can never disagree about what our record says.

    A group's surviving items keep their text and their own ``,`` / ``;`` separators; the
    rewritten ones collapse to ONE marker at the position of the first of them, because
    a second "we could not verify this" says nothing the first did not. Nothing outside a
    bracket is touched: the sentence a tag decorates comes through byte-identical.
    Idempotent — the marker is not itself an outlet name, so a second pass finds no
    qualifying item.

    An EMPTY ``labels`` is a measurement failure, not a verdict: a response whose chunks
    carry no renderable label (one archived section) gives the check no evidence base, and
    rewriting every tag off an empty record would dress our own render failure as the
    model's embellishment. The guard lives here rather than at the call site so no future
    caller can bypass it.
    """
    if not labels:
        return AttributionCheck(text=text, tagged=0, unsupported=0, generic=0, groups_rewritten=0)

    tagged = 0
    unsupported = 0
    generic = 0
    groups_rewritten = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal tagged, unsupported, generic, groups_rewritten
        rebuilt, group_tagged, group_unsupported, group_generic = _rewrite_group(match.group("inner"), labels)
        tagged += group_tagged
        unsupported += group_unsupported
        generic += group_generic
        if rebuilt is None:
            return match.group(0)
        groups_rewritten += 1
        # ``rebuild_group`` re-emits the space the shared pattern may have consumed in front
        # of the group; this pass never deletes a group, so that space always comes back.
        return rebuild_group(match, rebuilt)

    return AttributionCheck(
        text=BRACKET_GROUP_RE.sub(replace, text),
        tagged=tagged,
        unsupported=unsupported,
        generic=generic,
        groups_rewritten=groups_rewritten,
    )
