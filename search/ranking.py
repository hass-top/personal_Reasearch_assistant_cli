"""Quality score for the web sources returned by the search backends.

Sources are ranked from the most to the least reliable:

    NASA / NOAA / government ................ 100
    University / scientific organization ....  85
    Scientific paper ........................  75
    Established educational website .........  60
    Wikipedia ...............................  45
    Reddit / forums .........................  25
    unrecognised domain .....................  20

The tiers and their scores live in `config.py`; the domain tables they are
matched against live in `data/source_domains.json`. This module only holds the
logic. Nothing here touches the network.
"""

from urllib.parse import urlsplit

import config
from config import (
    SOURCE_DOMAINS,
    SOURCE_TIER_LABELS,
    SOURCE_TIER_SCORES,
    SOURCE_TIER_SHORT_LABELS,
    SOURCE_TIERS,
)

# Tier names, as defined by SOURCE_TIERS in config.py.
GOVERNMENT_TIER = "government"
UNIVERSITY_TIER = "university"
FORUM_TIER = "forum"
UNKNOWN_TIER = SOURCE_TIERS[-1][0]

RANKING_OFF_LABEL = "not ranked"

# Best tier first: used to break ties when several listed domains match.
_TIER_ORDER = tuple(name for name, _, _ in SOURCE_TIERS)

# Extension labels used when a domain is not in data/source_domains.json.
_GOVERNMENT_LABELS = frozenset({"gov", "gouv", "gob", "governo", "mil"})
_UNIVERSITY_LABELS = frozenset({"edu", "ac"})
_UNIVERSITY_PREFIXES = ("univ", "universit")
_FORUM_TOKENS = ("forum",)


def ranking_enabled() -> bool:
    """True when sources should be scored (set SOURCE_RANKING=off to disable)."""
    return bool(config.SOURCE_RANKING_ENABLED)


def min_score() -> int:
    """Lowest score a source may have to stay in the list (0 means no filter)."""
    return max(0, int(config.MIN_SOURCE_SCORE))


def domain_of(url: str) -> str:
    """Host of an URL, lower case and without ``www.`` (empty when unusable).

    >>> domain_of("https://WWW.NASA.gov/topics/earth?x=1")
    'nasa.gov'
    """
    raw = (url or "").strip()
    if not raw:
        return ""

    if "://" not in raw:
        # Network-path reference, so urlsplit still finds the host.
        raw = f"//{raw}"

    try:
        host = urlsplit(raw).hostname or ""
    except ValueError:
        # Malformed host, e.g. an unclosed "[" bracket.
        return ""

    return host.strip().lower().rstrip(".").removeprefix("www.")


def _listed_tier(domain: str) -> str | None:
    """Tier of the most specific listed domain matching ``domain``.

    The longest entry wins, so ``pubmed.ncbi.nlm.nih.gov`` stays a scientific
    paper even though ``nih.gov`` is listed as government.
    """
    best: tuple[int, int, str] | None = None

    for rank, name in enumerate(_TIER_ORDER):
        for entry in SOURCE_DOMAINS.get(name, ()):
            if domain != entry and not domain.endswith(f".{entry}"):
                continue

            candidate = (len(entry), -rank, name)
            if best is None or candidate > best:
                best = candidate

    return best[2] if best else None


def _has_institutional_label(domain: str, labels: frozenset[str]) -> bool:
    """True for .gov / .gouv.fr / .ac.uk style extensions."""
    parts = domain.split(".")
    if len(parts) < 2:
        return False

    if parts[-1] in labels:
        return True

    # Country code extensions: legifrance.gouv.fr, ox.ac.uk, gov.uk
    return parts[-2] in labels and len(parts[-1]) == 2


def classify(url: str) -> str:
    """Tier name of a source: listed domains first, then extension heuristics."""
    domain = domain_of(url)
    if not domain:
        return UNKNOWN_TIER

    listed = _listed_tier(domain)
    if listed is not None:
        return listed

    if _has_institutional_label(domain, _GOVERNMENT_LABELS):
        return GOVERNMENT_TIER

    university_label = any(
        part.startswith(_UNIVERSITY_PREFIXES) for part in domain.split(".")[:-1]
    )
    if _has_institutional_label(domain, _UNIVERSITY_LABELS) or university_label:
        return UNIVERSITY_TIER

    if any(
        token in part
        for part in domain.split(".")[:-1]
        for token in _FORUM_TOKENS
    ):
        return FORUM_TIER

    return UNKNOWN_TIER


def score(url: str) -> int:
    """Quality score (0-100) of a source, or 0 when the ranking is disabled."""
    if not ranking_enabled():
        return 0

    return SOURCE_TIER_SCORES[classify(url)]


def label(url: str) -> str:
    """Quality label of a source, e.g. "University / scientific organization"."""
    if not ranking_enabled():
        return RANKING_OFF_LABEL

    return SOURCE_TIER_LABELS[classify(url)]


def short_label(url: str) -> str:
    """Compact quality label, used by the CLI table."""
    if not ranking_enabled():
        return RANKING_OFF_LABEL

    return SOURCE_TIER_SHORT_LABELS[classify(url)]


# --- Working on the {"title", "url", "snippet"} dicts ------------------------


def annotate(source: dict) -> dict:
    """Return a copy of ``source`` with ``tier``, ``score`` and ``quality``."""
    url = str(source.get("url") or "")
    return {
        **source,
        "tier": classify(url),
        "score": score(url),
        "quality": label(url),
    }


def source_score(source: dict) -> int:
    """Score of a source dict, computed on the fly when it is missing."""
    if "score" in source:
        return int(source["score"])

    return score(str(source.get("url") or ""))


def source_quality(source: dict) -> str:
    """Quality label of a source dict, computed on the fly when it is missing."""
    if "quality" in source:
        return str(source["quality"])

    return label(str(source.get("url") or ""))


def source_short_quality(source: dict) -> str:
    """Compact quality label of a source dict (CLI table)."""
    if not ranking_enabled():
        return RANKING_OFF_LABEL

    tier = str(source.get("tier") or classify(str(source.get("url") or "")))
    return SOURCE_TIER_SHORT_LABELS[tier]


def rank_sources(sources: list[dict]) -> list[dict]:
    """Annotate sources and sort them from the most to the least reliable.

    The sort is stable: sources sharing a score keep the order the search
    backends returned them in, and when SOURCE_RANKING is off every score is 0,
    so the original order is preserved.
    """
    return sorted(
        (annotate(source) for source in sources),
        key=lambda source: -int(source["score"]),
    )


def select_sources(sources: list[dict], limit: int | None = None) -> list[dict]:
    """Annotate, sort by quality, drop weak sources, then apply ``limit``.

    Sources scoring below MIN_SOURCE_SCORE are dropped, unless that would leave
    an empty list: the best source is always kept so the model still gets
    something to work with.
    """
    ranked = rank_sources(sources)

    threshold = min_score() if ranking_enabled() else 0
    if threshold > 0:
        kept = [source for source in ranked if int(source["score"]) >= threshold]
        ranked = kept or ranked[:1]

    if limit is None:
        return ranked

    return ranked[:limit]


def average_score(sources: list[dict]) -> float | None:
    """Mean quality score of a list of sources (``None`` when it is empty)."""
    if not sources:
        return None

    return round(sum(source_score(source) for source in sources) / len(sources), 1)
