"""Build the subject strategy for a classified question.

Same idea as ``search.factory.create_search`` / ``llm.factory.create_llm``:
the graph asks for a subject and gets the matching strategy back. Unknown
(or empty) subjects always fall back to the general strategy, and a new
subject can be plugged in with ``register_strategy`` without touching the
graph.
"""

from config import DEFAULT_SUBJECT

from .base import SubjectStrategy
from .cybersecurity import CybersecurityStrategy
from .english import EnglishStrategy
from .general import GeneralStrategy

# subject -> strategy class, in fallback-check order (general first).
_STRATEGIES: dict[str, type[SubjectStrategy]] = {
    "general": GeneralStrategy,
    "english": EnglishStrategy,
    "cybersecurity": CybersecurityStrategy,
}


def _compact(value: str) -> str:
    """Lowercase and drop punctuation ("Cyber Security" -> "cybersecurity")."""
    return "".join(ch for ch in (value or "").lower() if ch.isalnum())


def register_strategy(subject: str, strategy: type[SubjectStrategy]) -> None:
    """Plug in a new subject (and later its own tools) at runtime."""
    _STRATEGIES[_compact(subject)] = strategy


def strategy_subjects() -> tuple[str, ...]:
    """Subjects that have a registered strategy ("general" included)."""
    return tuple(_STRATEGIES)


def create_strategy(subject: str | None = None) -> SubjectStrategy:
    """Build the strategy for ``subject``, falling back to the general one.

    Matching is forgiving (like the classifier snapping): "Cyber Security",
    "the english language" or "CVE stuff" all land on the right strategy.
    """
    cleaned = " ".join((subject or "").lower().split())
    compact = _compact(cleaned)

    # Exact match first, so a longer name never shadows a shorter one.
    for name, strategy in _STRATEGIES.items():
        if cleaned == name or compact == _compact(name):
            return strategy()

    # Otherwise the subject must contain the registered name (or the other
    # way round), which keeps "mathematics"-style answers working.
    for name, strategy in _STRATEGIES.items():
        key = _compact(name)
        if key and (key in compact or compact in key):
            return strategy()

    return _STRATEGIES[_compact(DEFAULT_SUBJECT)]()