import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


_FALSE_VALUES = ("0", "off", "false", "no")


def _int_env(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back to default."""
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default

    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    """Read an on/off flag (0/off/false/no disable it, anything else enables it)."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default

    return value.strip().lower() not in _FALSE_VALUES


DEFAULT_MAX_SOURCES = _int_env("MAX_SOURCES", 5)
DEFAULT_MAX_QUERIES = _int_env("MAX_QUERIES", 3)

SEARCH_PROVIDER = os.getenv("SEARCH_PROVIDER", "duckduckgo")

PLANNER_ENABLED = _bool_env("PLANNER", True)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ---------------------------------------------------------------------------
# Question understanding (Phase 1)
# ---------------------------------------------------------------------------
# Every question is understood once before the rest of the graph runs:
#
#   subject      = WHAT the question is about  (english / cybersecurity)
#   topic        = the main thing asked about
#   intent       = WHAT THE USER WANTS TO DO   (learn / practice / verify /
#                                                research)
#   task         = the specific operation requested (explain, solve, ...)
#   ambiguous    = the request would search differently under another reading
#   clarification= the question to ask back in that case
#   clarification_options = 2-4 numbered readings the user can pick from
#
# The lists are intentionally short: small local models classify more
# reliably against few options. Questions that fit neither subject fall
# back to DEFAULT_SUBJECT ("general"). An ambiguous request is routed to
# the `clarify` node instead of the planner, so no search runs until the
# meaning is settled.

SUBJECTS = ("english", "cybersecurity")
INTENTS = ("learn", "practice", "verify", "research")

# CLASSIFY=off skips the understanding call (state keeps the defaults).
CLASSIFY_ENABLED = _bool_env("CLASSIFY", True)

# Fallbacks when classification is off or the model replied nonsense.
DEFAULT_SUBJECT = "general"
DEFAULT_INTENT = "research"

# STRUCTURED_OUTPUT=off asks for JSON in the prompt only. The provider then
# never sees the schema, so the answer has to be parsed out of the text (the
# parser is forgiving, but structured output is the reliable path). Turn it off
# for a local model that does not support tool calling.
STRUCTURED_OUTPUT_ENABLED = _bool_env("STRUCTURED_OUTPUT", True)


# ---------------------------------------------------------------------------
# Reading the real pages (evidence)
# ---------------------------------------------------------------------------
# A search snippet is a two-line teaser, so an answer written from snippets
# alone is written from something the model never really read. The `evidence`
# node downloads the best sources and puts their text in the prompt instead:
#
#   EVIDENCE=off          -> answer from the search snippets (old behaviour)
#   EVIDENCE_SOURCES=3    -> how many pages to read (the best ranked ones)
#   EVIDENCE_CHARS=1500   -> how many characters to keep per page
#
# Only the top sources are read, each with a short timeout, and a page that
# cannot be downloaded is skipped, so the node never slows the run down much.
EVIDENCE_ENABLED = _bool_env("EVIDENCE", True)
EVIDENCE_SOURCES = _int_env("EVIDENCE_SOURCES", 3)
EVIDENCE_CHARS = _int_env("EVIDENCE_CHARS", 1500)


# ---------------------------------------------------------------------------
# Source quality ranking
# ---------------------------------------------------------------------------
# Every web source gets a quality score (0-100), from the most to the least
# reliable one:
#
#   NASA / NOAA / government ................ 100
#   University / scientific organization ....  85
#   Scientific paper ........................  75
#   Established educational website .........  60
#   Wikipedia ...............................  45
#   Reddit / forums .........................  25
#   unrecognised domain .....................  20
#
# `search/ranking.py` turns a URL into a score; the scores below and the
# domain lists in `data/source_domains.json` are the only two things you
# normally need to edit to make the ranking smarter. A domain matches
# when it is listed or when it is a sub-domain of a listed one
# (fr.wikipedia.org -> wikipedia.org); when several entries match, the most
# specific one wins (pubmed.ncbi.nlm.nih.gov stays a paper even though nih.gov
# is listed as government).

# Tier name, score and label. Order matters: the best tier comes first.
SOURCE_TIERS = (
    ("government", 100, "NASA / NOAA / government"),
    ("university", 85, "University / scientific organization"),
    ("scientific_paper", 75, "Scientific paper"),
    ("educational", 60, "Established educational website"),
    ("wikipedia", 45, "Wikipedia"),
    ("forum", 25, "Reddit / forums"),
    ("unknown", 20, "Unrecognised domain"),
)

SOURCE_TIER_SCORES = {name: score for name, score, _ in SOURCE_TIERS}
SOURCE_TIER_LABELS = {name: label for name, _, label in SOURCE_TIERS}

# Compact labels for the CLI table.
SOURCE_TIER_SHORT_LABELS = {
    "government": "government",
    "university": "university",
    "scientific_paper": "paper",
    "educational": "education",
    "wikipedia": "wikipedia",
    "forum": "forum",
    "unknown": "unrecognised",
}

# ---------------------------------------------------------------------------
# Source domains
# ---------------------------------------------------------------------------
# The domain tables live in `data/source_domains.json` rather than here: they
# are the one part of this file you are expected to edit often, and keeping
# them out means this module stays readable as code. `data/README.md` has the
# format and the reasoning behind the language-learning entries.
#
# This list only covers the domains worth listing by hand. Anything else falls
# back on the extension heuristics in `ranking.py` (.gov, .edu, .ac.uk, ...).

SOURCE_DOMAINS_FILE = os.getenv(
    "SOURCE_DOMAINS_FILE",
    str(Path(__file__).with_name("data") / "source_domains.json"),
)

#: Entries look like "a scheme" or "a host with a port", not a bare hostname.
_DOMAIN_ERRORS = (
    ("://", "a scheme, e.g. https://"),
    ("/", "a path or a trailing slash"),
    (":", "a port"),
)


def _load_source_domains(path: str) -> dict[str, tuple[str, ...]]:
    """Read the per-tier domain lists, or explain precisely what is wrong.

    A typo here is invisible at runtime: the domain just stops matching and
    scores 20 as "unrecognised", with nothing to say why. So the list is
    checked once, on load, and a bad entry is an error rather than a silent
    downgrade.
    """
    with open(path, encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from None

    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of tier -> domains")

    known = {name for name, _, _ in SOURCE_TIERS}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown tier(s) {', '.join(unknown)}; "
            f"expected one of {', '.join(sorted(known))}"
        )

    domains: dict[str, tuple[str, ...]] = {}
    seen: dict[str, str] = {}

    for tier, entries in raw.items():
        if not isinstance(entries, list):
            raise ValueError(f"{path}: '{tier}' must be a list of domains")

        cleaned: list[str] = []
        for entry in entries:
            if not isinstance(entry, str):
                raise ValueError(f"{path}: '{tier}' holds a non-string entry")

            domain = entry.strip()
            if not domain:
                raise ValueError(f"{path}: '{tier}' holds an empty domain")

            for needle, reason in _DOMAIN_ERRORS:
                if needle in domain:
                    raise ValueError(
                        f"{path}: '{domain}' looks like a URL, not a bare "
                        f"hostname - it has {reason}"
                    )

            if domain.startswith("www."):
                raise ValueError(
                    f"{path}: '{domain}' should not start with 'www.'"
                )

            if domain != domain.lower():
                raise ValueError(
                    f"{path}: '{domain}' should be lower case, like the URLs"
                )

            if domain.endswith("."):
                raise ValueError(
                    f"{path}: '{domain}' should not end with a dot"
                )

            previous = seen.get(domain)
            if previous is not None and previous != tier:
                raise ValueError(
                    f"{path}: '{domain}' is listed under both '{previous}' and "
                    f"'{tier}'; it can only score once"
                )

            seen[domain] = tier

            if domain not in cleaned:
                cleaned.append(domain)

        domains[tier] = tuple(cleaned)

    return domains


SOURCE_DOMAINS = _load_source_domains(SOURCE_DOMAINS_FILE)

# Ranking switch and citation threshold:
#   SOURCE_RANKING=off  -> sources keep their search order, no quality column
#   MIN_SOURCE_SCORE=60 -> keep only educational websites and better (a list
#                          that would end up empty is kept unchanged)
SOURCE_RANKING_ENABLED = _bool_env("SOURCE_RANKING", True)
MIN_SOURCE_SCORE = _int_env("MIN_SOURCE_SCORE", 0)

