import os

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
# `search/ranking.py` turns a URL into a score; the tables below are the only
# thing you normally need to edit to make the ranking smarter. A domain matches
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

# Domains per tier: add your own domains here.
SOURCE_DOMAINS = {
    "government": {
        "nasa.gov",
        "noaa.gov",
        "usgs.gov",
        "nist.gov",
        "cdc.gov",
        "nih.gov",
        "nsf.gov",
        "epa.gov",
        "fda.gov",
        "who.int",
        "unicef.org",
        "unesco.org",
        "europa.eu",
        "santepubliquefrance.fr",
        "insee.fr",
    },
    "university": {
        "cnrs.fr",
        "inserm.fr",
        "inria.fr",
        "pasteur.fr",
        "institutpasteur.fr",
        "cea.fr",
        "ird.fr",
        "onera.fr",
        "cern.ch",
        "home.cern",
        "esa.int",
        "ipcc.ch",
        "mpg.de",
        "helmholtz.de",
        "healthychildren.org",
    },
    "scientific_paper": {
        "arxiv.org",
        "biorxiv.org",
        "medrxiv.org",
        "hal.science",
        "pubmed.ncbi.nlm.nih.gov",
        "ncbi.nlm.nih.gov",
        "doi.org",
        "nature.com",
        "science.org",
        "sciencedirect.com",
        "springer.com",
        "ieee.org",
        "acm.org",
        "plos.org",
        "mdpi.com",
        "frontiersin.org",
        "jstor.org",
        "scielo.org",
        "semanticscholar.org",
        "researchgate.net",
        "tandfonline.com",
        "wiley.com",
        "bmj.com",
        "thelancet.com",
        "cell.com",
        "acs.org",
        "iop.org",
        "oup.com",
    },
    "educational": {
        "khanacademy.org",
        "britannica.com",
        "nationalgeographic.com",
        "natgeokids.com",
        "openstax.org",
        "ck12.org",
        "coursera.org",
        "edx.org",
        "fun-mooc.fr",
        "openclassrooms.com",
        "larousse.fr",
        "universalis.fr",
        "w3schools.com",
        "bbc.co.uk",
        "pbs.org",
        "pbslearningmedia.org",
        "kidshealth.org",
        "pourlascience.fr",
        "futura-sciences.com",
        "science-et-vie.com",
        "lelivrescolaire.fr",
    },
    "wikipedia": {
        "wikipedia.org",
        "wikimedia.org",
        "wiktionary.org",
        "wikibooks.org",
        "wikisource.org",
        "wikiversity.org",
        "wikinews.org",
        "wikiquote.org",
        "wikidata.org",
        "wikiwand.com",
    },
    "forum": {
        "reddit.com",
        "quora.com",
        "stackexchange.com",
        "stackoverflow.com",
        "askubuntu.com",
        "superuser.com",
        "serverfault.com",
        "answers.yahoo.com",
        "wikihow.com",
        "facebook.com",
        "instagram.com",
        "twitter.com",
        "x.com",
        "tiktok.com",
        "youtube.com",
        "youtu.be",
        "dailymotion.com",
        "vimeo.com",
        "twitch.tv",
        "linkedin.com",
        "pinterest.com",
        "tumblr.com",
        "discord.com",
        "t.me",
        "medium.com",
        "blogspot.com",
        "wordpress.com",
        "over-blog.com",
        "wixsite.com",
        "weebly.com",
        "forumactif.com",
        "github.com",
        "gitlab.com",
    },
}

# Ranking switch and citation threshold:
#   SOURCE_RANKING=off  -> sources keep their search order, no quality column
#   MIN_SOURCE_SCORE=60 -> keep only educational websites and better (a list
#                          that would end up empty is kept unchanged)
SOURCE_RANKING_ENABLED = _bool_env("SOURCE_RANKING", True)
MIN_SOURCE_SCORE = _int_env("MIN_SOURCE_SCORE", 0)

