"""DuckDuckGo search backend (no API key required).

The primary path uses the `ddgs` package. If `ddgs` is missing or fails, the
provider falls back to a hand written parser for the DuckDuckGo Lite endpoint.
"""

import html
import re
from urllib.parse import parse_qs, unquote, urlparse

import requests

try:
    from ddgs import DDGS
except ImportError:
    DDGS = None


_LITE_URL = "https://lite.duckduckgo.com/lite/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_ANCHOR_RE = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_HREF_RE = re.compile(r"href\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_SNIPPET_RE = re.compile(
    r"class=['\"]result-snippet['\"][^>]*>(.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

_SNIPPET_CHARS = 300


def _clean(fragment: str) -> str:
    """Remove tags/entities from an HTML fragment and collapse whitespace."""
    text = html.unescape(_TAG_RE.sub("", fragment))
    return _SPACE_RE.sub(" ", text).strip()


def _resolve(href: str) -> str:
    """Unwrap DuckDuckGo redirect links (/l/?uddg=...) into the real URL."""
    if href.startswith("//"):
        href = f"https:{href}"
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg")
        if target:
            return unquote(target[0])
    return href


def _normalise(title, url, snippet) -> dict | None:
    """Build a source dict, or return None when the row is unusable."""
    title = _clean(str(title or ""))
    url = _resolve(str(url or "").strip())
    snippet = _clean(str(snippet or ""))[:_SNIPPET_CHARS]

    if not title or not url.startswith("http"):
        return None

    return {"title": title, "url": url, "snippet": snippet}


def _dedupe(sources: list[dict]) -> list[dict]:
    """Drop repeated URLs while keeping the original order."""
    seen = set()
    unique = []
    for source in sources:
        if source["url"] in seen:
            continue
        seen.add(source["url"])
        unique.append(source)
    return unique


class DuckDuckGoSearch:
    """Search the web through DuckDuckGo."""

    def __init__(self, max_results: int = 5, timeout: int = 15):
        self.max_results = max_results
        self.timeout = timeout

    def search(self, query: str) -> list[dict]:
        """Return up to ``max_results`` {"title", "url", "snippet"} dicts.

        Never raises: a network or parsing failure returns an empty list so the
        research graph can fall back to the model's own knowledge.
        """
        for backend in (self._search_ddgs, self._search_lite):
            try:
                sources = backend(query)
            except Exception:
                continue

            sources = _dedupe(sources)
            if sources:
                return sources[: self.max_results]

        return []

    def _search_ddgs(self, query: str) -> list[dict]:
        if DDGS is None:
            return []

        rows = DDGS(timeout=self.timeout).text(
            query,
            region="wt-wt",
            max_results=self.max_results,
        )

        sources = []
        for row in rows:
            source = _normalise(row.get("title"), row.get("href"), row.get("body"))
            if source:
                sources.append(source)
        return sources

    def _search_lite(self, query: str) -> list[dict]:
        response = requests.post(
            _LITE_URL,
            data={"q": query, "kl": "wt-wt"},
            headers=_HEADERS,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return self._parse_lite(response.text)

    @staticmethod
    def _parse_lite(markup: str) -> list[dict]:
        """Parse the Lite endpoint markup (anchors carry class='result-link')."""
        links = []
        for attributes, label in _ANCHOR_RE.findall(markup):
            if "result-link" not in attributes:
                continue
            match = _HREF_RE.search(attributes)
            if match:
                links.append((match.group(1), label))

        snippets = _SNIPPET_RE.findall(markup)

        sources = []
        for index, (href, label) in enumerate(links):
            snippet = snippets[index] if index < len(snippets) else ""
            source = _normalise(label, href, snippet)
            if source:
                sources.append(source)
        return sources
