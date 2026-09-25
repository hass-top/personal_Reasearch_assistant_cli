"""Download the real text of the best search results.

A search snippet is a teaser: two lines written for a results page, not enough
to answer from. This module fetches the pages themselves and turns them into
plain text, so the research node writes the answer from what the pages actually
say.

No new dependency: the HTML is stripped with the standard library and fetched
with ``requests`` (already used by the DuckDuckGo backend).
"""

from html.parser import HTMLParser

import requests

from .ranking import ranking_enabled, source_quality, source_score

_USER_AGENT = "Mozilla/5.0 (compatible; research-assistant/1.0)"
_TIMEOUT = 5
# A page with less text than this is a landing page, a cookie banner or a
# JavaScript shell: nothing useful to quote, so it is skipped.
_MIN_TEXT_CHARS = 120
# Tags whose content is never part of the article.
_SKIPPED_TAGS = frozenset(
    {"script", "style", "noscript", "nav", "footer", "header", "aside", "form"}
)


class _TextExtractor(HTMLParser):
    """Collect the visible text of an HTML page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIPPED_TAGS:
            self._hidden += 1

    def handle_endtag(self, tag):
        if tag in _SKIPPED_TAGS and self._hidden:
            self._hidden -= 1

    def handle_data(self, data):
        if self._hidden:
            return

        text = " ".join(data.split())
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


def html_to_text(html: str) -> str:
    """Turn an HTML page into readable plain text.

    A malformed page still yields whatever was readable before the error, so
    this never raises.
    """
    parser = _TextExtractor()

    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass

    return parser.text()


def fetch_url(url: str, timeout: int = _TIMEOUT) -> str:
    """Download one page and return its readable text ("" on any failure)."""
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": _USER_AGENT},
        )
        response.raise_for_status()
    except Exception:
        return ""

    # A PDF or an image has no text this module can read.
    content_type = str(response.headers.get("Content-Type", "")).lower()
    if content_type and "html" not in content_type and "text" not in content_type:
        return ""

    return html_to_text(response.text)


def fetch_evidence(
    sources: list[dict],
    limit: int = 3,
    max_chars: int = 1500,
    timeout: int = _TIMEOUT,
) -> list[dict]:
    """Read the best ``limit`` sources and return what they really say.

    Each entry is ``{"title", "url", "text", "quality"}``. Sources already
    carrying ``raw_content`` (Tavily returns the page text with the search
    results) are used as they are instead of being downloaded twice.

    Never raises: a page that cannot be read is skipped, and a run without any
    readable page simply falls back to the search snippets.
    """
    evidence = []

    for source in sources[: max(0, limit)]:
        url = str(source.get("url") or "").strip()
        title = str(source.get("title") or url).strip()

        text = str(source.get("raw_content") or "").strip()
        if not text:
            text = fetch_url(url, timeout=timeout)

        # Some backends hand back raw HTML in `raw_content`.
        if "<" in text and ">" in text:
            text = html_to_text(text)

        text = " ".join(text.split())
        if len(text) < _MIN_TEXT_CHARS:
            continue

        quality = ""
        if ranking_enabled():
            quality = f"{source_score(source)}/100 ({source_quality(source)})"

        evidence.append(
            {
                "title": title,
                "url": url,
                "text": text[:max_chars],
                "quality": quality,
            }
        )

    return evidence
