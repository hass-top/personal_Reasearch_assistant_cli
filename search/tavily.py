"""Tavily search backend (requires an API key)."""

from tavily import TavilyClient

_SNIPPET_CHARS = 300


class TavilySearch:
    """Search the web through the Tavily API."""

    def __init__(self, api_key: str, max_results: int = 5):
        if not api_key:
            raise ValueError("Tavily API key required")

        self.client = TavilyClient(api_key=api_key)
        self.max_results = max_results

    def search(self, query: str) -> list[dict]:
        """Return up to ``max_results`` {"title", "url", "snippet"} dicts.

        Never raises: an API failure returns an empty list so the research graph
        can fall back to the model's own knowledge.
        """
        try:
            response = self.client.search(query, max_results=self.max_results)
        except Exception:
            return []

        sources = []
        for row in response.get("results", []):
            url = str(row.get("url") or "").strip()
            if not url.startswith("http"):
                continue

            sources.append(
                {
                    "title": str(row.get("title") or url).strip(),
                    "url": url,
                    "snippet": str(row.get("content") or "").strip()[
                        :_SNIPPET_CHARS
                    ],
                }
            )

        return sources[: self.max_results]
