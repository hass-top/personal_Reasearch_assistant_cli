from .duckduckgo import DuckDuckGoSearch


def create_search(
    provider: str,
    api_key: str | None = None,
    max_results: int = 5,
):
    if provider in ("duckduckgo", "ddgs"):
        return DuckDuckGoSearch(max_results=max_results)

    if provider == "tavily":
        if api_key is None:
            raise ValueError("Tavily API key required")

        from .tavily import TavilySearch

        return TavilySearch(
            api_key=api_key,
            max_results=max_results,
        )

    raise ValueError(f"unsupported search provider: {provider}")
