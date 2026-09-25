"""Helpers to combine results coming from several search queries."""

from .ranking import select_sources


def merge_sources(batches: list[list[dict]], max_results: int) -> list[dict]:
    """Merge search batches, score every source and keep the best ones.

    Duplicate URLs are dropped (the first occurrence wins), then the sources are
    annotated with their quality score and sorted from the most to the least
    reliable before ``max_results`` is applied. The numbering used in the prompt
    ([1], [2], ...) therefore points at the strongest sources first.
    """
    seen = set()
    merged = []

    for batch in batches:
        for source in batch:
            url = source.get("url")
            if not url or url in seen:
                continue

            seen.add(url)
            merged.append(source)

    return select_sources(merged, limit=max_results)

