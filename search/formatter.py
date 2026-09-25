"""Helpers that turn search results into prompt text."""

from .ranking import ranking_enabled, source_quality, source_score


def format_context(sources: list[dict]) -> str:
    """Render sources as a numbered block that is injected into the prompt.

    Every source carries its quality score, so the model can lean on the
    reliable ones and say when a claim only rests on a forum or on an
    unrecognised domain.
    """
    if not sources:
        return ""

    blocks = []
    for index, source in enumerate(sources, start=1):
        quality = ""
        if ranking_enabled():
            quality = (
                f"Quality: {source_score(source)}/100"
                f" ({source_quality(source)})\n"
            )

        blocks.append(
            f"[{index}] {source['title']}\n"
            f"{source['url']}\n"
            f"{quality}"
            f"{source['snippet']}"
        )

    return "\n\n".join(blocks)


