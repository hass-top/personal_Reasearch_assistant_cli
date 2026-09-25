"""Fallback strategy for every subject without a dedicated one."""

from .base import SearchGuidance, SubjectProfile, SubjectStrategy


class GeneralStrategy(SubjectStrategy):
    """Neutral profile: keeps the prompts exactly as generic as before."""

    tools: tuple[str, ...] = ()

    profile = SubjectProfile(
        name="general",
        description="any question that fits no dedicated subject",
        preferred_source_types=[],
        important_features=[],
        verification_rules=[],
        search_guidance=SearchGuidance(),
    )