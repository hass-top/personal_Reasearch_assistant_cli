"""Subject strategies: one profile + one behaviour per subject (Phase 2).

Every subject the assistant understands is described by a ``SubjectProfile``
(static facts: description, preferred source types, important features,
verification rules and the search guidance used to plan queries) and driven
by a ``SubjectStrategy`` (how the graph turns that profile into prompts).

The base class already renders the profile into the two prompt directives
the graph needs, so a concrete strategy normally only declares its
``profile``. A strategy may also carry ``tools``: names of subject specific
tools (a conjugator for english, a CVE lookup for cybersecurity, ...) that
the graph can run per subject later.

Strategies are built through ``profiles.factory.create_strategy(subject)``,
the same way search backends come from ``search.factory.create_search``.
"""

from dataclasses import dataclass, field


@dataclass
class SearchGuidance:
    """How to search for this subject (feeds the planner node)."""

    query_hints: list[str] = field(default_factory=list)
    lexicon: list[str] = field(default_factory=list)
    preferred_domains: list[str] = field(default_factory=list)
    avoid_domains: list[str] = field(default_factory=list)
    require_recent: bool = False


@dataclass
class SubjectProfile:
    """Static description of a subject (english, cybersecurity, ...)."""

    name: str
    description: str
    preferred_source_types: list[str] = field(default_factory=list)
    important_features: list[str] = field(default_factory=list)
    verification_rules: list[str] = field(default_factory=list)
    search_guidance: SearchGuidance = field(default_factory=SearchGuidance)


class SubjectStrategy:
    """Behaviour for one subject (the Strategy part of the pattern).

    Subclasses set ``profile`` (and optionally ``tools``) and may override
    ``planner_directive`` / ``answer_directive`` to change how the profile
    reaches the prompts. The default implementations below are built
    entirely from the profile, so most subjects need no extra code.
    """

    #: Subject specific tools for this field, e.g. ``("cve_lookup",)``.
    #: Empty by default; kept as the extension point for per-subject tools.
    tools: tuple[str, ...] = ()

    #: Set by every concrete subclass.
    profile: SubjectProfile

    def __init__(self, profile: SubjectProfile | None = None):
        # Optional override, mainly useful for tests and dynamic profiles.
        if profile is not None:
            self.profile = profile

    @property
    def name(self) -> str:
        return self.profile.name

    # --- directives (used by the AI nodes of the graph) --------------------

    def planner_directive(self) -> str:
        """Extra instructions for the `plan` node (search queries)."""
        profile = self.profile
        guidance = profile.search_guidance
        lines = [f"Subject: {profile.name} - {profile.description}"]

        if guidance.query_hints:
            lines.append("Query guidance:")
            lines.extend(f"- {hint}" for hint in guidance.query_hints)

        if guidance.lexicon:
            lines.append(
                "Use this field's vocabulary when useful: "
                + ", ".join(guidance.lexicon)
                + "."
            )

        if guidance.preferred_domains:
            lines.append(
                "Prefer results from: " + ", ".join(guidance.preferred_domains) + "."
            )

        if guidance.avoid_domains:
            lines.append(
                "Avoid results from: " + ", ".join(guidance.avoid_domains) + "."
            )

        if guidance.require_recent:
            lines.append(
                "Prefer recent sources: this field changes fast, so prefer "
                "sources from the last couple of years."
            )

        return "\n".join(lines)

    def answer_directive(self) -> str:
        """Extra instructions for the `research` node (writing the answer)."""
        profile = self.profile
        lines = []

        if profile.preferred_source_types:
            lines.append(
                "Preferred source types for this subject: "
                + ", ".join(profile.preferred_source_types)
                + "."
            )

        if profile.important_features:
            lines.append("A good answer in this field includes:")
            lines.extend(f"- {feature}" for feature in profile.important_features)

        if profile.verification_rules:
            lines.append("Verify before finalising:")
            lines.extend(f"- {rule}" for rule in profile.verification_rules)

        return "\n".join(lines)