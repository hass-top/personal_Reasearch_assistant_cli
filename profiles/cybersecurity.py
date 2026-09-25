"""Cybersecurity subject strategy: vulnerabilities, attacks and defences."""

from .base import SearchGuidance, SubjectProfile, SubjectStrategy


class CybersecurityStrategy(SubjectStrategy):
    """Plans queries and answers the way a security advisor would."""

    #: Later: ("cve_lookup", "nvd_search") once subject tools exist.
    tools: tuple[str, ...] = ()

    profile = SubjectProfile(
        name="cybersecurity",
        description=(
            "computer security: vulnerabilities, attacks, defences and hardening"
        ),
        preferred_source_types=["government", "scientific_paper", "university"],
        important_features=[
            "the affected products and versions",
            "the mitigation or patch, step by step",
            "CVE / advisory identifiers and a publication date",
        ],
        verification_rules=[
            "prefer official advisories (cisa.gov, nvd.nist.gov, the vendor) "
            "over blog posts",
            "check the publication date: guidance that is years old may be "
            "outdated",
            "cross-check any critical claim with a second independent source",
        ],
        search_guidance=SearchGuidance(
            query_hints=[
                "add the identifier: CVE-2024-..., advisory, CVSS",
                "name the product and its version",
                "add a verb: exploit, mitigation, patch, detection, IOC",
            ],
            lexicon=["CVE", "CVSS", "exploit", "zero-day", "IOC", "hardening"],
            preferred_domains=[
                "nvd.nist.gov",
                "cisa.gov",
                "cert.fr",
                "cve.mitre.org",
                "owasp.org",
            ],
            avoid_domains=["reddit.com", "quora.com", "medium.com"],
            require_recent=True,
        ),
    )