"""Tests for the source quality ranking (search/ranking.py and its wiring).

Run them from the project root with:

    python -m unittest discover -s tests -v
"""

import unittest
from unittest import mock

import config
from graph import build_prompt
from main import quality_cell
from search import ranking
from search.formatter import format_context
from search.merger import merge_sources
from state import State

_SAVED_CONFIG: tuple[bool, int] | None = None


def setUpModule() -> None:
    """Pin the ranking defaults so ambient env variables cannot skew the tests."""
    global _SAVED_CONFIG
    _SAVED_CONFIG = (config.SOURCE_RANKING_ENABLED, config.MIN_SOURCE_SCORE)
    config.SOURCE_RANKING_ENABLED = True
    config.MIN_SOURCE_SCORE = 0


def tearDownModule() -> None:
    """Put the ambient configuration back."""
    assert _SAVED_CONFIG is not None
    config.SOURCE_RANKING_ENABLED, config.MIN_SOURCE_SCORE = _SAVED_CONFIG


def source(url: str, title: str = "Title") -> dict:
    """Build a source dict the way the search backends do."""
    return {"title": title, "url": url, "snippet": "snippet"}


class DomainTests(unittest.TestCase):
    def test_normalises_scheme_case_www_port_and_query(self):
        self.assertEqual(
            ranking.domain_of("HTTPS://WWW.NASA.gov/earth?x=1"), "nasa.gov"
        )
        self.assertEqual(ranking.domain_of("example.org:8080/a?b=1#c"), "example.org")
        self.assertEqual(ranking.domain_of("https://arxiv.org/abs/1234"), "arxiv.org")

    def test_empty_or_malformed_urls_are_unusable(self):
        self.assertEqual(ranking.domain_of(""), "")
        self.assertEqual(ranking.domain_of("   "), "")
        self.assertEqual(ranking.domain_of(None), "")
        self.assertEqual(ranking.domain_of("http://[bad"), "")


class ClassificationTests(unittest.TestCase):
    def assertTier(self, tier: str, urls: list[str]) -> None:
        for url in urls:
            self.assertEqual(ranking.classify(url), tier, url)

    def test_government_sites(self):
        self.assertTier(
            "government",
            [
                "https://www.nasa.gov/earth",
                "https://www.noaa.gov/",
                "https://legifrance.gouv.fr/",
                "https://www.gov.uk/government/",
                "https://commission.europa.eu/",
                "https://www.who.int/",
            ],
        )

    def test_university_and_scientific_organizations(self):
        self.assertTier(
            "university",
            [
                "https://www.mit.edu/",
                "https://www.ox.ac.uk/",
                "https://sciences.univ-paris.fr/",
                "https://www.cnrs.fr/",
                "https://home.cern/",
            ],
        )

    def test_scientific_papers(self):
        self.assertTier(
            "scientific_paper",
            [
                "https://arxiv.org/abs/2401.00001",
                "https://www.nature.com/articles/s41586-020",
                "https://doi.org/10.1038/xyz",
                "https://www.biorxiv.org/content/10.1101/1",
            ],
        )

    def test_most_specific_listed_domain_wins(self):
        # nih.gov is government, but pubmed and ncbi are paper repositories.
        self.assertTier("government", ["https://www.nih.gov/"])
        self.assertTier(
            "scientific_paper",
            [
                "https://pubmed.ncbi.nlm.nih.gov/1/",
                "https://www.ncbi.nlm.nih.gov/pmc/1/",
            ],
        )

    def test_educational_sites(self):
        self.assertTier(
            "educational",
            [
                "https://www.khanacademy.org/math",
                "https://www.britannica.com/science",
                "https://openstax.org/details/books",
            ],
        )

    def test_wikipedia_and_siblings(self):
        self.assertTier(
            "wikipedia",
            [
                "https://en.wikipedia.org/wiki/Sky",
                "https://www.wikidata.org/wiki/Q1",
                "https://commons.wikimedia.org/",
            ],
        )

    def test_forums_and_social_media(self):
        self.assertTier(
            "forum",
            [
                "https://www.reddit.com/r/learnpython/",
                "https://stackoverflow.com/questions/1",
                "https://www.youtube.com/watch?v=x",
                "https://myblog.blogspot.com/",
            ],
        )

    def test_unlisted_domain_with_a_forum_subdomain_is_a_forum(self):
        self.assertEqual(
            ranking.classify("https://forum.example-paris.fr/t/topic"),
            "forum",
        )

    def test_unrecognised_domains(self):
        self.assertTier(
            "unknown",
            [
                "https://randomblog.example.org/post",
                "https://www.notarxiv.org/",
                "https://evil-wikipedia.com/",
            ],
        )

    def test_unusable_urls_are_unknown(self):
        self.assertEqual(ranking.classify(""), "unknown")
        self.assertEqual(ranking.classify("not a url"), "unknown")

    def test_config_tables_cover_every_tier(self):
        tier_names = [name for name, _, _ in config.SOURCE_TIERS]
        self.assertNotIn("unknown", config.SOURCE_DOMAINS)

        for name in config.SOURCE_DOMAINS:
            self.assertIn(name, tier_names)

        for name in tier_names:
            self.assertIn(name, config.SOURCE_TIER_SHORT_LABELS)


class ScoreTests(unittest.TestCase):
    def test_tier_scores_are_strictly_decreasing(self):
        scores = [score for _, score, _ in config.SOURCE_TIERS]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(len(set(scores)), len(scores))

    def test_expected_scores(self):
        self.assertEqual(ranking.score("https://www.nasa.gov/"), 100)
        self.assertEqual(ranking.score("https://www.mit.edu/"), 85)
        self.assertEqual(ranking.score("https://arxiv.org/abs/1"), 75)
        self.assertEqual(ranking.score("https://www.britannica.com/"), 60)
        self.assertEqual(ranking.score("https://en.wikipedia.org/wiki/X"), 45)
        self.assertEqual(ranking.score("https://www.reddit.com/r/x"), 25)
        self.assertEqual(ranking.score("https://random.example.io/"), 20)

    def test_labels_and_short_labels(self):
        self.assertEqual(
            ranking.label("https://www.nasa.gov/"), "NASA / NOAA / government"
        )
        self.assertEqual(ranking.short_label("https://www.nasa.gov/"), "government")
        self.assertEqual(ranking.short_label("https://arxiv.org/abs/1"), "paper")
        self.assertEqual(ranking.short_label("https://www.reddit.com/r/x"), "forum")

    def test_ranking_can_be_disabled(self):
        with mock.patch.object(config, "SOURCE_RANKING_ENABLED", False):
            self.assertFalse(ranking.ranking_enabled())
            self.assertEqual(ranking.score("https://www.nasa.gov/"), 0)
            self.assertEqual(
                ranking.label("https://www.nasa.gov/"), ranking.RANKING_OFF_LABEL
            )
            self.assertEqual(
                ranking.short_label("https://www.nasa.gov/"), ranking.RANKING_OFF_LABEL
            )
            # classify() stays a pure helper even when the switch is off.
            self.assertEqual(ranking.classify("https://www.nasa.gov/"), "government")

    def test_min_score_helper(self):
        with mock.patch.object(config, "MIN_SOURCE_SCORE", 60):
            self.assertEqual(ranking.min_score(), 60)

        with mock.patch.object(config, "MIN_SOURCE_SCORE", 0):
            self.assertEqual(ranking.min_score(), 0)


class SourceDictTests(unittest.TestCase):
    def test_annotate_adds_tier_score_and_quality_without_mutating(self):
        raw = source("https://arxiv.org/abs/1", "Paper")
        annotated = ranking.annotate(raw)

        self.assertEqual(annotated["tier"], "scientific_paper")
        self.assertEqual(annotated["score"], 75)
        self.assertEqual(annotated["quality"], "Scientific paper")
        self.assertEqual(annotated["title"], "Paper")
        self.assertEqual(raw, source("https://arxiv.org/abs/1", "Paper"))

    def test_helpers_compute_values_that_are_missing(self):
        raw = source("https://en.wikipedia.org/wiki/X")

        self.assertEqual(ranking.source_score(raw), 45)
        self.assertEqual(ranking.source_quality(raw), "Wikipedia")
        self.assertEqual(ranking.source_short_quality(raw), "wikipedia")

    def test_stored_values_win_over_recomputation(self):
        stored = {
            "title": "t",
            "url": "u",
            "snippet": "s",
            "tier": "government",
            "score": 99,
            "quality": "custom",
        }

        self.assertEqual(ranking.source_score(stored), 99)
        self.assertEqual(ranking.source_quality(stored), "custom")
        self.assertEqual(ranking.source_short_quality(stored), "government")

    def test_sources_without_a_url_are_treated_as_unknown(self):
        broken = {"title": "t", "snippet": "s"}

        self.assertEqual(ranking.source_score(broken), 20)
        self.assertEqual(ranking.source_quality(broken), "Unrecognised domain")
        self.assertEqual(ranking.source_short_quality(broken), "unrecognised")

    def test_average_score(self):
        self.assertIsNone(ranking.average_score([]))
        self.assertEqual(
            ranking.average_score(
                [
                    source("https://www.nasa.gov/earth"),
                    source("https://en.wikipedia.org/wiki/X"),
                ]
            ),
            72.5,
        )


class RankAndSelectTests(unittest.TestCase):
    def setUp(self):
        self.mixed = [
            source("https://www.reddit.com/r/x", "forum"),
            source("https://www.nasa.gov/earth", "gov"),
            source("https://en.wikipedia.org/wiki/X", "wiki"),
            source("https://arxiv.org/abs/1", "paper"),
        ]

    def test_rank_sources_sorts_best_first(self):
        ranked = ranking.rank_sources(self.mixed)

        self.assertEqual(
            [item["url"] for item in ranked],
            [
                "https://www.nasa.gov/earth",
                "https://arxiv.org/abs/1",
                "https://en.wikipedia.org/wiki/X",
                "https://www.reddit.com/r/x",
            ],
        )
        self.assertEqual([item["score"] for item in ranked], [100, 75, 45, 25])

    def test_rank_sources_keeps_ties_in_search_order(self):
        same_score = [
            source("https://arxiv.org/abs/1", "a"),
            source("https://www.nature.com/articles/x", "b"),
            source("https://doi.org/10.1/x", "c"),
        ]
        ranked = ranking.rank_sources(same_score)

        self.assertEqual([item["title"] for item in ranked], ["a", "b", "c"])

    def test_rank_sources_on_an_empty_list(self):
        self.assertEqual(ranking.rank_sources([]), [])

    def test_select_sources_applies_the_limit_after_ranking(self):
        kept = ranking.select_sources(self.mixed, limit=2)

        self.assertEqual(len(kept), 2)
        self.assertEqual([item["score"] for item in kept], [100, 75])

    def test_weak_sources_are_dropped_above_the_threshold(self):
        with mock.patch.object(config, "MIN_SOURCE_SCORE", 60):
            kept = ranking.select_sources(self.mixed)

        self.assertEqual([item["score"] for item in kept], [100, 75])

    def test_the_best_source_survives_a_threshold_that_drops_everything(self):
        with mock.patch.object(config, "MIN_SOURCE_SCORE", 90):
            kept = ranking.select_sources(self.mixed)

        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["score"], 100)

    def test_threshold_and_sorting_are_ignored_when_the_ranking_is_off(self):
        with mock.patch.object(config, "SOURCE_RANKING_ENABLED", False):
            kept = ranking.select_sources(self.mixed, limit=4)

        self.assertEqual(
            [item["title"] for item in kept], ["forum", "gov", "wiki", "paper"]
        )


class MergerTests(unittest.TestCase):
    def test_duplicates_are_dropped_and_results_ranked(self):
        batches = [
            [
                source("https://www.reddit.com/r/x", "forum"),
                source("https://www.nasa.gov/earth", "gov"),
            ],
            [
                source("https://www.nasa.gov/earth", "gov again"),
                source("https://arxiv.org/abs/1", "paper"),
            ],
        ]
        merged = merge_sources(batches, max_results=10)

        self.assertEqual([item["title"] for item in merged], ["gov", "paper", "forum"])
        self.assertEqual(merged[0]["score"], 100)

    def test_limit_keeps_the_best_sources(self):
        merged = merge_sources(
            [
                [
                    source("https://www.reddit.com/r/x"),
                    source("https://www.nasa.gov/earth"),
                ]
            ],
            max_results=1,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["score"], 100)

    def test_sources_without_a_url_are_skipped(self):
        self.assertEqual(
            merge_sources([[{"title": "no url", "snippet": "s"}]], max_results=5),
            [],
        )


class FormatterTests(unittest.TestCase):
    def test_context_carries_the_quality_score(self):
        context = format_context([source("https://www.nasa.gov/earth", "Earth")])

        self.assertIn("[1] Earth", context)
        self.assertIn("Quality: 100/100 (NASA / NOAA / government)", context)

    def test_empty_context(self):
        self.assertEqual(format_context([]), "")

    def test_no_quality_line_when_the_ranking_is_off(self):
        with mock.patch.object(config, "SOURCE_RANKING_ENABLED", False):
            context = format_context([source("https://www.nasa.gov/earth", "Earth")])

        self.assertIn("[1] Earth", context)
        self.assertNotIn("Quality:", context)


class PromptTests(unittest.TestCase):
    def test_prompt_asks_to_prefer_the_best_sources(self):
        state = State(
            question="Why is the sky blue?",
            context=format_context([source("https://www.nasa.gov/earth", "Earth")]),
        )
        prompt = build_prompt(state)

        self.assertIn("Cite every source you use", prompt)
        self.assertIn("quality score", prompt)
        self.assertIn("low scored source", prompt)

    def test_prompt_without_sources_is_unchanged(self):
        prompt = build_prompt(State(question="Q"))

        self.assertIn("No web sources were available", prompt)
        self.assertNotIn("quality score", prompt)

    def test_prompt_without_quality_guidance_when_the_ranking_is_off(self):
        state = State(question="Q", context="context here")

        with mock.patch.object(config, "SOURCE_RANKING_ENABLED", False):
            prompt = build_prompt(state)

        self.assertIn("Cite every source you use", prompt)
        self.assertNotIn("quality score", prompt)


class CliTests(unittest.TestCase):
    def test_quality_cell_shows_score_and_label(self):
        cell = quality_cell(source("https://www.nasa.gov/earth"))

        self.assertIn("100", cell)
        self.assertIn("government", cell)

    def test_quality_cell_for_weak_sources(self):
        cell = quality_cell(source("https://www.reddit.com/r/x"))

        self.assertIn("25", cell)
        self.assertIn("forum", cell)


if __name__ == "__main__":
    unittest.main()
