"""Tests for Phase 2: subject profiles, strategies and their factory."""

import unittest
from types import SimpleNamespace
from unittest import mock

import config
import graph
from profiles.base import SearchGuidance, SubjectProfile, SubjectStrategy
from profiles.cybersecurity import CybersecurityStrategy
from profiles.english import EnglishStrategy
from profiles.factory import (
    create_strategy,
    register_strategy,
    strategy_subjects,
)
from profiles.general import GeneralStrategy
from state import State


def _unregister(subject: str) -> None:
    """Undo register_strategy (tests share one registry)."""
    from profiles import factory

    factory._STRATEGIES.pop(subject, None)


class ProfileTests(unittest.TestCase):
    """Every profile carries the full SubjectProfile contract."""

    def test_profile_fields(self):
        profile = EnglishStrategy.profile

        self.assertEqual(profile.name, "english")
        self.assertTrue(profile.description)
        self.assertTrue(profile.preferred_source_types)
        self.assertTrue(profile.important_features)
        self.assertTrue(profile.verification_rules)
        self.assertIsInstance(profile.search_guidance, SearchGuidance)

    def test_search_guidance_fields(self):
        guidance = CybersecurityStrategy.profile.search_guidance

        self.assertTrue(guidance.query_hints)
        self.assertTrue(guidance.lexicon)
        self.assertTrue(guidance.preferred_domains)
        self.assertTrue(guidance.avoid_domains)
        self.assertIsInstance(guidance.require_recent, bool)

    def test_every_config_subject_has_a_dedicated_strategy(self):
        for subject in config.SUBJECTS:
            strategy = create_strategy(subject)

            self.assertEqual(strategy.name, subject)
            self.assertNotIsInstance(strategy, GeneralStrategy)


class FactoryTests(unittest.TestCase):
    """create_strategy mirrors search.factory: right object, safe fallback."""

    def test_unknown_subject_falls_back_to_general(self):
        for value in (None, "", "mathematics", "cooking"):
            self.assertIsInstance(create_strategy(value), GeneralStrategy)

    def test_sloppy_answers_are_snapped_onto_the_vocabulary(self):
        self.assertIsInstance(
            create_strategy("Cyber Security"), CybersecurityStrategy
        )
        self.assertIsInstance(create_strategy("ENGLISH"), EnglishStrategy)
        self.assertIsInstance(create_strategy("the english language"), EnglishStrategy)

    def test_default_subject_is_general(self):
        self.assertEqual(config.DEFAULT_SUBJECT, "general")
        self.assertIsInstance(
            create_strategy(config.DEFAULT_SUBJECT), GeneralStrategy
        )

    def test_strategy_subjects_are_registered(self):
        subjects = strategy_subjects()

        self.assertIn("general", subjects)
        for subject in config.SUBJECTS:
            self.assertIn(subject, subjects)

    def test_register_strategy_plugs_in_a_new_subject(self):
        class MathsStrategy(SubjectStrategy):
            profile = SubjectProfile(name="maths", description="numbers")
            tools = ("unit_converter",)

        register_strategy("maths", MathsStrategy)
        self.addCleanup(_unregister, "maths")

        # Exact and "subject contains the registered name" both resolve.
        strategy = create_strategy("maths")
        self.assertIsInstance(strategy, MathsStrategy)
        self.assertEqual(strategy.tools, ("unit_converter",))
        self.assertIsInstance(create_strategy("maths for beginners"), MathsStrategy)

    def test_tools_hook_is_empty_by_default(self):
        """Concrete subjects expose `tools` so per-subject tools can plug in."""
        self.assertEqual(EnglishStrategy().tools, ())
        self.assertEqual(CybersecurityStrategy().tools, ())


class DirectiveTests(unittest.TestCase):
    """The profile is rendered into the prompts the AI nodes send."""

    def test_planner_directive_contains_the_search_guidance(self):
        text = create_strategy("cybersecurity").planner_directive()

        self.assertIn("Subject: cybersecurity", text)
        self.assertIn("Query guidance:", text)
        self.assertIn("CVE-", text)
        self.assertIn("nvd.nist.gov", text)
        self.assertIn("Avoid results from", text)
        self.assertIn("recent sources", text)

    def test_answer_directive_contains_rules_and_features(self):
        text = create_strategy("english").answer_directive()

        self.assertIn("Preferred source types", text)
        self.assertIn("example sentences", text)
        self.assertIn("Verify before finalising", text)
        self.assertIn("learner dictionary", text)

    def test_general_strategy_is_neutral(self):
        general = create_strategy("general")

        self.assertIn("Subject: general", general.planner_directive())
        self.assertEqual(general.answer_directive(), "")


class FakeLLM:
    """Answers the planner/research prompts and records every prompt."""

    def __init__(self, classify_reply='{"subject": "english", "intent": "learn"}'):
        self.classify_reply = classify_reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        if "analyzing a research request" in prompt:
            return SimpleNamespace(content=self.classify_reply)

        if "planning web searches" in prompt:
            return SimpleNamespace(content='["english exercises"]')

        return SimpleNamespace(content="Answer based on sources [1].")

    def prompt_containing(self, text: str) -> str:
        return next(prompt for prompt in self.prompts if text in prompt)


class FakeSearch:
    """Returns one stable source so the real merger/formatter run."""

    def search(self, query):
        return [
            {
                "title": "English exercises",
                "url": "https://example.edu/exercises",
                "snippet": "Practice makes perfect.",
            }
        ]


def build_app(llm, classify=True):
    """Build the real graph with fake LLM, search and fetcher (no network)."""
    with mock.patch.object(graph, "create_llm", return_value=llm), mock.patch.object(
        graph, "create_search", return_value=FakeSearch()
    ), mock.patch.object(graph, "fetch_evidence", return_value=[]):
        return graph.create_graph(
            provider="ollama",
            model="fake",
            classify=classify,
        )


class GraphTests(unittest.TestCase):
    """Both AI nodes receive the strategy of the classified subject."""

    def test_plan_node_gets_the_subject_search_guidance(self):
        llm = FakeLLM(
            classify_reply='{"subject": "cybersecurity", "intent": "research"}'
        )

        build_app(llm).invoke({"question": "how do I patch a CVE?"})

        prompt = llm.prompt_containing("planning web searches")
        self.assertIn("Subject: cybersecurity", prompt)
        self.assertIn("CVE-", prompt)

    def test_research_node_gets_the_subject_rules(self):
        llm = FakeLLM()

        build_app(llm).invoke({"question": "teach me present perfect"})

        prompt = llm.prompt_containing("Web sources:")
        self.assertIn("For this subject (english)", prompt)
        self.assertIn("learner dictionary", prompt)

    def test_disabled_classifier_uses_the_general_strategy(self):
        llm = FakeLLM()

        build_app(llm, classify=False).invoke({"question": "teach me present perfect"})

        plan = llm.prompt_containing("planning web searches")
        self.assertIn("Subject: general", plan)

        research = llm.prompt_containing("Web sources:")
        self.assertNotIn("For this subject", research)


class BuildPromptTests(unittest.TestCase):
    """build_prompt itself renders the strategy directive."""

    def test_directive_is_appended_when_sources_exist(self):
        state = State(
            question="what is phishing?",
            subject="cybersecurity",
            context="[1] https://cisa.gov/news",
        )

        prompt = graph.build_prompt(state)

        self.assertIn("For this subject (cybersecurity)", prompt)
        self.assertIn("official advisories", prompt)

    def test_general_subject_adds_nothing_extra(self):
        state = State(
            question="hello",
            subject="general",
            context="[1] https://example.com",
        )

        prompt = graph.build_prompt(state)

        self.assertNotIn("For this subject", prompt)


if __name__ == "__main__":
    unittest.main()

