"""Tests for Phase 1: query understanding (subject/topic/intent/task +
ambiguity) and its routing into the planner or the clarify node."""

import unittest
from types import SimpleNamespace
from unittest import mock

import config
import graph
import main
from graph import (
    build_prompt,
    format_clarification,
    format_evidence,
    normalise_understanding,
    parse_classification,
    parse_understanding,
    route_after_understanding,
    structured_understanding,
    task_directive,
)
from rich.panel import Panel
from state import State, Understanding


def make_streaming_app(result):
    """A stand-in for the compiled graph, driven through ``app.stream``.

    ``run_graph`` iterates ``app.stream(..., stream_mode="updates")`` and
    merges every node's patch into one final dict, so each call must return an
    *iterable of updates*, and one update carrying the whole result reproduces
    the real thing: the accumulated ``final`` is the last node's full state.

    The nesting is easy to get wrong. A bare ``{"research": result}`` is a
    dict, and iterating a dict yields its keys - so the streaming loop would
    receive the string ``"research"`` and die on ``.items()``.

    A list of results means a list of *runs*, which is what the clarification
    tests need: the first run comes back ambiguous, the second is the answer
    for whichever reading the user picked.
    """
    app = mock.Mock()

    if isinstance(result, list):
        app.stream.side_effect = [[{"research": one}] for one in result]
    else:
        app.stream.return_value = [{"research": result}]

    return app


DEFAULT_UNDERSTANDING = (
    '{"subject": "english", "topic": "present perfect", '
    '"intent": "learn", "task": "explain", '
    '"ambiguous": false, "clarification": "", '
    '"clarification_options": []}'
)

AMBIGUOUS_UNDERSTANDING = (
    '{"subject": "english", "topic": "scorpion", '
    '"intent": "learn", "task": "word meaning", '
    '"ambiguous": true, '
    '"clarification": "Do you mean the animal, or words made from its letters?", '
    '"clarification_options": ['
    '"Meaning and facts about the scorpion animal", '
    '"Words that can be made from the letters of scorpion"]}'
)


class UnderstandingParseTests(unittest.TestCase):
    """parse_understanding must extract every field of the JSON contract."""

    def test_full_json_is_extracted(self):
        data = parse_understanding(AMBIGUOUS_UNDERSTANDING)

        self.assertEqual(
            data,
            {
                "subject": "english",
                "topic": "scorpion",
                "intent": "learn",
                "task": "word meaning",
                "ambiguous": True,
                "clarification": (
                    "Do you mean the animal, or words made from its letters?"
                ),
                "clarification_options": [
                    "Meaning and facts about the scorpion animal",
                    "Words that can be made from the letters of scorpion",
                ],
            },
        )

    def test_missing_fields_get_safe_defaults(self):
        data = parse_understanding('{"subject": "english", "intent": "learn"}')

        self.assertEqual(data["topic"], "")
        self.assertEqual(data["task"], "")
        self.assertIs(data["ambiguous"], False)
        self.assertEqual(data["clarification"], "")
        self.assertEqual(data["clarification_options"], [])

    def test_options_are_capped_cleaned_and_filtered(self):
        reply = (
            '{"ambiguous": true, "clarification": "Which one?", '
            '"clarification_options": [" first ", "", 5, null, '
            '"second", "third", "fourth", "fifth"]}'
        )

        data = parse_understanding(reply)

        self.assertEqual(
            data["clarification_options"],
            ["first", "second", "third", "fourth"],
        )

    def test_single_string_option_is_wrapped_into_a_list(self):
        reply = '{"ambiguous": true, "clarification_options": "One reading"}'

        self.assertEqual(
            parse_understanding(reply)["clarification_options"],
            ["One reading"],
        )

    def test_options_are_salvaged_from_broken_json(self):
        reply = (
            'Result: {"ambiguous": true, "clarification": "Which one?", '
            '"clarification_options": ["about the animal", "word play"],}'
        )

        data = parse_understanding(reply)

        self.assertIs(data["ambiguous"], True)
        self.assertEqual(
            data["clarification_options"],
            ["about the animal", "word play"],
        )

    def test_string_ambiguous_is_coerced_to_bool(self):
        reply = '{"ambiguous": "true", "clarification": "Which one?"}'

        self.assertIs(parse_understanding(reply)["ambiguous"], True)

        reply = '{"ambiguous": "false", "clarification": "Which one?"}'

        self.assertIs(parse_understanding(reply)["ambiguous"], False)

    def test_broken_json_is_salvaged_field_by_field(self):
        reply = (
            'Result: {"subject": "english", "topic": " phrasal verbs ", '
            '"intent": "practice", "task": "exercise", '
            '"ambiguous": true, "clarification": "Which verbs?",}'
        )

        data = parse_understanding(reply)

        self.assertEqual(data["topic"], "phrasal verbs")
        self.assertEqual(data["task"], "exercise")
        self.assertIs(data["ambiguous"], True)

    def test_garbage_returns_the_safe_defaults(self):
        data = parse_understanding("no idea")

        self.assertEqual(
            data,
            {
                "subject": "general",
                "topic": "",
                "intent": "research",
                "task": "",
                "ambiguous": False,
                "clarification": "",
                "clarification_options": [],
            },
        )


class ClarificationFormatTests(unittest.TestCase):
    """The clarify node renders the question plus numbered readings."""

    def test_options_are_numbered_under_the_question(self):
        text = format_clarification(
            "Do you mean the animal?",
            ["Meaning of the animal", "Word play"],
        )

        self.assertIn("**Clarification needed**", text)
        self.assertIn("Do you mean the animal?", text)
        self.assertIn("1. Meaning of the animal", text)
        self.assertIn("2. Word play", text)
        self.assertIn("Reply with the number", text)

    def test_without_options_the_open_question_is_used(self):
        text = format_clarification("", [])

        self.assertIn("Could you add more detail", text)
        self.assertNotIn("1. ", text)
        self.assertNotIn("Reply with the number", text)


class RoutingTests(unittest.TestCase):
    """understand routes to clarify only when the request is ambiguous."""

    def test_clear_request_goes_to_the_planner(self):
        state = State(question="teach me present perfect")

        self.assertEqual(route_after_understanding(state), "plan")

    def test_ambiguous_request_goes_to_clarify(self):
        state = State(question="scorpion", ambiguous=True)

        self.assertEqual(route_after_understanding(state), "clarify")


class ConfigTests(unittest.TestCase):
    """The vocabulary lives in config.py and matches the Phase 1 spec."""

    def test_only_two_subjects_for_now(self):
        self.assertEqual(config.SUBJECTS, ("english", "cybersecurity"))

    def test_four_intents(self):
        self.assertEqual(
            config.INTENTS,
            ("learn", "practice", "verify", "research"),
        )

    def test_defaults(self):
        self.assertEqual(config.DEFAULT_SUBJECT, "general")
        self.assertEqual(config.DEFAULT_INTENT, "research")

    def test_classify_flag_is_a_bool(self):
        self.assertIsInstance(config.CLASSIFY_ENABLED, bool)


class FakeLLM:
    """Answers the understanding/planner/research prompts with canned replies."""

    def __init__(
        self,
        understand_reply=DEFAULT_UNDERSTANDING,
        fail_understand=False,
    ):
        self.understand_reply = understand_reply
        self.fail_understand = fail_understand
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)

        if "analyzing a research request" in prompt:
            if self.fail_understand:
                raise RuntimeError("llm is down")
            return SimpleNamespace(content=self.understand_reply)

        if "planning web searches" in prompt:
            return SimpleNamespace(content='["english exercises"]')

        return SimpleNamespace(content="Answer based on sources [1].")


class FakeSearch:
    """Returns one stable source so the real merger/formatter run."""

    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [
            {
                "title": "English exercises",
                "url": "https://example.edu/exercises",
                "snippet": "Practice makes perfect.",
            }
        ]


FAKE_EVIDENCE = [
    {
        "title": "English exercises",
        "url": "https://example.edu/exercises",
        "text": (
            "Fill in the blank: I ___ (go) to school every day. "
            "He ___ (play) tennis on Sundays with his friends."
        ),
        "quality": "60/100 (educational)",
    }
]


class PatchedApp:
    """The real graph, run with the fakes still in place.

    ``create_llm`` / ``create_search`` are called while the graph is built, but
    ``fetch_evidence`` is looked up when the evidence node *runs*, so the patch
    has to still be active at ``invoke`` time. The patches are therefore
    applied around the invocation instead of around the construction.

    ``evidence`` can be overridden per call. That exists because of a trap: a
    test cannot wrap ``invoke`` in its own ``patch.object(graph,
    "fetch_evidence", ...)``. The patch below is entered *inside* that one, so
    it wins, and the test's ``side_effect`` is discarded in silence - the
    assertion then fails on data the test never actually produced. Passing the
    fetcher in removes the nesting entirely.
    """

    #: Distinguishes "no override given" from "override with None".
    _KEEP = object()

    def __init__(self, llm, classify, search, evidence):
        self._llm = llm
        self._classify = classify
        self._search = search
        self._evidence = evidence

    def invoke(self, state, evidence=_KEEP):
        fetcher = self._evidence if evidence is self._KEEP else evidence
        # A bare list is the common case and is what build_app stores, but
        # patching it in with new= would make the list itself the function. Only
        # a Mock is passed through untouched, so a test can keep asserting on it.
        if not isinstance(fetcher, mock.Mock):
            fetcher = mock.Mock(return_value=fetcher)
        with mock.patch.object(
            graph, "create_llm", return_value=self._llm
        ), mock.patch.object(
            graph, "create_search", return_value=self._search
        ), mock.patch.object(
            graph, "fetch_evidence", new=fetcher
        ):
            return graph.create_graph(
                provider="ollama",
                model="fake",
                classify=self._classify,
            ).invoke(state)


def build_app(llm, classify=True, search=None, evidence=None):
    """Build the real graph with fake LLM, search and fetcher (no network)."""
    return PatchedApp(
        llm,
        classify,
        search if search is not None else FakeSearch(),
        FAKE_EVIDENCE if evidence is None else evidence,
    )


class GraphTests(unittest.TestCase):
    """The understand node runs first and always leaves valid labels."""

    def test_full_run_fills_subject_and_intent(self):
        llm = FakeLLM()

        state = build_app(llm).invoke(
            {"question": "teach me present perfect"}
        )

        self.assertEqual(state["subject"], "english")
        self.assertEqual(state["intent"], "learn")
        self.assertEqual(state["topic"], "present perfect")
        self.assertEqual(state["task"], "explain")
        self.assertFalse(state["ambiguous"])
        self.assertTrue(state["result"])
        self.assertTrue(state["sources"])
        # understand runs before plan: it owns the first LLM call
        self.assertIn("analyzing a research request", llm.prompts[0])

    def test_disabled_classifier_skips_the_call(self):
        llm = FakeLLM()

        state = build_app(llm, classify=False).invoke(
            {"question": "teach me present perfect"}
        )

        self.assertEqual(state["subject"], "general")
        self.assertEqual(state["intent"], "research")
        self.assertEqual(state["topic"], "")
        self.assertFalse(state["ambiguous"])
        self.assertFalse(
            any("analyzing a research" in p for p in llm.prompts)
        )
        self.assertTrue(state["result"])

    def test_classifier_failure_falls_back_but_run_continues(self):
        llm = FakeLLM(fail_understand=True)

        state = build_app(llm).invoke(
            {"question": "teach me present perfect"}
        )

        self.assertEqual(
            (state["subject"], state["intent"]),
            ("general", "research"),
        )
        self.assertFalse(state["ambiguous"])
        self.assertTrue(state["result"])

    def test_ambiguous_request_stops_at_clarify_without_searching(self):
        llm = FakeLLM(understand_reply=AMBIGUOUS_UNDERSTANDING)
        search = FakeSearch()

        state = build_app(llm, search=search).invoke(
            {"question": "scorpion"}
        )

        self.assertTrue(state["ambiguous"])
        self.assertIn("Do you mean the animal", state["result"])
        # The readings are rendered as a numbered list the CLI can pick from.
        self.assertIn("1. Meaning and facts about the scorpion animal", state["result"])
        self.assertIn("2. Words that can be made from the letters of scorpion", state["result"])
        # No planner call and no search ran before the clarify stop, so the
        # search-related channels were never even written to.
        self.assertEqual(state.get("sources", []), [])
        self.assertEqual(state.get("queries", []), [])
        self.assertEqual(search.queries, [])
        self.assertFalse(
            any("planning web searches" in p for p in llm.prompts)
        )


class OutputTests(unittest.TestCase):
    """The CLI prints the labels under the answer, or a clarification panel."""

    DEFAULT_ANSWERS = ["1", "1", "1", "teach me english", "q"]

    def invoke_main(self, graph_result, answers=None):
        """Run main(); return the raw print args (``self.app`` is the fake graph)."""
        answer_iter = iter(answers or self.DEFAULT_ANSWERS)
        app = make_streaming_app(graph_result)

        self.app = app

        with mock.patch.object(main, "msvcrt", None), mock.patch.object(
            main, "console", mock.MagicMock()
        ) as console, mock.patch(
            "builtins.input", side_effect=lambda *a: next(answer_iter)
        ), mock.patch.object(main, "create_graph", return_value=app):
            main.main()

        return [
            call.args[0]
            for call in console.print.call_args_list
            if call.args
        ]

    def run_main(self, graph_result, answers=None):
        return [
            str(arg)
            for arg in self.invoke_main(graph_result, answers=answers)
        ]

    def test_subject_intent_line_is_printed(self):
        result = {
            "result": "Answer [1].",
            "queries": ["teach me english"],
            "sources": [
                {
                    "title": "T",
                    "url": "https://example.edu/a",
                    "snippet": "s",
                }
            ],
            "subject": "english",
            "intent": "learn",
        }

        printed = self.run_main(result)

        self.assertTrue(
            any(
                "Subject: english" in text and "Intent: learn" in text
                for text in printed
            ),
            printed,
        )

    def test_line_is_skipped_when_nothing_was_classified(self):
        result = {
            "result": "Answer [1].",
            "queries": [],
            "sources": [],
        }

        printed = self.run_main(result)

        self.assertFalse(any("Subject:" in text for text in printed))

    def test_topic_and_task_are_added_to_the_label_line(self):
        result = {
            "result": "Answer [1].",
            "queries": ["teach me english"],
            "sources": [],
            "subject": "english",
            "intent": "learn",
            "topic": "present perfect",
            "task": "explain",
        }

        printed = self.run_main(result)

        self.assertTrue(
            any(
                "Topic: present perfect" in text and "Task: explain" in text
                for text in printed
            ),
            printed,
        )

    def test_ambiguous_request_prints_the_clarification_panel(self):
        result = {
            "result": "**Clarification needed**\n\nDo you mean the animal?",
            "queries": [],
            "sources": [],
            "ambiguous": True,
            "clarification": "Do you mean the animal?",
            "subject": "english",
            "intent": "learn",
            "topic": "scorpion",
            "task": "word meaning",
        }

        printed = self.invoke_main(result)
        panels = [arg for arg in printed if isinstance(arg, Panel)]
        lines = [str(arg) for arg in printed if isinstance(arg, str)]
        titles = [str(panel.title) for panel in panels]

        self.assertTrue(
            any("Clarification Needed" in title for title in titles),
            titles,
        )
        self.assertFalse(
            any("Research Result" in title for title in titles),
            titles,
        )
        self.assertTrue(
            any(
                "Do you mean the animal"
                in str(getattr(panel.renderable, "markup", panel.renderable))
                for panel in panels
            ),
            printed,
        )
        # The graph stopped before searching: the "no sources" warning
        # (which would be misleading here) must not appear.
        self.assertFalse(
            any("No web sources found" in line for line in lines),
            lines,
        )
        self.assertTrue(
            any("Topic: scorpion" in line and "Task: word meaning" in line
                for line in lines),
            lines,
        )

    @staticmethod
    def ambiguous_result():
        """A clarify-node result with two numbered readings."""
        return {
            "result": (
                "**Clarification needed**\n\n"
                "Do you mean the animal?\n"
                "1. Meaning of the scorpion animal\n"
                "2. Words made from scorpion letters\n\n"
                "Reply with the number of the option you mean."
            ),
            "ambiguous": True,
            "clarification": "Do you mean the animal?",
            "clarification_options": [
                "Meaning of the scorpion animal",
                "Words made from scorpion letters",
            ],
            "subject": "english",
            "intent": "learn",
            "topic": "scorpion",
            "task": "word meaning",
        }

    @staticmethod
    def answered_result():
        """A normal research result for the second invoke."""
        return {
            "result": "The scorpion is an arachnid [1].",
            "ambiguous": False,
            "queries": ["meaning of the scorpion animal"],
            "sources": [
                {
                    "title": "Scorpion",
                    "url": "https://example.edu/scorpion",
                    "snippet": "An arachnid.",
                }
            ],
            "subject": "english",
            "intent": "learn",
            "topic": "scorpion",
            "task": "explain",
        }

    def test_picking_an_option_reinvokes_the_graph_with_that_reading(self):
        printed = self.invoke_main(
            [self.ambiguous_result(), self.answered_result()],
            answers=["1", "1", "1", "scorpion", "1", "q"],
        )
        lines = [str(arg) for arg in printed if isinstance(arg, str)]
        panels = [arg for arg in printed if isinstance(arg, Panel)]
        titles = [str(panel.title) for panel in panels]

        # The numbered menu is shown before the choice is asked for.
        self.assertTrue(
            any("Which one do you mean?" in line for line in lines),
            lines,
        )
        self.assertTrue(
            any("[1]  Meaning of the scorpion animal" in line for line in lines),
            lines,
        )

        # The graph runs a second time with the chosen reading as question.
        self.assertEqual(self.app.stream.call_count, 2)
        self.app.stream.assert_any_call(
            {"question": "Meaning of the scorpion animal"},
            stream_mode="updates",
        )

        # The final answer is shown as a normal research result.
        self.assertTrue(
            any("Research Result" in title for title in titles),
            titles,
        )

    def test_going_back_skips_the_second_run(self):
        printed = self.invoke_main(
            [self.ambiguous_result()],
            answers=["1", "1", "1", "scorpion", "b", "q"],
        )
        lines = [str(arg) for arg in printed if isinstance(arg, str)]

        self.assertEqual(self.app.stream.call_count, 1)
        self.assertFalse(any("Research Result" in line for line in lines))
        # The "no sources" warning stays hidden: nothing was searched.
        self.assertFalse(any("No web sources found" in line for line in lines))

    def test_quit_at_the_pick_quits_the_application(self):
        self.invoke_main(
            [self.ambiguous_result()],
            answers=["1", "1", "1", "scorpion", "q"],
        )

        self.assertEqual(self.app.stream.call_count, 1)

    def test_out_of_range_number_goes_back_without_a_second_run(self):
        printed = self.invoke_main(
            [self.ambiguous_result()],
            answers=["1", "1", "1", "scorpion", "9", "q"],
        )
        lines = [str(arg) for arg in printed if isinstance(arg, str)]

        self.assertEqual(self.app.stream.call_count, 1)
        self.assertTrue(
            any("Unknown option" in line for line in lines),
            lines,
        )

    def test_still_ambiguous_result_stops_after_the_single_round(self):
        printed = self.invoke_main(
            [self.ambiguous_result(), self.ambiguous_result()],
            answers=["1", "1", "1", "scorpion", "1", "q"],
        )
        panels = [arg for arg in printed if isinstance(arg, Panel)]
        titles = [str(panel.title) for panel in panels]

        self.assertEqual(self.app.stream.call_count, 2)
        self.assertTrue(
            any("Clarification Needed" in title for title in titles),
            titles,
        )
        self.assertFalse(
            any("Research Result" in title for title in titles),
            titles,
        )



class FakeStructuredLLM(FakeLLM):
    """A provider exposing the raw chat model, like the real wrappers do.

    ``graph.structured_understanding`` goes through ``llm.llm`` and
    ``with_structured_output``, so a fake without that attribute exercises the
    fallback path and this one exercises the schema path.
    """

    def __init__(self, structured_reply=None, structured_error=None, **kwargs):
        super().__init__(**kwargs)
        self.structured_reply = structured_reply
        self.structured_error = structured_error
        self.structured_prompts = []
        self.structured_schema = None

        outer = self

        class _Bound:
            def invoke(inner_self, prompt):
                outer.structured_prompts.append(prompt)
                if outer.structured_error is not None:
                    raise outer.structured_error

                return outer.structured_reply

        class _Chat:
            def with_structured_output(inner_self, schema):
                outer.structured_schema = schema
                return _Bound()

        self.llm = _Chat()


class FakeProseThenJSONLLM(FakeLLM):
    """Ignores the JSON instruction once, then answers properly on the retry."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.understand_calls = 0

    def invoke(self, prompt):
        self.prompts.append(prompt)

        if "not valid JSON" in prompt:
            return SimpleNamespace(content=self.understand_reply)

        if "analyzing a research request" in prompt:
            self.understand_calls += 1
            if self.understand_calls == 1:
                return SimpleNamespace(
                    content="Sure! Let me think about that question first."
                )

            return SimpleNamespace(content=self.understand_reply)

        return super().invoke(prompt)


class FakeProseLLM(FakeLLM):
    """Never answers with JSON, not even on the repair round-trip."""

    def invoke(self, prompt):
        self.prompts.append(prompt)

        if "analyzing a research request" in prompt or "not valid JSON" in prompt:
            return SimpleNamespace(content="I would rather not say.")

        return super().invoke(prompt)


class NormaliseTests(unittest.TestCase):
    """A well-formed object is still snapped onto the vocabulary."""

    def test_valid_object_passes_through(self):
        data = normalise_understanding(
            {
                "subject": "english",
                "topic": "present perfect",
                "intent": "learn",
                "task": "explain",
                "ambiguous": False,
            }
        )

        self.assertEqual(data["subject"], "english")
        self.assertEqual(data["intent"], "learn")

    def test_out_of_vocabulary_subject_is_snapped(self):
        # Structured output guarantees the *shape*, not the vocabulary.
        data = normalise_understanding({"subject": "mathematics"})

        self.assertEqual(data["subject"], "general")
        self.assertEqual(data["intent"], "research")

    def test_string_ambiguous_and_options_are_coerced(self):
        data = normalise_understanding(
            {"ambiguous": "true", "clarification_options": "just one reading"}
        )

        self.assertIs(data["ambiguous"], True)
        self.assertEqual(data["clarification_options"], ["just one reading"])

    def test_garbage_payload_returns_the_defaults(self):
        self.assertEqual(
            normalise_understanding(None), graph._default_understanding()
        )
        self.assertEqual(
            normalise_understanding("nope"), graph._default_understanding()
        )





class StructuredOutputTests(unittest.TestCase):
    """Schema first, then the JSON parser, then one repair, then defaults."""

    def test_schema_reply_is_used_without_parsing(self):
        llm = FakeStructuredLLM(
            structured_reply=Understanding(
                subject="english",
                topic="present perfect",
                intent="learn",
                task="explain",
            )
        )

        state = build_app(llm).invoke({"question": "teach me present perfect"})

        self.assertIs(llm.structured_schema, Understanding)
        self.assertEqual(state["subject"], "english")
        self.assertEqual(state["topic"], "present perfect")
        # The schema answered, so the JSON-in-text prompt was never used.
        self.assertFalse(
            any("analyzing a research request" in p for p in llm.prompts)
        )

    def test_unsupported_structured_output_falls_back_to_json(self):
        # A plain FakeLLM has no `.llm`, which is what a provider without tool
        # calling looks like: the text parser has to take over.
        llm = FakeLLM()

        state = build_app(llm).invoke({"question": "teach me present perfect"})

        self.assertEqual(state["subject"], "english")
        self.assertEqual(state["intent"], "learn")

    def test_failing_structured_call_falls_back_to_json(self):
        llm = FakeStructuredLLM(structured_error=RuntimeError("no tools here"))

        state = build_app(llm).invoke({"question": "teach me present perfect"})

        self.assertEqual(state["subject"], "english")
        self.assertTrue(state["result"])

    def test_prose_reply_is_repaired_with_one_retry(self):
        llm = FakeProseThenJSONLLM()

        state = build_app(llm).invoke({"question": "teach me present perfect"})

        # One understanding call that answered with prose, one repair call.
        repairs = [p for p in llm.prompts if "not valid JSON" in p]
        self.assertEqual(len(repairs), 1, llm.prompts)
        self.assertEqual(state["subject"], "english")

    def test_hopeless_reply_falls_back_to_the_defaults(self):
        state = build_app(FakeProseLLM()).invoke(
            {"question": "teach me present perfect"}
        )

        self.assertEqual(
            (state["subject"], state["intent"]), ("general", "research")
        )
        # The run still finishes.
        self.assertTrue(state["result"])

    def test_helper_returns_none_without_a_chat_model(self):
        self.assertIsNone(structured_understanding(FakeLLM(), "prompt"))


class PlannerFocusTests(unittest.TestCase):
    """The planner now sees topic, intent and task, not only the question."""

    def focus(self, **kwargs):
        return graph.planner_focus(State(question="teach me present perfect", **kwargs))

    def test_topic_intent_and_task_reach_the_planner(self):
        block = self.focus(
            topic="present perfect", intent="practice", task="exercises"
        )

        self.assertIn("Topic: present perfect", block)
        self.assertIn("Intent: practice", block)
        self.assertIn("Task: exercises", block)

    def test_practice_intent_asks_for_exercise_pages(self):
        block = self.focus(intent="practice")

        self.assertIn("exercises", block.lower())
        self.assertIn("answer keys", block.lower())

    def test_learn_and_verify_intent_differ(self):
        learn = self.focus(intent="learn")
        verify = self.focus(intent="verify")

        self.assertIn("tutorial", learn.lower())
        self.assertIn("authoritative", verify.lower())

    def test_no_understanding_means_no_focus_block(self):
        self.assertEqual(self.focus(), "")

    def test_the_block_is_part_of_the_planner_prompt(self):
        llm = FakeLLM(
            understand_reply=(
                '{"subject": "english", "topic": "present perfect", '
                '"intent": "practice", "task": "exercises", "ambiguous": false}'
            )
        )

        build_app(llm).invoke(
            {"question": "give me present perfect exercises to practise"}
        )

        prompt = next(p for p in llm.prompts if "planning web searches" in p)
        # The understanding reply said `practice`, so the planner must have
        # been told to look for exercise pages instead of tutorials.
        self.assertIn("What the user actually asked for:", prompt)
        self.assertIn("Intent: practice", prompt)
        self.assertIn("answer keys", prompt)


class TaskDirectiveTests(unittest.TestCase):
    """The free-text task becomes one structural instruction."""

    def test_compare_asks_for_a_comparison(self):
        self.assertIn("point by point", task_directive("compare the two"))

    def test_meaning_asks_for_a_definition_first(self):
        self.assertIn("definition", task_directive("word meaning").lower())

    def test_solve_asks_for_the_steps(self):
        self.assertIn("steps", task_directive("solve this equation").lower())

    def test_summary_asks_for_conciseness(self):
        self.assertIn("concise", task_directive("summarise scorpions").lower())

    def test_unrelated_task_adds_nothing(self):
        self.assertEqual(task_directive("explain"), "")


class BuildPromptTests(unittest.TestCase):
    """The research prompt uses the evidence and the understanding fields."""

    def state(self, **kwargs):
        return State(
            question="teach me present perfect",
            context="[1] Exercises\nhttps://ex.com\nPractice makes perfect.",
            **kwargs,
        )

    def test_evidence_replaces_the_snippets(self):
        prompt = build_prompt(
            self.state(
                evidence=[{"title": "T", "url": "https://ex.com", "text": "Real page text."}]
            )
        )

        self.assertIn("Source excerpts:", prompt)
        self.assertIn("Real page text.", prompt)
        self.assertNotIn("Web sources:", prompt)

    def test_evidence_prompt_demands_per_claim_citations(self):
        prompt = build_prompt(
            self.state(
                evidence=[{"title": "T", "url": "https://ex.com", "text": "Real page text."}]
            )
        )

        self.assertIn("ground every claim", prompt)
        self.assertIn("instead of filling", prompt)

    def test_without_evidence_it_falls_back_to_the_snippets(self):
        prompt = build_prompt(self.state())

        self.assertIn("Web sources:", prompt)
        self.assertIn("Practice makes perfect.", prompt)
        self.assertNotIn("Source excerpts:", prompt)

    def test_practice_intent_asks_for_exercises_and_an_answer_key(self):
        prompt = build_prompt(self.state(intent="practice"))

        self.assertIn("3 to 5 exercises", prompt)
        self.assertIn("answer key", prompt)

    def test_verify_intent_asks_for_the_verdict_first(self):
        self.assertIn("verdict first", build_prompt(self.state(intent="verify")))

    def test_topic_and_task_are_carried_into_the_prompt(self):
        prompt = build_prompt(
            self.state(intent="learn", topic="present perfect", task="compare")
        )

        self.assertIn("The question is about present perfect.", prompt)
        self.assertIn("point by point", prompt)

    def test_no_sources_at_all_still_carries_the_intent(self):
        prompt = build_prompt(State(question="q", intent="practice"))

        self.assertIn("not backed by sources", prompt)
        self.assertIn("3 to 5 exercises", prompt)

    def test_evidence_renders_the_same_numbered_block(self):
        block = format_evidence(
            [
                {
                    "title": "First",
                    "url": "https://a.example",
                    "text": "one",
                    "quality": "60/100 (educational)",
                },
                {"title": "Second", "url": "https://b.example", "text": "two"},
            ]
        )

        self.assertIn("[1] First", block)
        self.assertIn("[2] Second", block)
        self.assertIn("Quality: 60/100 (educational)", block)

    def test_empty_evidence_renders_nothing(self):
        self.assertEqual(format_evidence([]), "")


class EvidenceNodeTests(unittest.TestCase):
    """The evidence node reads the pages between search and research."""

    def test_pages_are_read_into_the_state(self):
        llm = FakeLLM()

        state = build_app(llm).invoke({"question": "teach me present perfect"})

        self.assertEqual(state["evidence"], FAKE_EVIDENCE)

    def test_research_prompt_uses_the_fetched_text(self):
        llm = FakeLLM()

        build_app(llm).invoke({"question": "teach me present perfect"})

        research_prompt = llm.prompts[-1]
        self.assertIn("Source excerpts:", research_prompt)
        self.assertIn("Fill in the blank", research_prompt)

    def test_no_readable_page_falls_back_to_the_snippets(self):
        llm = FakeLLM()

        state = build_app(llm, evidence=[]).invoke(
            {"question": "teach me present perfect"}
        )

        self.assertEqual(state["evidence"], [])
        self.assertIn("Web sources:", llm.prompts[-1])
        self.assertTrue(state["result"])

    def test_evidence_node_never_raises(self):
        llm = FakeLLM()
        app = build_app(llm)

        # The fetcher blowing up (DNS, firewall, ...) must not kill the run.
        fetch = mock.Mock(side_effect=RuntimeError("network is down"))
        state = app.invoke({"question": "teach me present perfect"}, evidence=fetch)

        # Asserting the call matters as much as the empty evidence: it proves
        # the fetcher really was the failing one, not the default fake.
        self.assertTrue(fetch.called)
        self.assertEqual(state["evidence"], [])
        self.assertTrue(state["result"])

    def test_clarify_still_runs_before_any_fetch(self):
        llm = FakeLLM(understand_reply=AMBIGUOUS_UNDERSTANDING)

        fetch = mock.Mock(return_value=FAKE_EVIDENCE)
        state = build_app(llm).invoke({"question": "scorpion"}, evidence=fetch)

        fetch.assert_not_called()
        self.assertEqual(state.get("evidence", []), [])


if __name__ == "__main__":
    unittest.main()

    def test_compare_asks_for_a_comparison(self):
        self.assertIn("point by point", task_directive("compare the two"))

    def test_meaning_asks_for_a_definition_first(self):
        self.assertIn("definition", task_directive("word meaning").lower())

    def test_solve_asks_for_the_steps(self):
        self.assertIn("steps", task_directive("solve this equation").lower())

    def test_unrelated_task_adds_nothing(self):
        self.assertEqual(task_directive("explain"), "")




