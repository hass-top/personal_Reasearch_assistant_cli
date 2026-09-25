"""Tests for the interactive main menu: loop, parameters, Ctrl+Q quit."""

import unittest
from unittest import mock

import config
import main
from main import (
    Settings,
    is_quit_like,
    parse_menu_key,
    parse_on_off,
    settings_summary,
)

GRAPH_RESULT = {
    "result": "Because Rayleigh scattering scatters blue light [1].",
    "queries": ["rayleigh scattering sky blue"],
    "sources": [
        {
            "title": "NASA - Why is the sky blue?",
            "url": "https://www.nasa.gov/ask/sky-blue",
            "snippet": "Sunlight scatters off air molecules.",
            "tier": "government",
            "score": 100,
            "quality": "NASA / NOAA / government",
        }
    ],
}


class QuitTests(unittest.TestCase):
    """q / quit / exit and the Ctrl+Q control character must quit."""

    def test_quit_answers_are_recognised(self):
        answers = ("q", "Q", "  quit ", "Exit", "\x11", "sky blue?\x11")

        for answer in answers:
            with self.subTest(answer=answer):
                self.assertTrue(is_quit_like(answer))

    def test_normal_answers_do_not_quit(self):
        answers = ("", None, "b", "why is the sky blue", "1")

        for answer in answers:
            with self.subTest(answer=answer):
                self.assertFalse(is_quit_like(answer))


class MenuKeyTests(unittest.TestCase):
    """Single keypresses map to menu actions; anything else is ignored."""

    def test_keys_map_to_actions(self):
        cases = {
            "1": "research",
            "\r": "research",
            "\n": "research",
            "2": "parameters",
            "P": "parameters",
            "q": "quit",
            "\x11": "quit",
        }

        for key, action in cases.items():
            with self.subTest(key=key):
                self.assertEqual(parse_menu_key(key), action)

    def test_unknown_keys_are_ignored(self):
        for key in ("", "z", "\xe0H", "  "):
            with self.subTest(key=key):
                self.assertEqual(parse_menu_key(key), "")


class OnOffTests(unittest.TestCase):
    """The research-options prompts accept on/off style answers."""

    def test_truthy_answers(self):
        for answer in ("on", "ON", "yes", "y", "1", "true"):
            with self.subTest(answer=answer):
                self.assertTrue(parse_on_off(answer))

    def test_falsy_answers(self):
        for answer in ("off", "No", "n", "0", "false"):
            with self.subTest(answer=answer):
                self.assertFalse(parse_on_off(answer))

    def test_unknown_answers(self):
        self.assertIsNone(parse_on_off("maybe"))
        self.assertIsNone(parse_on_off(""))


class SettingsSummaryTests(unittest.TestCase):
    """The line above the menu shows every current knob."""

    def test_summary_shows_every_knob(self):
        settings = Settings(
            provider="groq",
            model="llama-3.3-70b-versatile",
            search_provider="tavily",
            max_sources=4,
            planner=False,
        )

        summary = settings_summary(settings)

        self.assertIn("model groq/llama-3.3-70b-versatile", summary)
        self.assertIn("search tavily", summary)
        self.assertIn("sources 4", summary)
        self.assertIn("planner off", summary)

        expected = "on" if config.SOURCE_RANKING_ENABLED else "off"
        self.assertIn(f"ranking {expected}", summary)


class ParameterTests(unittest.TestCase):
    """The Parameters menu edits model / search / research options."""

    def test_choose_model_switches_provider_and_key(self):
        settings = Settings()

        with mock.patch.object(main, "GROQ_API_KEY", None), mock.patch(
            "builtins.input",
            side_effect=["3", "groq-model", "secret"],
        ):
            main.choose_model(settings)

        self.assertEqual(settings.provider, "groq")
        self.assertEqual(settings.model, "groq-model")
        self.assertEqual(settings.api_key, "secret")

    def test_choose_search_switches_backend_and_key(self):
        settings = Settings()

        with mock.patch.object(main, "TAVILY_API_KEY", None), mock.patch(
            "builtins.input",
            side_effect=["2", "tvly-key"],
        ):
            main.choose_search(settings)

        self.assertEqual(settings.search_provider, "tavily")
        self.assertEqual(settings.search_api_key, "tvly-key")

    def test_research_options_change_thresholds(self):
        settings = Settings()

        with mock.patch.object(
            config, "SOURCE_RANKING_ENABLED", True
        ), mock.patch.object(config, "MIN_SOURCE_SCORE", 0), mock.patch(
            "builtins.input",
            side_effect=["4", "on", "off", "50"],
        ):
            main.research_options(settings)

            self.assertFalse(config.SOURCE_RANKING_ENABLED)
            self.assertEqual(config.MIN_SOURCE_SCORE, 50)

        self.assertEqual(settings.max_sources, 4)
        self.assertTrue(settings.planner)


class MenuFlowTests(unittest.TestCase):
    """Drive main() end-to-end with a scripted keyboard (no real terminal)."""

    def run_main(self, inputs, graph_result=None):
        """Run main() feeding ``inputs`` to every prompt; return the fake app."""
        answers = iter(inputs)
        app = None

        if graph_result is not None:
            app = mock.Mock()
            app.invoke.return_value = graph_result

        with mock.patch.object(main, "msvcrt", None), mock.patch.object(
            main, "console", mock.MagicMock()
        ), mock.patch("builtins.input", side_effect=lambda *a: next(answers)):
            if app is None:
                main.main()
            else:
                with mock.patch.object(
                    main, "create_graph", return_value=app
                ) as create:
                    main.main()
                    create.assert_called_once()

        return app

    def test_loop_opens_parameters_then_back_then_quits(self):
        # model, search, menu->parameters, back, menu->quit
        self.run_main(["1", "1", "2", "b", "q"])

    def test_research_returns_to_the_menu_then_quits(self):
        app = self.run_main(
            ["1", "1", "1", "why is the sky blue", "q"],
            graph_result=GRAPH_RESULT,
        )

        app.invoke.assert_called_once_with({"question": "why is the sky blue"})

    def test_ctrl_q_quits_from_the_main_menu(self):
        self.run_main(["1", "1", "\x11"])

    def test_ctrl_q_quits_from_the_question_prompt(self):
        self.run_main(["1", "1", "1", "\x11"])

    def test_unknown_menu_key_is_ignored_then_quit(self):
        self.run_main(["1", "1", "z", "q"])

    def test_parameters_research_options_reach_the_config(self):
        with mock.patch.object(
            config, "SOURCE_RANKING_ENABLED", True
        ), mock.patch.object(config, "MIN_SOURCE_SCORE", 0):
            # model, search, menu->parameters, research options,
            # max sources, planner, ranking, min score, back, quit
            self.run_main(
                ["1", "1", "2", "3", "4", "on", "off", "50", "b", "q"]
            )

            self.assertFalse(config.SOURCE_RANKING_ENABLED)
            self.assertEqual(config.MIN_SOURCE_SCORE, 50)


if __name__ == "__main__":
    unittest.main()
