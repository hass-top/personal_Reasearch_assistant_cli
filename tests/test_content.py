"""Tests for ``search.content``: reading the real text of the best pages."""

import unittest
from unittest import mock

from search import content


class FakeResponse:
    def __init__(self, text="", content_type="text/html; charset=utf-8"):
        self.text = text
        self.headers = {"Content-Type": content_type}

    def raise_for_status(self):
        pass


PAGE = """
<html><head><title>Ignored</title><style>body{color:red}</style></head>
<body><nav>Home About</nav><h1>Present perfect</h1>
<p>It is used for actions that happened in the past and are still true, such as
"I have lived here for three years".</p>
<p>It is formed with have or has followed by the past participle of the verb,
and it is used with a period of time rather than a finished time.</p>
<script>console.log('x')</script>
<footer>copyright</footer></body></html>
"""


class HtmlToTextTests(unittest.TestCase):
    """Only the visible text of a page is kept."""

    def test_scripts_styles_and_navigation_are_dropped(self):
        text = content.html_to_text(PAGE)

        self.assertIn("Present perfect", text)
        self.assertIn("still true", text)
        self.assertNotIn("console.log", text)
        self.assertNotIn("color:red", text)
        self.assertNotIn("Home About", text)

    def test_malformed_html_still_yields_what_was_readable(self):
        text = content.html_to_text("<p>first part<p>second part</div></p>")

        self.assertIn("first part", text)

    def test_entities_are_decoded(self):
        self.assertIn("&", content.html_to_text("<p>A &amp; B</p>"))


class FetchUrlTests(unittest.TestCase):
    """A download failure is a skip, never a crash."""

    def test_successful_download_is_cleaned(self):
        with mock.patch.object(
            content.requests, "get", return_value=FakeResponse(PAGE)
        ):
            self.assertIn("Present perfect", content.fetch_url("https://x.example"))

    def test_network_error_returns_empty_text(self):
        with mock.patch.object(
            content.requests, "get", side_effect=OSError("connection refused")
        ):
            self.assertEqual(content.fetch_url("https://x.example"), "")

    def test_a_timeout_is_passed_to_requests(self):
        with mock.patch.object(content.requests, "get", return_value=FakeResponse(PAGE)) as get:
            content.fetch_url("https://x.example", timeout=2)

        self.assertEqual(get.call_args.kwargs["timeout"], 2)

    def test_non_text_responses_are_skipped(self):
        with mock.patch.object(
            content.requests,
            "get",
            return_value=FakeResponse("%PDF-1.4", "application/pdf"),
        ):
            self.assertEqual(content.fetch_url("https://x.example/file.pdf"), "")


class FetchEvidenceTests(unittest.TestCase):
    """Only the best pages are read, and thin pages are left out."""

    def sources(self, count=3):
        return [
            {
                "title": f"Source {index}",
                "url": f"https://example{index}.com",
                "snippet": "short teaser",
            }
            for index in range(count)
        ]

    def test_only_the_best_sources_are_downloaded(self):
        with mock.patch.object(
            content, "fetch_url", return_value="A" * 500
        ) as fetch:
            evidence = content.fetch_evidence(self.sources(5), limit=2)

        self.assertEqual(len(evidence), 2)
        self.assertEqual(fetch.call_count, 2)

    def test_text_is_capped(self):
        with mock.patch.object(content, "fetch_url", return_value="A" * 5000):
            evidence = content.fetch_evidence(self.sources(1), max_chars=100)

        self.assertEqual(len(evidence[0]["text"]), 100)

    def test_raw_content_is_used_without_downloading(self):
        sources = self.sources(1)
        sources[0]["raw_content"] = "B" * 500

        with mock.patch.object(content, "fetch_url") as fetch:
            evidence = content.fetch_evidence(sources)

        fetch.assert_not_called()
        self.assertTrue(evidence[0]["text"].startswith("B"))

    def test_raw_html_is_cleaned(self):
        sources = self.sources(1)
        sources[0]["raw_content"] = PAGE

        evidence = content.fetch_evidence(sources)

        self.assertIn("Present perfect", evidence[0]["text"])
        self.assertNotIn("<p>", evidence[0]["text"])

    def test_a_page_with_no_real_text_is_skipped(self):
        with mock.patch.object(content, "fetch_url", return_value="hi"):
            self.assertEqual(content.fetch_evidence(self.sources(2)), [])

    def test_a_broken_page_is_skipped_and_the_others_kept(self):
        with mock.patch.object(
            content, "fetch_url", side_effect=["", "C" * 500, ""]
        ):
            evidence = content.fetch_evidence(self.sources(3))

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["url"], "https://example1.com")

    def test_quality_score_is_attached_when_ranking_is_on(self):
        with mock.patch.object(content, "fetch_url", return_value="D" * 500), mock.patch.object(
            content, "ranking_enabled", return_value=True
        ):
            evidence = content.fetch_evidence(self.sources(1))

        self.assertIn("/100", evidence[0]["quality"])

    def test_no_sources_means_no_work(self):
        with mock.patch.object(content, "fetch_url") as fetch:
            self.assertEqual(content.fetch_evidence([]), [])

        fetch.assert_not_called()


if __name__ == "__main__":
    unittest.main()