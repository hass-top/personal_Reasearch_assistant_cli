"""Tests for the timing/token report and the ``stage`` context manager.

The regression these guard: ``summary()`` used to read a ``failed`` attribute
that ``RunReport`` never defined, so printing the trace of *every* finished run
raised ``AttributeError`` inside ``run_graph``'s ``finally`` block and threw
the answer away.
"""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import observability
from observability import (
    RunReport,
    StageTiming,
    end_run,
    stage,
    start_run,
)


def report_with(*stages: StageTiming) -> RunReport:
    return RunReport(question="give my vocabulary of A1", model="ollama/x", stages=list(stages))


class SummaryTests(unittest.TestCase):
    """``summary()`` is the last line of every run, so it must never raise."""

    def test_summary_of_a_clean_run(self):
        report = report_with(
            StageTiming(name="understand", seconds=7.89),
            StageTiming(name="research", seconds=9.68),
        )
        text = report.summary()

        self.assertIn("2 stage(s)", text)
        self.assertNotIn("failed", text)
        self.assertEqual(report.failed, [])

    def test_summary_of_a_run_with_no_stages(self):
        self.assertIn("0 stage(s)", RunReport().summary())

    def test_summary_names_failed_stages(self):
        report = report_with(
            StageTiming(name="plan", seconds=1.0),
            StageTiming(name="evidence", seconds=2.0, ok=False, error="TimeoutError: slow"),
        )
        text = report.summary()

        self.assertIn("failed: evidence", text)
        self.assertNotIn("failed: plan", text)

    def test_summary_reports_the_token_split(self):
        report = report_with(
            StageTiming(name="research.llm", input_tokens=120, output_tokens=20, reasoning_tokens=80),
        )
        text = report.summary()

        self.assertIn("in=120", text)
        self.assertIn("out=20", text)
        self.assertIn("thinking=80", text)
        self.assertIn("80%", text)


class FailedStagesTests(unittest.TestCase):
    def test_failed_keeps_run_order(self):
        report = report_with(
            StageTiming(name="understand"),
            StageTiming(name="plan", ok=False),
            StageTiming(name="search"),
            StageTiming(name="evidence", ok=False),
        )

        self.assertEqual([s.name for s in report.failed], ["plan", "evidence"])

    def test_slowest_is_marked_and_failures_flagged(self):
        slow = StageTiming(name="research", seconds=9.68)
        report = report_with(StageTiming(name="search", seconds=1.0), StageTiming(name="evidence", ok=False), slow)
        rows = report.lines()

        self.assertIn("<-- slowest", rows[2])
        self.assertIn("FAILED", rows[1])
        self.assertNotIn("FAILED", rows[0])


class StageContextManagerTests(unittest.TestCase):
    def test_error_is_recorded_and_re_raised(self):
        report = RunReport()

        with self.assertRaises(ValueError):
            with stage(report, "evidence"):
                raise ValueError("unreadable page")

        timing = report.stages[0]
        self.assertFalse(timing.ok)
        self.assertEqual(timing.error, "ValueError: unreadable page")
        self.assertEqual([s.name for s in report.failed], ["evidence"])
        self.assertIn("failed: evidence", report.summary())

    def test_successful_stage_is_recorded(self):
        report = RunReport()

        with stage(report, "search") as timing:
            timing.detail = "5 sources"

        self.assertTrue(report.stages[0].ok)
        self.assertEqual(report.failed, [])


class TraceFileTests(unittest.TestCase):
    def test_trace_is_appended(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "traces.jsonl"

            for _ in range(2):
                report_with(StageTiming(name="search", seconds=0.5)).write_trace(str(path))

            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 2)

    def test_unwritable_trace_never_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(report_with().write_trace(folder))

    def test_end_run_flushes_and_clears_the_active_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "traces.jsonl"

            with mock.patch.object(observability, "TRACE_PATH", str(path)):
                report = start_run(question="q", model="m")
                with stage(report, "search"):
                    pass

                self.assertIs(end_run(), report)
                self.assertIsNone(end_run())

            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()