"""Timing and token accounting for one research run.

The CLI used to wrap the whole graph in a single ``console.status`` spinner, so
a slow run looked exactly like a hung one: no stage was named, no duration was
shown, and there was no way to tell a slow model from a slow network. This
module records what each stage actually cost so the run can explain itself.

Two things are measured:

* **Stages** -- the graph nodes (understand, plan, search, evidence, research)
  plus the LLM calls inside them. ``stage()`` is a context manager that times
  the block and never swallows the block's own errors, so instrumenting a node
  cannot change what the graph does.
* **Tokens** -- input/output/reasoning token counts per LLM call, when the
  provider reports them. Reasoning tokens are the interesting column: a model
  like ``deepseek-r1`` can spend an entire budget thinking and return no answer
  at all, which is invisible unless the two counts are shown side by side.

Every completed run is appended to a JSONL trace file (``TRACE_PATH``) so
consecutive runs can be compared and a regression spotted later. Writing is
best effort: a broken or read-only trace file never fails a research run.
"""

from __future__ import annotations

import json
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator


#: Where the per-run traces are appended. Empty disables file tracing.
TRACE_PATH = os.getenv("TRACE_PATH", "traces.jsonl")

#: Print each stage to the console as it finishes, not just at the end.
TRACE_VERBOSE = os.getenv("TRACE_VERBOSE", "on").strip().lower() not in {
    "0",
    "off",
    "false",
    "no",
}

#: Only keep this many traces in the file, so it cannot grow forever.
TRACE_MAX_LINES = 500


@dataclass
class StageTiming:
    """One timed block: a graph node, or a single LLM call inside one."""

    name: str
    seconds: float = 0.0
    ok: bool = True
    error: str = ""
    detail: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunReport:
    """Accumulates every stage of a single run and renders it for humans."""

    question: str = ""
    model: str = ""
    stages: list[StageTiming] = field(default_factory=list)
    started: float = field(default_factory=time.perf_counter)

    # -- recording ---------------------------------------------------------

    def record(self, timing: StageTiming) -> StageTiming:
        self.stages.append(timing)
        return timing

    @property
    def total_seconds(self) -> float:
        return time.perf_counter() - self.started

    @property
    def slowest(self) -> StageTiming | None:
        return max(self.stages, key=lambda s: s.seconds) if self.stages else None

    @property
    def failed(self) -> list[StageTiming]:
        """The stages that raised, in the order they ran.

        ``summary()`` names them so a run that degraded instead of crashing
        (a fallback planner, an unreachable source) is visible in one line.
        """
        return [s for s in self.stages if not s.ok]

    def token_totals(self) -> tuple[int, int, int]:
        """Summed (input, output, reasoning) tokens over every LLM call."""
        def total(attr: str) -> int:
            return sum(
                getattr(s, attr) or 0
                for s in self.stages
                if getattr(s, attr) is not None
            )

        return total("input_tokens"), total("output_tokens"), total("reasoning_tokens")

    # -- rendering ---------------------------------------------------------

    def lines(self) -> list[str]:
        """Human readable ``stage  time  detail`` rows, slowest stage marked."""
        slowest = self.slowest
        rows = []

        for stage in self.stages:
            marker = "  <-- slowest" if stage is slowest else ""
            mark = "" if stage.ok else "  FAILED"
            detail = f"  {stage.detail}" if stage.detail else ""
            rows.append(
                f"  {stage.name:<26} {stage.seconds:6.2f}s{mark}{detail}{marker}"
            )

        return rows

    def summary(self) -> str:
        """One-paragraph total, including the token split when known."""
        inputs, outputs, reasoning = self.token_totals()
        text = f"total {self.total_seconds:.2f}s over {len(self.stages)} stage(s)"

        if inputs or outputs or reasoning:
            text += f" | tokens in={inputs} out={outputs}"

            # The diagnostic that matters: a run where most of the budget went
            # into thinking is slow *and* may have produced nothing usable.
            if reasoning:
                share = reasoning / max(reasoning + outputs, 1)
                text += f" thinking={reasoning} ({share:.0%} of generated)"

        if self.failed:
            names = ", ".join(s.name for s in self.failed)
            text += f" | failed: {names}"

        return text

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "model": self.model,
            "total_seconds": round(self.total_seconds, 3),
            "stages": [s.to_dict() for s in self.stages],
        }

    # -- persistence -------------------------------------------------------

    def write_trace(self, path: str | None = None) -> bool:
        """Append this run to the JSONL trace file. Never raises."""
        target = path if path is not None else TRACE_PATH

        if not target:
            return False

        try:
            file_path = Path(target)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            _trim(file_path, TRACE_MAX_LINES)

            with file_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(self.to_dict(), ensure_ascii=False) + "\n"
                )

            return True
        except Exception:
            # Tracing is a diagnostic aid, never a reason to fail a run.
            return False


def _trim(path: Path, max_lines: int) -> None:
    """Keep the trace file bounded by dropping its oldest lines."""
    if not path.exists():
        return

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return

    if len(lines) > max_lines:
        path.write_text(
            "\n".join(lines[-max_lines:]) + "\n",
            encoding="utf-8",
        )


@contextmanager
def stage(
    report: RunReport,
    name: str,
    on_finish=None,
) -> Iterator[StageTiming]:
    """Time a block and record it on ``report``.

    ``on_finish(timing)`` is called once the block ends, which lets the CLI
    print the stage as it completes instead of only at the end. An exception
    from the block is recorded on the timing and re-raised, so instrumenting a
    node never hides a real failure.
    """
    timing = StageTiming(name=name)
    start = time.perf_counter()
    previous = current_stage()
    _set_current_stage(name)

    try:
        yield timing
    except Exception as exc:
        timing.ok = False
        timing.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _set_current_stage(previous)
        timing.seconds = time.perf_counter() - start
        report.record(timing)

        if on_finish is not None:
            try:
                on_finish(timing)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Which stage is running right now
# ---------------------------------------------------------------------------

#: Thread local, so a stage name stays correct inside the search threads.
_local = threading.local()


def _set_current_stage(name: str | None) -> None:
    _local.stage = name


def current_stage() -> str | None:
    """The name of the stage currently executing on this thread, if any.

    Providers use it to label an inner LLM call after the node that made it
    (``research.llm`` rather than a bare ``llm``), which keeps the trace
    readable without changing any ``invoke`` signature.
    """
    return getattr(_local, "stage", None)


# ---------------------------------------------------------------------------
# The report for the run currently in progress
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_current: RunReport | None = None


def current_report() -> RunReport:
    """The active report, created on first use.

    Graph nodes and provider wrappers both need to reach the same report
    without threading one through every signature, so it is kept here. The
    lock matters because searches run in parallel threads.
    """
    global _current

    with _lock:
        if _current is None:
            _current = RunReport()

        return _current


def start_run(question: str = "", model: str = "") -> RunReport:
    """Begin a new report, replacing any previous one."""
    global _current

    with _lock:
        _current = RunReport(question=question, model=model)
        return _current


def end_run() -> RunReport | None:
    """Finish the active report (writing its trace) and return it."""
    global _current

    with _lock:
        report = _current
        _current = None

    if report is not None:
        report.write_trace()

    return report


def report_tokens(timing: StageTiming, message) -> None:
    """Copy a LangChain message's token counts onto a stage timing.

    ``reasoning_tokens`` is nested one level down in ``output_tokens_details``,
    which is the number that explains a slow deepseek-r1 run.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    timing.input_tokens = usage.get("input_tokens")
    timing.output_tokens = usage.get("output_tokens")

    details = usage.get("output_tokens_details") or {}
    if isinstance(details, dict):
        reasoning = details.get("reasoning_tokens")
        if isinstance(reasoning, int):
            timing.reasoning_tokens = reasoning
