"""Local Ollama provider.

Two settings matter for speed and are easy to get wrong:

``num_predict``
    A hard cap on generated tokens. Without it Ollama generates until the
    context window is full, so a slow model looks exactly like a hung one.

``client_kwargs={"timeout": ...}``
    A hard cap on the HTTP request. Verified against Ollama 0.34.4: the call
    raises ``ReadTimeout`` at exactly the configured number of seconds, so a
    run can never block forever waiting for the local server.

``keep_alive`` keeps the model resident between questions, which removes the
reload pause at the start of every run after the first.

``reasoning`` is forwarded to Ollama's ``think`` flag. Reasoning models such as
``deepseek-r1`` spend their whole token budget thinking and can return an empty
answer, so it defaults to off; set ``REASONING=on`` for a model that needs it.
"""

import os

from langchain_ollama import ChatOllama

from observability import (
    current_report,
    current_stage,
    report_tokens,
    stage,
)

#: Seconds any single call may take before the request is aborted.
DEFAULT_TIMEOUT = 120

#: Generated-token ceiling. Large enough for a real answer, small enough that a
#: runaway generation is cut off instead of running for minutes.
DEFAULT_NUM_PREDICT = 1024

#: How long Ollama keeps the model loaded after a request.
DEFAULT_KEEP_ALIVE = "10m"

#: Context window; 4096 fits the research prompt with a few evidence excerpts.
DEFAULT_NUM_CTX = 4096


def _int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, ""))
    except ValueError:
        return default

    return value if value > 0 else default


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    return value.strip().lower() not in {"0", "off", "false", "no"}


class OllamaProvider:
    """Chat model served by a local Ollama daemon."""

    def __init__(
        self,
        model: str,
        timeout: int | None = None,
        num_predict: int | None = None,
        keep_alive: str | None = None,
        num_ctx: int | None = None,
        reasoning: bool | None = None,
    ):
        self.model = model
        self.timeout = (
            timeout if timeout is not None else _int_env("OLLAMA_TIMEOUT", DEFAULT_TIMEOUT)
        )
        self.num_predict = (
            num_predict
            if num_predict is not None
            else _int_env("OLLAMA_NUM_PREDICT", DEFAULT_NUM_PREDICT)
        )
        self.num_ctx = (
            num_ctx if num_ctx is not None else _int_env("OLLAMA_NUM_CTX", DEFAULT_NUM_CTX)
        )
        self.keep_alive = keep_alive or os.getenv(
            "OLLAMA_KEEP_ALIVE", DEFAULT_KEEP_ALIVE
        )
        self.reasoning = (
            reasoning if reasoning is not None else _bool_env("REASONING", False)
        )

        self.llm = ChatOllama(
            model=model,
            temperature=0,
            num_predict=self.num_predict,
            num_ctx=self.num_ctx,
            keep_alive=self.keep_alive,
            reasoning=self.reasoning,
            # The bound that stops an infinite wait on the local server.
            client_kwargs={"timeout": self.timeout},
        )

    def invoke(self, prompt: str, stage_name: str | None = None):
        """Send ``prompt`` and return the LangChain message.

        The timing entry is labelled after the graph node that made the call
        (``research.llm`` rather than a bare ``llm``), which is what makes a
        slow run readable. A timeout surfaces as a normal exception, which every
        caller in the graph already handles.
        """
        report = current_report()
        node = current_stage()
        label = stage_name or (f"{node}.llm" if node else "llm")

        with stage(report, label) as timing:
            message = self.llm.invoke(prompt)
            report_tokens(timing, message)

            content = message.content
            text = content if isinstance(content, str) else str(content or "")

            # An empty answer after a full token budget is the signature of a
            # reasoning model that thought the whole time and said nothing.
            # Recording it makes that visible instead of silent.
            if not text.strip():
                timing.detail = "empty reply (budget spent?)"
            else:
                timing.detail = f"{len(text)} chars"

            return message


    def invoke(self , prompt: str):
        return self.llm.invoke(prompt)