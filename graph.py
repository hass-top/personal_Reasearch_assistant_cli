import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langgraph.graph import StateGraph, END ,START
from observability import current_report, stage
from state import State, Understanding
from config import (
    DEFAULT_INTENT,
    DEFAULT_SUBJECT,
    INTENTS,
    SUBJECTS,
)
from llm.factory import create_llm
from profiles.factory import create_strategy
from search.content import fetch_evidence
from search.factory import create_search
from search.formatter import format_context
from search.merger import merge_sources
from search.ranking import ranking_enabled


PLANNER_PROMPT = """You are planning web searches for a research assistant.

Question: {question}

{focus}
Subject context (plan queries for this field):
{guidance}

Write {count} short and different web search queries that together would answer
the question. Reply with a JSON array of strings and nothing else.

Example reply:
["first query", "second query", "third query"]"""


# What the user wants to do decides what kind of page is worth searching for.
# Without this the planner searches the same way whatever the intent is.
INTENT_QUERY_HINTS = {
    "learn": (
        "The user wants to learn: favour pages that explain and teach "
        "(tutorial, guide, explained, examples)."
    ),
    "practice": (
        "The user wants to practise: favour pages with exercises, worksheets, "
        "drills and answer keys."
    ),
    "verify": (
        "The user wants to check something: favour authoritative references "
        "(official documentation, reliable sources) to check the claim against."
    ),
    "research": (
        "The user wants sources: favour primary sources (official "
        "documentation, standards, papers)."
    ),
}


CLASSIFIER_PROMPT = """You are analyzing a research request before any search is planned.
Question: {question}

Subjects: {subjects}
Intents: {intents}

subject = WHAT the question is about: pick one of the subjects above, or
general when the question fits none of them.
topic = the main thing the user wants to know about (a few words).
intent = WHAT THE USER WANTS TO DO:
  learn = explain or understand something,
  practice = do exercises or drills,
  verify = check an answer or a solution,
  research = find sources or documentation.
task = the specific operation requested (explain, solve, define, compare...).
ambiguous = true ONLY when different interpretations would lead to
substantially different searches.
clarification = a short question asked only when ambiguous is true,
otherwise an empty string.
clarification_options = when ambiguous is true, 2 to 4 mutually exclusive
readings of the request (each one a short, self-contained restatement that
could be used directly as a search question); an empty list when the
request is not ambiguous.
Do not invent information that is not present in the question.

Reply with JSON only and nothing else:
{{"subject": "...", "topic": "...", "intent": "...", "task": "...",
  "ambiguous": false, "clarification": "",
  "clarification_options": []}}"""


# Small models sometimes answer with prose and no JSON at all. Instead of
# silently falling back to the defaults, ask once more, pointing at the format.
REPAIR_PROMPT = """Your previous reply was not valid JSON.

Reply again with ONLY a JSON object, no prose and no code fences, using exactly
these keys:
subject, topic, intent, task, ambiguous, clarification, clarification_options"""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _compact(value: str) -> str:
    """Lowercase and drop punctuation ("Cyber Security" -> "cybersecurity")."""
    return "".join(ch for ch in value.lower() if ch.isalnum())


def _snap_subject(value: str) -> str:
    """Map a raw model answer onto SUBJECTS, else the default subject."""
    cleaned = " ".join(value.lower().split())
    compact = _compact(cleaned)

    if cleaned in SUBJECTS:
        return cleaned

    for subject in SUBJECTS:
        if subject in cleaned or _compact(subject) in compact:
            return subject

    return DEFAULT_SUBJECT


_INTENT_SYNONYMS = (
    ("understand", "learn"),
    ("explain", "learn"),
    ("exercise", "practice"),
    ("drill", "practice"),
    ("check", "verify"),
    ("correct", "verify"),
    ("documentation", "research"),
    ("sources", "research"),
)


def _snap_intent(value: str) -> str:
    """Map a raw model answer onto INTENTS, else the default intent."""
    cleaned = " ".join(value.lower().split())

    if cleaned in INTENTS:
        return cleaned

    for intent in INTENTS:
        if intent in cleaned:
            return intent

    for word, intent in _INTENT_SYNONYMS:
        if word in cleaned:
            return intent

    return DEFAULT_INTENT


def _clean_text(value: object, limit: int = 200) -> str:
    """Collapse whitespace and cap the length of a free-text field."""
    if value is None or isinstance(value, bool):
        return ""

    return " ".join(str(value).split())[:limit]


def _to_bool(value: object) -> bool:
    """Coerce a model answer into a real bool (defaults to False)."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "on"}

    if isinstance(value, (int, float)):
        return bool(value)

    return False


_UNDERSTANDING_KEYS = ("subject", "topic", "intent", "task", "clarification")

#: At most this many readings are kept for an ambiguous request.
MAX_CLARIFICATION_OPTIONS = 4


def _clean_options(value: object) -> list[str]:
    """Keep up to ``MAX_CLARIFICATION_OPTIONS`` non-empty strings.

    Small models sometimes return a single string instead of a list (or
    numbers/nulls next to the real options), so anything that is not a
    usable string is dropped.
    """
    if isinstance(value, str):
        value = [value] if value.strip() else []

    if not isinstance(value, (list, tuple)):
        return []

    options: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue

        text = _clean_text(item, 300)
        if not text or text in options:
            continue

        options.append(text)
        if len(options) == MAX_CLARIFICATION_OPTIONS:
            break

    return options


def _default_understanding() -> dict:
    """Safe fallback for the understand node (off, failed or unusable reply)."""
    return {
        "subject": DEFAULT_SUBJECT,
        "topic": "",
        "intent": DEFAULT_INTENT,
        "task": "",
        "ambiguous": False,
        "clarification": "",
        "clarification_options": [],
    }


def normalise_understanding(raw: object) -> dict:
    """Coerce any understanding payload onto the vocabulary and the types.

    Both paths of the understand node end up here -- the structured output
    (already a valid ``Understanding`` object) and the JSON text parser -- so a
    well-formed answer still cannot smuggle an unknown subject, a ``"true"``
    string or a twelve-option list into the state. A missing or non-dict
    payload returns the safe defaults.
    """
    parsed = raw if isinstance(raw, dict) else {}

    return {
        "subject": _snap_subject(_clean_text(parsed.get("subject"))),
        "topic": _clean_text(parsed.get("topic")),
        "intent": _snap_intent(_clean_text(parsed.get("intent"))),
        "task": _clean_text(parsed.get("task")),
        "ambiguous": _to_bool(parsed.get("ambiguous")),
        "clarification": _clean_text(parsed.get("clarification"), 500),
        "clarification_options": _clean_options(
            parsed.get("clarification_options")
        ),
    }


def parse_understanding(reply: str) -> dict:
    """Pull the query understanding fields out of a model reply.

    Returns ``subject`` / ``intent`` snapped onto the configured vocabulary
    (``topic`` / ``task`` / ``clarification`` kept as cleaned free text) and
    ``ambiguous`` as a real bool.

    Small models often wrap or break the JSON, so a per-field regex salvage
    path runs when no JSON object can be decoded; anything still unusable
    falls back to the safe defaults (general / research / not ambiguous).
    """
    reply = reply or ""
    parsed: dict = {}

    match = _JSON_OBJECT_RE.search(reply)
    if match:
        try:
            candidate = json.loads(match.group(0))
        except json.JSONDecodeError:
            candidate = None

        if isinstance(candidate, dict):
            parsed = candidate

    if not parsed:
        for key in _UNDERSTANDING_KEYS:
            found = re.search(rf'"{key}"\s*:\s*"([^"]*)"', reply, re.IGNORECASE)
            if found:
                parsed[key] = found.group(1)

        found = re.search(r'"ambiguous"\s*:\s*(true|false|\d+)', reply, re.IGNORECASE)
        if found:
            parsed["ambiguous"] = found.group(1)

        # The options array cannot be decoded as a whole (the JSON is
        # broken), so its quoted items are pulled out one by one.
        found = re.search(
            r'"clarification_options"\s*:\s*\[(.*?)\]',
            reply,
            re.DOTALL | re.IGNORECASE,
        )
        if found:
            parsed["clarification_options"] = re.findall(r'"([^"]*)"', found.group(1))

    return normalise_understanding(parsed)


def parse_classification(reply: str) -> tuple[str, str]:
    """Back-compatible (subject, intent) view of ``parse_understanding``."""
    data = parse_understanding(reply)
    return data["subject"], data["intent"]


def structured_understanding(llm, prompt: str):
    """Ask the provider for the JSON contract through the schema itself.

    Every provider wrapper keeps the raw LangChain model as ``.llm``, and
    ``with_structured_output`` then makes the model return an ``Understanding``
    object, so there is no text to parse. Returns ``None`` when that is not
    possible -- a custom wrapper without ``.llm``, or a local model without
    tool calling -- so the caller can fall back to the plain prompt.
    """
    chat = getattr(llm, "llm", None)
    if chat is None or not hasattr(chat, "with_structured_output"):
        return None

    reply = chat.with_structured_output(Understanding).invoke(prompt)

    if isinstance(reply, Understanding):
        return reply.model_dump()

    if isinstance(reply, dict):
        return reply

    return None


def route_after_understanding(state: State) -> str:
    """Ambiguous requests ask first, clear ones go straight to the planner.

    understand --> ambiguous? --yes--> clarify
                            \\--no---> plan
    """
    if state.ambiguous:
        return "clarify"

    return "plan"


def format_clarification(clarification: str, options: list[str]) -> str:
    """Render the clarify node's result: the question + numbered readings.

    Without usable options the plain open question is shown instead, so the
    user can always answer in free text on the next run.
    """
    question = (clarification or "").strip() or (
        "Could you add more detail so the search targets the right topic?"
    )

    lines = ["**Clarification needed**", "", question]

    for number, option in enumerate(options, start=1):
        lines.append(f"{number}. {option}")

    if options:
        lines.extend(("", "Reply with the number of the option you mean."))

    return "\n".join(lines)


def parse_queries(reply: str, count: int) -> list[str]:
    """Pull up to ``count`` search queries out of a model reply.

    Small models often wrap the JSON in prose or ignore the format completely,
    so a line-by-line fallback runs when no JSON array can be decoded.
    """
    if count < 1:
        return []

    candidates = []

    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, list):
            candidates = [item for item in parsed if isinstance(item, str)]

    if not candidates and match:
        # JSON that is *nearly* valid (trailing commas, stray quotes): salvage it.
        for chunk in match.group(0).strip().strip("[]").split(","):
            candidates.append(chunk.strip().strip("\"'` "))

    if not candidates:
        for line in reply.splitlines():
            if "`" in line:
                continue

            cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            candidates.append(cleaned.strip().strip("\"'` "))

    queries = []
    seen = set()
    for candidate in candidates:
        query = " ".join(candidate.split())
        if len(query) < 3 or len(query) > 200:
            continue
        if any(char in query for char in '[]{}`'):
            continue
        if not any(char.isalnum() for char in query):
            continue

        key = query.lower()
        if key in seen:
            continue

        seen.add(key)
        queries.append(query)

        if len(queries) == count:
            break

    return queries


def format_evidence(evidence: list[dict]) -> str:
    """Render the fetched page text as the same numbered block as the sources.

    The numbering matches ``search.formatter.format_context`` ([1], [2], ...)
    so the model cites the excerpt it actually used, and the quality score is
    kept so it can still tell a university page from a forum post.
    """
    blocks = []

    for index, item in enumerate(evidence, start=1):
        quality = ""
        if item.get("quality"):
            quality = f"Quality: {item['quality']}\n"

        blocks.append(
            f"[{index}] {item.get('title', '')}\n"
            f"{item.get('url', '')}\n"
            f"{quality}"
            f"{item.get('text', '')}"
        )

    return "\n\n".join(blocks)


# What the user wants to do decides how the answer is written. The intent was
# already paid for by the understanding call, so it is used here too instead of
# only being displayed.
INTENT_ANSWER_HINTS = {
    "learn": (
        "The user is learning this: explain it step by step, define the key "
        "terms, and illustrate with concrete examples."
    ),
    "practice": (
        "The user wants to practise: end with 3 to 5 exercises of increasing "
        "difficulty, followed by their answer key."
    ),
    "verify": (
        "The user wants to check something: give the verdict first (correct, "
        "partly correct or wrong), then justify it, and point out any "
        "disagreement between the sources."
    ),
    "research": (
        "The user wants sources: be dense and factual, lead with the key "
        "findings, and keep every claim tied to a source."
    ),
}

# The task is free text, so it is matched by keyword rather than vocabulary.
_TASK_HINTS = (
    (
        ("compare", "versus", " vs ", "difference", "better"),
        "Compare the options point by point on the same criteria instead of "
        "describing each one separately, and finish with a verdict.",
    ),
    (
        ("define", "definition", "meaning", "what is", "what are"),
        "Start with a short precise definition, then add the detail.",
    ),
    (
        ("solve", "calculate", "compute", "work out", "how do i"),
        "Show the steps of the solution, not only the final result.",
    ),
    (
        ("summar", "recap", "tldr", "brief"),
        "Be concise: a short summary first, details only if they matter.",
    ),
    (
        ("list", "examples", "give me some"),
        "Answer as a clear list of items.",
    ),
)


def task_directive(task: str) -> str:
    """Turn the free-text task into one structural instruction ("" if none)."""
    text = f" {task.lower()} "

    for keywords, hint in _TASK_HINTS:
        if any(keyword in text for keyword in keywords):
            return hint

    return ""


def understanding_guidance(state: State) -> str:
    """The answer-style instructions carried over from the understanding node.

    Empty when the understanding call is off or the model said nothing useful,
    so the prompt stays exactly as it was before.
    """
    parts = []

    hint = INTENT_ANSWER_HINTS.get(state.intent, "")
    if hint:
        parts.append(hint)

    hint = task_directive(state.task)
    if hint:
        parts.append(hint)

    if state.topic:
        parts.append(f"The question is about {state.topic}.")

    return " ".join(parts)


def planner_focus(state: State) -> str:
    """The understanding fields the planner needs, as prompt text.

    Empty when the understanding node is off, which keeps the planner prompt
    as it was before.
    """
    parts = []

    if state.topic:
        parts.append(f"Topic: {state.topic}")
    if state.intent:
        parts.append(f"Intent: {state.intent}")
    if state.task:
        parts.append(f"Task: {state.task}")

    hint = INTENT_QUERY_HINTS.get(state.intent, "")
    if hint:
        parts.append(hint)

    if not parts:
        return ""

    lines = ["What the user actually asked for:"]
    lines.extend(f"- {part}" for part in parts)

    return "\n".join(lines) + "\n"


def build_prompt(state: State) -> str:
    """Assemble the research prompt from what the graph actually collected.

    When the evidence node managed to read the pages, their text replaces the
    search snippets (a snippet is a teaser, not something to answer from) and
    the answer is asked to cite the excerpt it used. With no readable page the
    prompt falls back to the snippets. The understanding fields decide how the
    answer is written.
    """
    understood = understanding_guidance(state)
    evidence = format_evidence(state.evidence)

    if not state.context and not evidence:
        return (
            f"Question: {state.question}\n\n"
            "No web sources were available, so answer from your own knowledge "
            "and clearly say that the answer is not backed by sources."
            + (f"\n\n{understood}" if understood else "")
        )

    if evidence:
        # The pages were read, so the answer can be written from them and every
        # claim can be tied to the excerpt that supports it.
        sources_block = f"Source excerpts:\n{evidence}"
        guidance = (
            "Write a clear, well structured answer to the question using the "
            "source excerpts above. They are the real text of the pages, so "
            "ground every claim in them and cite the excerpt you used with "
            "bracketed numbers like [1], sentence by sentence. Quote figures, "
            "dates and names only when an excerpt states them. If the excerpts "
            "do not answer the question, say so explicitly instead of filling "
            "the gap from memory."
        )
    else:
        # No page could be read: fall back to the search snippets.
        sources_block = f"Web sources:\n{state.context}"
        guidance = (
            "Write a clear, well structured answer to the question using the "
            "web sources above. Cite every source you use with bracketed "
            "numbers like [1]. If the sources do not answer the question, say "
            "so explicitly."
        )

    if ranking_enabled():
        guidance += (
            " Sources are listed best quality first and each one carries a "
            "quality score out of 100: base the answer on the highest scored "
            "sources, and say when a claim only rests on a low scored source "
            "such as a forum or an unrecognised domain."
        )

    strategy = create_strategy(state.subject)
    subject_guidance = strategy.answer_directive()
    if subject_guidance:
        guidance += f"\n\nFor this subject ({strategy.name}):\n{subject_guidance}"

    if understood:
        guidance += f"\n\n{understood}"

    return (
        f"Question: {state.question}\n\n"
        f"{sources_block}\n\n"
        f"{guidance}"
    )


def create_graph(
        provider:str,
        model:str,
        api_key:str | None = None,
        search_provider:str = "duckduckgo",
        search_api_key:str | None = None,
        max_sources:int = 5,
        planner:bool = True,
        max_queries:int = 3,
        classify:bool = True,
        structured_output:bool = True,
        evidence:bool = True,
        evidence_sources:int = 3,
        evidence_chars:int = 1500,
):
    llm =create_llm(provider=provider,model=model,api_key=api_key)
    search = create_search(
        provider=search_provider,
        api_key=search_api_key,
        max_results=max_sources,
    )

    # Captured under another name: the nested `def understand` below rebinds
    # the `classify` parameter in this scope, so read it while it is still
    # the flag. `classify=off` now skips the whole understanding call.
    understand_enabled = classify

    # Same reason: the `evidence` parameter is read here, before the nested
    # `def collect_evidence` shadows it with the node itself.
    evidence_enabled = evidence


    def understand(state:State):
        """Read the request before planning: subject, topic, intent, task
        and whether it is ambiguous (Query Understanding, Phase 1).

        Three layers, from the most to the least reliable:

        1. the schema itself (``with_structured_output``), so the model cannot
           answer with something that is not the object we asked for;
        2. the JSON text parser, for providers without tool calling;
        3. one repair round-trip, then the safe defaults.
        """
        if not understand_enabled:
            return _default_understanding()

        prompt = CLASSIFIER_PROMPT.format(
            question=state.question,
            subjects=", ".join(SUBJECTS) + ", " + DEFAULT_SUBJECT,
            intents=", ".join(INTENTS),
        )

        with stage(current_report(), "understand") as timing:
            if structured_output:
                try:
                    data = structured_understanding(llm, prompt)
                except Exception:
                    # Tool calling is not supported by every model: fall through
                    # to the text parser instead of losing the understanding.
                    data = None

                if data is not None:
                    timing.detail = "structured output"
                    return normalise_understanding(data)

            try:
                reply = llm.invoke(prompt)
            except Exception:
                timing.detail = "classifier failed, using defaults"
                return _default_understanding()

            if not isinstance(reply.content, str):
                return _default_understanding()

            # Prose with no JSON at all: ask once more before giving up.
            if not _JSON_OBJECT_RE.search(reply.content):
                try:
                    repaired = llm.invoke(REPAIR_PROMPT)
                except Exception:
                    return _default_understanding()

                if not isinstance(repaired.content, str):
                    return _default_understanding()

                timing.detail = "repaired malformed JSON"
                return parse_understanding(repaired.content)

            timing.detail = "parsed JSON"
            return parse_understanding(reply.content)


    def clarify(state:State):
        """Stop before searching: hand the readings back to the user."""
        return{
            "result": format_clarification(
                state.clarification,
                state.clarification_options,
            ),
        }


    def plan(state:State):
        if not planner:
            return{"queries": []}

        strategy = create_strategy(state.subject)
        prompt = PLANNER_PROMPT.format(
            question=state.question,
            focus=planner_focus(state),
            guidance=strategy.planner_directive(),
            count=max_queries,
        )

        with stage(current_report(), "plan") as timing:
            try:
                reply = llm.invoke(prompt)
            except Exception:
                timing.detail = "planner failed"
                return{"queries": []}

            if not isinstance(reply.content, str):
                return{"queries": []}

            queries = parse_queries(reply.content, max_queries)
            timing.detail = f"{len(queries)} extra queries"

            return{"queries": queries}


    def find_sources(state:State):
        queries = [state.question] + [
            query for query in state.queries
            if query.lower() != state.question.lower()
        ]

        with stage(current_report(), "search") as timing:
            # Queries are independent network calls, so they run together
            # instead of one after another. With the planner off this is a
            # single query and the pool costs nothing.
            if len(queries) > 1:
                with ThreadPoolExecutor(max_workers=len(queries)) as pool:
                    futures = [pool.submit(search.search, q) for q in queries]
                    batches = [
                        future.result() for future in as_completed(futures)
                    ]
            else:
                batches = [search.search(queries[0])]

            sources = merge_sources(batches, max_sources)
            timing.detail = f"{len(sources)} sources from {len(queries)} queries"

            return{
                "sources": sources,
                "context": format_context(sources),
                "queries": queries,
            }


    def collect_evidence(state:State):
        """Read the best pages so the answer is written from their real text.

        A search snippet is a teaser, so answering from snippets alone means
        answering from something the model never read. Sources already sorted
        best-first by the merger are the ones worth downloading. A page that
        cannot be read is skipped, and the research node then falls back to the
        snippets.
        """
        if not evidence_enabled or not state.sources:
            return{"evidence": []}

        with stage(current_report(), "evidence") as timing:
            try:
                pages = fetch_evidence(
                    state.sources,
                    limit=evidence_sources,
                    max_chars=evidence_chars,
                )
            except Exception:
                timing.detail = "fetch failed"
                return{"evidence": []}

            timing.detail = f"{len(pages)} pages read"
            return{"evidence": pages}


    def research(state:State):
        with stage(current_report(), "research") as timing:
            try:
                result = llm.invoke(build_prompt(state))
            except Exception as exc:
                # Every other node degrades gracefully, so this one must too: a
                # timeout or a dead Ollama server must not kill the CLI.
                timing.detail = f"answer failed ({type(exc).__name__})"
                return{
                    "result": (
                        "I could not generate an answer. The model call failed "
                        f"({type(exc).__name__}). Try a smaller model, raise "
                        "OLLAMA_TIMEOUT, or check that Ollama is running."
                    )
                }

            return{
                "result": result.content
            }
    graph = StateGraph(State)
    graph.add_node("understand",understand)
    graph.add_node("clarify",clarify)
    graph.add_node("plan",plan)
    graph.add_node("search",find_sources)
    graph.add_node("evidence",collect_evidence)
    graph.add_node("research",research)
    graph.add_edge(START , "understand")
    graph.add_conditional_edges(
        "understand",
        route_after_understanding,
        {
            "clarify": "clarify",
            "plan": "plan",
        },
    )
    graph.add_edge("clarify", END)
    graph.add_edge("plan", "search")
    graph.add_edge("search", "evidence")
    graph.add_edge("evidence", "research")
    graph.add_edge("research",END)

    return graph.compile()

