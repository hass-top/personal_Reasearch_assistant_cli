import json
import re

from langgraph.graph import StateGraph, END ,START
from state import State
from llm.factory import create_llm
from search.factory import create_search
from search.formatter import format_context
from search.merger import merge_sources
from search.ranking import ranking_enabled


PLANNER_PROMPT = """You are planning web searches for a research assistant.

Question: {question}

Write {count} short and different web search queries that together would answer
the question. Reply with a JSON array of strings and nothing else.

Example reply:
["first query", "second query", "third query"]"""


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


def build_prompt(state: State) -> str:
    if not state.context:
        return (
            f"Question: {state.question}\n\n"
            "No web sources were available, so answer from your own knowledge "
            "and clearly say that the answer is not backed by sources."
        )

    guidance = (
        "Write a clear, well structured answer to the question using the web "
        "sources above. Cite every source you use with bracketed numbers like "
        "[1]. If the sources do not answer the question, say so explicitly."
    )

    if ranking_enabled():
        guidance += (
            " Sources are listed best quality first and each one carries a "
            "quality score out of 100: base the answer on the highest scored "
            "sources, and say when a claim only rests on a low scored source "
            "such as a forum or an unrecognised domain."
        )

    return (
        f"Question: {state.question}\n\n"
        f"Web sources:\n{state.context}\n\n"
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
):
    llm =create_llm(provider=provider,model=model,api_key=api_key)
    search = create_search(
        provider=search_provider,
        api_key=search_api_key,
        max_results=max_sources,
    )


    def plan(state:State):
        if not planner:
            return{"queries": []}

        prompt = PLANNER_PROMPT.format(
            question=state.question,
            count=max_queries,
        )

        try:
            reply = llm.invoke(prompt)
        except Exception:
            return{"queries": []}

        if not isinstance(reply.content, str):
            return{"queries": []}

        return{
            "queries": parse_queries(reply.content, max_queries)
        }


    def find_sources(state:State):
        queries = [state.question] + [
            query for query in state.queries
            if query.lower() != state.question.lower()
        ]

        batches = [search.search(query) for query in queries]
        sources = merge_sources(batches, max_sources)

        return{
            "sources": sources,
            "context": format_context(sources),
            "queries": queries,
        }


    def research(state:State):
        result= llm.invoke(build_prompt(state)) # state.question because of pydantic
        return{
            "result":result.content
        }
    graph = StateGraph(State)
    graph.add_node("plan",plan)
    graph.add_node("search",find_sources)
    graph.add_node("research",research)
    graph.add_edge(START , "plan")
    graph.add_edge("plan", "search")
    graph.add_edge("search", "research")
    graph.add_edge("research",END)

    return graph.compile()

