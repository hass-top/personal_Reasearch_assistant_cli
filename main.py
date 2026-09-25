import re
from dataclasses import dataclass

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.table import Table

import config
from config import (
    DEFAULT_MAX_QUERIES,
    DEFAULT_MAX_SOURCES,
    GROQ_API_KEY,
    GROQ_MODEL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    PLANNER_ENABLED,
    SEARCH_PROVIDER,
    TAVILY_API_KEY,
)
from graph import create_graph
from search.ranking import (
    average_score,
    ranking_enabled,
    source_score,
    source_short_quality,
)


console = Console()

CTRL_Q = "\x11"

try:
    import msvcrt
except ImportError:  # pragma: no cover - non-Windows fallback
    msvcrt = None


def is_quit_like(text: str | None) -> bool:
    """True for q/quit/exit or a Ctrl+Q (ASCII 17) anywhere in the text."""
    if not text:
        return False

    if CTRL_Q in text:
        return True

    return text.strip().lower() in {"q", "quit", "exit"}


def parse_menu_key(key: str) -> str:
    """Map one main-menu keypress to an action (``""`` when unknown)."""
    if is_quit_like(key):
        return "quit"

    cleaned = key.strip().lower()

    if cleaned in {"1", "\r", "\n"}:
        return "research"

    if cleaned in {"2", "p"}:
        return "parameters"

    return ""


def read_menu_key() -> str:
    """Read a single main-menu keypress.

    On Windows the key is captured immediately, so Ctrl+Q quits without
    pressing Enter; other platforms fall back to a normal line prompt.
    """
    label = (
        "[bold green]❯ Your choice[/bold green] "
        "[dim](1 = research, 2 = parameters, q = quit)[/dim] "
    )

    if msvcrt is None:
        return Prompt.ask(label, default="1")

    console.print(label, end="")

    key = msvcrt.getwch()

    if key in ("\x00", "\xe0"):
        # Extended key (arrow, F1...): swallow its second byte.
        msvcrt.getwch()
        key = ""

    console.print()
    return key


def parse_on_off(answer: str) -> bool | None:
    """Parse an on/off answer, or ``None`` when it is neither."""
    cleaned = (answer or "").strip().lower()

    if cleaned in {"on", "yes", "y", "1", "true"}:
        return True

    if cleaned in {"off", "no", "n", "0", "false"}:
        return False

    return None


def ask_key(label: str, api_key: str | None) -> str:
    """Reuse a key taken from the environment, otherwise ask the user for it."""
    if api_key:
        return api_key

    return Prompt.ask(
        f"[bold green]❯ {label}[/bold green]",
        password=True,
    )


def quality_cell(source: dict) -> str:
    """Colour coded quality score for the sources table (100 = most reliable)."""
    value = source_score(source)

    if value >= 75:
        style = "bold green"
    elif value >= 45:
        style = "yellow"
    else:
        style = "bold red"

    return (
        f"[{style}]{value}[/{style}]\n"
        f"[dim]{escape(source_short_quality(source))}[/dim]"
    )


def print_sources(sources: list[dict], answer: str) -> None:
    """Show every web source below the answer so each link can be checked."""
    cited = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    show_quality = ranking_enabled()

    if not cited:
        console.print(
            f"[bold yellow]⚠ This answer has no {escape('[1]')} style citations, "
            "so treat every claim as unverified and check it against the sources "
            "below.[/bold yellow]"
        )

    caption = (
        f"Numbers match the {escape('[1]')} style citations in the "
        "answer above. Ctrl+click a link to open it."
    )

    if show_quality:
        average = average_score(sources)
        caption += (
            f" Quality is scored 0-100 (highest first), average {average}/100."
            if average is not None
            else " Quality is scored 0-100 (highest first)."
        )

    table = Table(
        title=f"[bold cyan]Sources[/bold cyan] [dim]({len(sources)})[/dim]",
        caption=f"[dim]{caption}[/dim]",
        border_style="cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("#", style="bold cyan", justify="right", width=3, no_wrap=True)
    table.add_column("Source", overflow="fold")

    if show_quality:
        table.add_column("Quality", justify="right", no_wrap=True, min_width=9)

    for index, source in enumerate(sources, start=1):
        url = source["url"]
        if "[" in url or "]" in url:
            url_cell = f"[cyan]{escape(url)}[/cyan]"
        else:
            url_cell = f"[link={url}][cyan]{escape(url)}[/cyan][/link]"

        used = "\n[dim](not cited in the answer)[/dim]"
        if not cited or index in cited:
            used = ""

        cells = [
            str(index),
            f"[bold]{escape(source['title'])}[/bold]\n"
            f"{url_cell}\n"
            f"[dim]{escape(source['snippet'])}[/dim]"
            f"{used}",
        ]

        if show_quality:
            cells.append(quality_cell(source))

        table.add_row(*cells)

    console.print(table)


@dataclass
class Settings:
    """Everything the Parameters menu may change between two questions."""

    provider: str = "ollama"
    model: str = OLLAMA_MODEL
    api_key: str | None = None
    search_provider: str = SEARCH_PROVIDER
    search_api_key: str | None = None
    max_sources: int = DEFAULT_MAX_SOURCES
    planner: bool = PLANNER_ENABLED


def choose_model(settings: Settings) -> None:
    """Pick the chat model provider, its name and its API key."""
    console.print("\n[bold yellow]Choose your model:[/bold yellow]")
    console.print("[1]  Ollama")
    console.print("[2]  OpenAI")
    console.print("[3]  Groq")

    choice = Prompt.ask(
        "[bold green]❯ Select model[/bold green]",
        choices=["1", "2", "3"],
        default="1",
    )

    if choice == "1":
        settings.provider = "ollama"
        settings.model = OLLAMA_MODEL
        settings.api_key = None

    elif choice == "2":
        settings.provider = "openai"
        settings.model = Prompt.ask(
            "[bold green]❯ Model[/bold green]",
            default=OPENAI_MODEL,
        )
        settings.api_key = ask_key("API key", OPENAI_API_KEY)

    else:
        settings.provider = "groq"
        settings.model = Prompt.ask(
            "[bold green]❯ Model[/bold green]",
            default=GROQ_MODEL,
        )
        settings.api_key = ask_key("API key", GROQ_API_KEY)


def choose_search(settings: Settings) -> None:
    """Pick the search backend and its API key."""
    console.print("\n[bold yellow]Choose your search source:[/bold yellow]")
    console.print("[1]  DuckDuckGo (no API key)")
    console.print("[2]  Tavily (API key)")

    choice = Prompt.ask(
        "[bold green]❯ Select search[/bold green]",
        choices=["1", "2"],
        default="1" if SEARCH_PROVIDER == "duckduckgo" else "2",
    )

    if choice == "1":
        settings.search_provider = "duckduckgo"
        settings.search_api_key = None

    else:
        settings.search_provider = "tavily"
        settings.search_api_key = ask_key("Tavily API key", TAVILY_API_KEY)


def research_options(settings: Settings) -> None:
    """Tune how many sources are used and how their quality is ranked."""
    answer = Prompt.ask(
        "[bold green]❯ Max sources per answer[/bold green]",
        default=str(settings.max_sources),
    )

    try:
        settings.max_sources = max(1, int(answer))
    except ValueError:
        pass

    planner = parse_on_off(
        Prompt.ask(
            "[bold green]❯ Plan extra search queries[/bold green] "
            "[dim](on/off)[/dim]",
            default="on" if settings.planner else "off",
        )
    )

    if planner is not None:
        settings.planner = planner

    ranking = parse_on_off(
        Prompt.ask(
            "[bold green]❯ Source quality ranking[/bold green] "
            "[dim](on/off)[/dim]",
            default="on" if config.SOURCE_RANKING_ENABLED else "off",
        )
    )

    if ranking is not None:
        config.SOURCE_RANKING_ENABLED = ranking

    threshold = Prompt.ask(
        "[bold green]❯ Minimum source score 0-100[/bold green]",
        default=str(config.MIN_SOURCE_SCORE),
    )

    try:
        config.MIN_SOURCE_SCORE = max(0, int(threshold))
    except ValueError:
        pass


def settings_summary(settings: Settings) -> str:
    """One-line summary of the current settings (shown above the menu)."""
    return (
        f"model {settings.provider}/{settings.model}"
        f" · search {settings.search_provider}"
        f" · sources {settings.max_sources}"
        f" · planner {'on' if settings.planner else 'off'}"
        f" · ranking {'on' if config.SOURCE_RANKING_ENABLED else 'off'}"
    )


def parameters_menu(settings: Settings) -> bool:
    """Settings sub-menu. Returns True when the user asked to quit."""
    while True:
        console.print(
            Panel(
                escape(settings_summary(settings)),
                title="Parameters",
                border_style="cyan",
            )
        )

        console.print("[bold yellow]What do you want to change?[/bold yellow]")
        console.print("[1]  Model (provider, name, API key)")
        console.print("[2]  Search backend (DuckDuckGo / Tavily + API key)")
        console.print("[3]  Research options (sources, planner, ranking)")
        console.print("[b]  Back to the main menu")

        answer = Prompt.ask(
            "[bold green]❯ Parameters[/bold green] [dim](b = back)[/dim]",
            default="b",
        )

        if is_quit_like(answer):
            return True

        choice = answer.strip().lower()

        if choice in {"b", "back", ""}:
            return False

        if choice == "1":
            choose_model(settings)
        elif choice == "2":
            choose_search(settings)
        elif choice == "3":
            research_options(settings)
        else:
            console.print("[dim]Unknown option - pick 1, 2, 3 or b.[/dim]")


def run_research(settings: Settings) -> bool:
    """Ask one question and print the answer.

    Returns True when the user asked to quit the whole application.
    """
    question = Prompt.ask(
        "\n[bold green]❯ What do you want to research?[/bold green] "
        "[dim](b = back, q or Ctrl+Q = quit)[/dim]",
        default="b",
    )

    if is_quit_like(question):
        return True

    if question.strip().lower() in {"b", "back", ""}:
        return False

    app = create_graph(
        provider=settings.provider,
        model=settings.model,
        api_key=settings.api_key,
        search_provider=settings.search_provider,
        search_api_key=settings.search_api_key,
        max_sources=settings.max_sources,
        planner=settings.planner,
        max_queries=DEFAULT_MAX_QUERIES,
    )

    with console.status(
        "[bold cyan]Researching...[/bold cyan]",
        spinner="dots",
    ):
        result = app.invoke({
            "question": question,
        })

    console.print()

    console.print(
        Panel(
            Markdown(result["result"]),
            title="[bold green]Research Result[/bold green]",
            border_style="green",
        )
    )

    queries = result.get("queries") or []
    sources = result.get("sources") or []

    console.print()

    if queries:
        console.print(
            "[dim]Search queries:[/dim] "
            + " · ".join(escape(query) for query in queries)
        )

    if sources:
        print_sources(sources, result["result"])
    else:
        console.print(
            "[bold yellow]⚠ No web sources found - the answer uses the "
            "model's own knowledge only.[/bold yellow]"
        )

    return False


def main() -> None:
    settings = Settings()

    console.print(
        Panel(
            "[bold cyan]Personal Research Assistant[/bold cyan]\n"
            "[dim]LangGraph + Ollama / API[/dim]",
            title="Research CLI",
            border_style="cyan",
        )
    )

    try:
        choose_model(settings)
        choose_search(settings)

        while True:
            console.print(f"\n[dim]{escape(settings_summary(settings))}[/dim]")
            console.print("[bold yellow]Main menu:[/bold yellow]")
            console.print("[1]  New research question")
            console.print("[2]  Parameters (model, API keys, search, ranking)")
            console.print("[q]  Quit [dim](Ctrl+Q also works)[/dim]")

            key = read_menu_key()
            action = parse_menu_key(key)

            if action == "quit":
                break

            if action == "research":
                if run_research(settings):
                    break

            elif action == "parameters":
                if parameters_menu(settings):
                    break

            else:
                console.print("[dim]Unknown key - press 1, 2 or q.[/dim]")

    except KeyboardInterrupt:
        console.print()

    console.print("\n[bold cyan]👋 Bye![/bold cyan]")


if __name__ == "__main__":
    main()