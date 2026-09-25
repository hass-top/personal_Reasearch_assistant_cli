from pydantic import BaseModel,Field

class State(BaseModel):
    question: str = Field(description="the user research question")
    subject: str = Field(
        default="",
        description="what the question is about (english, cybersecurity, general)",
    )
    topic: str = Field(
        default="",
        description="the main thing the user wants to know about",
    )
    intent: str = Field(
        default="",
        description="what the user wants to do (learn, practice, verify, research)",
    )
    task: str = Field(
        default="",
        description="the specific operation requested (explain, solve, define...)",
    )
    ambiguous: bool = Field(
        default=False,
        description=(
            "true only when different interpretations would lead to "
            "substantially different searches"
        ),
    )
    clarification: str = Field(
        default="",
        description="a short question asked only when the request is ambiguous",
    )
    clarification_options: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 mutually exclusive readings of an ambiguous request, "
            "each one usable directly as a search question"
        ),
    )
    queries: list[str] = Field(
        default_factory=list,
        description="the search queries used to find the sources",
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="the web sources found for the question",
    )
    context: str = Field(default="", description="the sources formatted for the prompt")
    evidence: list[dict] = Field(
        default_factory=list,
        description=(
            "the real text of the best sources, fetched and trimmed, so the "
            "answer is written from what the pages actually say"
        ),
    )
    result: str = Field(default="", description="the research result")


class Understanding(BaseModel):
    """The JSON contract the understanding node asks the model for.

    Passing this model to ``with_structured_output`` gives the provider the
    schema, so the reply is a valid object instead of text that has to be
    parsed. ``state.Understanding`` and the prompt below must stay in sync.
    """

    subject: str = Field(
        default="",
        description=(
            "what the question is about: english, cybersecurity, or general"
        ),
    )
    topic: str = Field(
        default="",
        description="the main thing the user wants to know about, a few words",
    )
    intent: str = Field(
        default="",
        description=(
            "what the user wants to do: learn, practice, verify or research"
        ),
    )
    task: str = Field(
        default="",
        description="the specific operation requested (explain, solve, define...)",
    )
    ambiguous: bool = Field(
        default=False,
        description=(
            "true only when different interpretations would lead to "
            "substantially different searches"
        ),
    )
    clarification: str = Field(
        default="",
        description="a short question asked only when ambiguous is true",
    )
    clarification_options: list[str] = Field(
        default_factory=list,
        description=(
            "2-4 mutually exclusive readings of an ambiguous request, each "
            "one a short self-contained restatement usable as a search "
            "question; empty when the request is not ambiguous"
        ),
    )
