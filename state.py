from pydantic import BaseModel,Field

class State(BaseModel):
    question: str = Field(description="the user research question")
    queries: list[str] = Field(
        default_factory=list,
        description="the search queries used to find the sources",
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="the web sources found for the question",
    )
    context: str = Field(default="", description="the sources formatted for the prompt")
    result: str = Field(default="", description="the research result")
