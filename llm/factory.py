from .ollama import OllamaProvider
from .openai import OpenAIProvider
from .groq import Groqprovider
def create_llm(provider:str , model:str , api_key:str | None = None):
    if provider == "ollama":
        return OllamaProvider(model)

    if provider == "openai":
        if api_key is None:
            raise ValueError("OpenAi API key required")

        return OpenAIProvider(
            model=model,
            api_key=api_key,
        )
    if provider == "groq":
        if api_key is None:
            raise ValueError("Groq API key required")
        return Groqprovider(
            model=model,
            api_key=api_key,
        )
    raise ValueError(f"unsupported provider: {provider}"
    )