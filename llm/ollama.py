from langchain_ollama import ChatOllama

class OllamaProvider:
    def __init__(self,model:str):
        self.llm=ChatOllama(
            model=model,
            temperature=0
        )

    def invoke(self , prompt: str):
        return self.llm.invoke(prompt)