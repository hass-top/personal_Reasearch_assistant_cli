from langchain_openai import ChatOpenAI

class OpenAIProvider:
    def __init__(self, model:str , api_key: str):
        self.llm= ChatOpenAI(
            model=model,
            temperature=0,
            api_key=api_key,
        )
    def invoke(self,prompt: str):
        return self.llm.invoke(prompt)