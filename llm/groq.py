from langchain_groq import ChatGroq

class Groqprovider:
    def __init__(self, model: str, api_key: str):
        self.llm = ChatGroq(
            model=model,
            temperature=0,
            api_key=api_key,
        )

    def invoke(self, prompt: str):
        return self.llm.invoke(prompt)