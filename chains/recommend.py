from langchain.agents import create_agent
from langchain.tools import BaseTool, tool
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from settings import Settings


def store_retriever(vectorstore: Chroma) -> BaseTool:
    @tool(response_format="content_and_artifact")
    def retrieve_context(input: str) -> tuple[str, list[Document]]:
        """
        Use this tool to find the most relevant anime to the user's query.
        """
        documents = vectorstore.similarity_search(input, k=3)
        serialized = [
            f"{doc.metadata.get('name', 'untitled')}: {doc.page_content}"
            for doc in documents
        ]
        return "\n".join(serialized), documents

    return retrieve_context


class RecommendChain:
    def __init__(self, settings: Settings, vectorstore: Chroma):
        self.settings: Settings = settings
        self.vectorstore: Chroma = vectorstore
        self.agent = create_agent(
            model=self.settings.agent_model,
            tools=[store_retriever(self.vectorstore)],
            system_prompt="""
            You are a helpful assistant that recommends anime to users.
            You are given a user query and should use the retriever tool to find the most relevant anime.
            You need to recommend the best anime to the user.
            """,
        )

    def ask(self, prompt: str) -> list[AIMessage | HumanMessage]:
        output = self.agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            }
        )
        return [
            message
            for message in output["messages"]
            if message.content and isinstance(message, (AIMessage, HumanMessage))
        ]
