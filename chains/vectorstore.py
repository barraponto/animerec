from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from settings import Settings

settings: Settings = Settings()


embedder: HuggingFaceEmbeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vectorstore: Chroma = Chroma(
    embedding_function=embedder,
    collection_name=settings.vectorstore_collection,
)
