import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import Settings


class IngestChain:
    required_columns: set[str] = {"name", "genres", "synopsis"}

    def __init__(self, settings: Settings, vectorstore: Chroma):
        self.settings: Settings = settings
        self.vectorstore: Chroma = vectorstore
        self.splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=64, chunk_overlap=16
            )
        )

    def _ensure_data(self, data: pd.DataFrame | None = None):
        if data is None:
            data = pd.read_csv(self.settings.data_path, on_bad_lines="skip")
        return data

    def _normalize_data(self, data: pd.DataFrame):
        data.columns = data.columns.str.lower()
        if missing_columns := self.required_columns - set(data.columns):
            raise ValueError(f"Missing required columns: {missing_columns!r}")
        return data.dropna()

    def _get_document(self, row: pd.Series) -> Document:
        metadata: dict[str, str] = row.drop("synopsis").to_dict()
        text = str(row["synopsis"])
        return Document(page_content=text, metadata=metadata)

    def ingest(self, data: pd.DataFrame | None = None):
        data = self._ensure_data(data)
        data = self._normalize_data(data)

        documents = [self._get_document(row) for _, row in data.iterrows()]

        self.vectorstore.add_documents(documents)
