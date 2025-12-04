from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    groq_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")

    data_path: Path = Field(default=Path("./data/anime_with_synopsis.csv"))
    vectorstore_collection: str = Field(default="anime")
    agent_model: str = Field(default="groq:llama-3.1-8b-instant")
