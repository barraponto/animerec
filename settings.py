from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict()

    groq_api_key: str = Field(default="")
    huggingface_api_key: str = Field(default="")
