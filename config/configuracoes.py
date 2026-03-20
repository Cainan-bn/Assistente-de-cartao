"""
Configurações da aplicação carregadas via variáveis de ambiente.

Utiliza pydantic-settings para validação tipada e segura
de todas as configurações necessárias ao runtime.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Configuracoes(BaseSettings):
    """Configurações centralizadas do assistente de cartões."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    openai_api_key: SecretStr = Field(alias="OPENAI_API_KEY")
    modelo_llm: str = Field(default="gpt-4o", alias="MODELO_LLM")
    url_api_cartoes: str = Field(
        default="https://api.cartoes.getronics.com",
        alias="URL_API_CARTOES",
    )
    redis_url: str = Field(
        default="redis://redis:6379",
        alias="REDIS_URL",
    )
    origens_permitidas: list[str] = Field(
        default=["https://app.getronics.com"],
        alias="ORIGENS_PERMITIDAS",
    )
    nivel_log: str = Field(default="INFO", alias="NIVEL_LOG")