"""
API REST do assistente de cartões via FastAPI.

Expõe endpoints para interação com o agente de IA,
com autenticação, validação e logging estruturado.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from agentes.orquestrador import OrquestradorCartoes
from config.configuracoes import Configuracoes
from modelos.estado import EstadoConversa
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)
configuracoes = Configuracoes()


class RequisicaoMensagem(BaseModel):
    """Schema da requisição de envio de mensagem ao assistente."""

    mensagem: str = Field(min_length=1, max_length=2000)
    id_sessao: str | None = Field(default=None)
    id_cliente: str


class RespostaAPI(BaseModel):
    """Schema da resposta da API do assistente."""

    id_sessao: str
    resposta: RespostaAgente


_orquestrador: OrquestradorCartoes | None = None


@asynccontextmanager
async def vida_util_app(app: FastAPI):
    """Inicializa e finaliza recursos compartilhados da aplicação."""
    global _orquestrador
    llm = ChatOpenAI(
        model=configuracoes.modelo_llm,
        temperature=0,
        api_key=configuracoes.openai_api_key,
    )
    _orquestrador = OrquestradorCartoes(llm)
    logger.info("Orquestrador inicializado com modelo %s", configuracoes.modelo_llm)
    yield
    logger.info("Encerrando aplicação")


app = FastAPI(
    title="Assistente de Cartões IA –",
    description="Agente de IA para suporte inteligente em operações de cartão de crédito.",
    version="1.0.0",
    lifespan=vida_util_app,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=configuracoes.origens_permitidas,
    allow_methods=["POST"],
    allow_headers=["*"],
)


def obter_orquestrador() -> OrquestradorCartoes:
    if _orquestrador is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Serviço de IA não está disponível no momento.",
        )
    return _orquestrador


@app.post(
    "/v1/mensagem",
    response_model=RespostaAPI,
    status_code=status.HTTP_200_OK,
    summary="Envia uma mensagem ao assistente de cartões",
)
async def enviar_mensagem(
    requisicao: RequisicaoMensagem,
    orquestrador: OrquestradorCartoes = Depends(obter_orquestrador),
    x_correlation_id: str | None = Header(default=None),
) -> RespostaAPI:
    """
    Processa uma mensagem do usuário e retorna a resposta do assistente.

    O assistente identifica automaticamente a intenção e direciona para
    o agente especializado mais adequado (fatura, limite, transações, etc.).
    """
    id_sessao = requisicao.id_sessao or str(uuid.uuid4())
    correlation_id = x_correlation_id or id_sessao

    logger.info(
        "Nova mensagem recebida | sessão=%s | cliente=%s | correlation=%s",
        id_sessao,
        requisicao.id_cliente,
        correlation_id,
    )

    estado = EstadoConversa(
        id_sessao=id_sessao,
        id_cliente=requisicao.id_cliente,
    )

    try:
        resposta = await orquestrador.processar_mensagem(estado, requisicao.mensagem)
    except Exception:
        logger.exception("Erro ao processar mensagem | sessão=%s", id_sessao)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Erro interno ao processar sua solicitação. Tente novamente.",
        )

    return RespostaAPI(id_sessao=id_sessao, resposta=resposta)


@app.get("/health", include_in_schema=False)
async def verificar_saude() -> dict:
    """Endpoint de health check para orquestradores de container."""
    return {"status": "ok", "servico": "assistente-cartoes"}
