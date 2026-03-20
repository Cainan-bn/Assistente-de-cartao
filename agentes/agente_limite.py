"""
Agente especializado em consulta de limite de crédito.

Usa tool calling para buscar dados reais de limite e gerar
respostas contextualizadas com cálculos de disponibilidade.
"""

from __future__ import annotations

import logging
from datetime import datetime

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from ferramentas.cliente_cartao_api import ClienteCartaoAPI
from modelos.estado import IntencaoUsuario
from modelos.limite import InformacaoLimite
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)

_PROMPT_LIMITE = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um assistente especializado em limites de cartão de crédito.
        Use as ferramentas disponíveis para buscar dados reais antes de responder.
        Formate valores em reais (R$). Se o limite disponível estiver abaixo de 20%
        do total, avise gentilmente o cliente.
        Data atual: {data_atual} | ID do cliente: {id_cliente}
        """,
    ),
    MessagesPlaceholder(variable_name="mensagens"),
])


class AgenteConsultaLimite:
    """
    Agente que responde consultas sobre limite de crédito disponível e total.

    Calcula automaticamente percentuais de uso e emite alertas
    quando o limite disponível está baixo.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._api_cartao = ClienteCartaoAPI()
        self._ferramentas = self._registrar_ferramentas()
        self._llm_com_ferramentas = llm.bind_tools(self._ferramentas)

    def _registrar_ferramentas(self) -> list:
        api = self._api_cartao

        @tool
        async def buscar_limite_cliente(id_cliente: str) -> dict:
            """Busca o limite total e disponível do cartão do cliente."""
            limite: InformacaoLimite = await api.buscar_limite(id_cliente)
            percentual_uso = (
                (limite.limite_total - limite.limite_disponivel) / limite.limite_total * 100
                if limite.limite_total > 0
                else 0
            )
            return {**limite.model_dump(), "percentual_uso": round(percentual_uso, 1)}

        return [buscar_limite_cliente]

    async def processar(self, mensagens: list[BaseMessage], id_cliente: str) -> RespostaAgente:
        mensagens_com_contexto = await _PROMPT_LIMITE.aformat_messages(
            mensagens=mensagens,
            data_atual=datetime.now().strftime("%d/%m/%Y"),
            id_cliente=id_cliente,
        )
        resposta = await self._llm_com_ferramentas.ainvoke(mensagens_com_contexto)

        while resposta.tool_calls:
            mensagens_com_contexto.append(resposta)
            resultados = await self._executar_ferramentas(resposta.tool_calls)
            mensagens_com_contexto.extend(resultados)
            resposta = await self._llm_com_ferramentas.ainvoke(mensagens_com_contexto)

        return RespostaAgente(
            texto=resposta.content,
            intencao=IntencaoUsuario.CONSULTA_LIMITE,
            confianca=1.0,
        )

    async def _executar_ferramentas(self, chamadas: list) -> list:
        from langchain_core.messages import ToolMessage

        resultados = []
        for chamada in chamadas:
            ferramenta = next((f for f in self._ferramentas if f.name == chamada["name"]), None)
            if not ferramenta:
                continue
            try:
                resultado = await ferramenta.ainvoke(chamada["args"])
                resultados.append(ToolMessage(content=str(resultado), tool_call_id=chamada["id"]))
            except Exception:
                logger.exception("Erro ao executar ferramenta %s", chamada["name"])
                resultados.append(
                    ToolMessage(content="Erro ao buscar limite.", tool_call_id=chamada["id"])
                )
        return resultados
