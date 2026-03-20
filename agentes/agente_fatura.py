"""
Agente especializado em consulta de faturas do cartão.

Utiliza LangChain com tool calling para buscar dados reais de fatura
via API REST e gerar respostas contextualizadas ao cliente.
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
from modelos.fatura import Fatura
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)

_PROMPT_FATURA = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um assistente especializado em faturas de cartão de crédito.
        Responda de forma clara, empática e objetiva.
        Use as ferramentas disponíveis para buscar dados reais antes de responder.
        Formate valores monetários em reais (R$) com duas casas decimais.
        Data atual: {data_atual}
        ID do cliente: {id_cliente}
        """,
    ),
    MessagesPlaceholder(variable_name="mensagens"),
])


class AgenteConsultaFatura:
    """
    Agente responsável por consultas de fatura usando tool calling.

    Conecta-se à API de cartões para buscar dados reais e gera
    respostas contextualizadas com base no histórico da conversa.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._api_cartao = ClienteCartaoAPI()
        self._ferramentas = self._registrar_ferramentas()
        self._llm_com_ferramentas = llm.bind_tools(self._ferramentas)

    def _registrar_ferramentas(self) -> list:
        api = self._api_cartao

        @tool
        async def buscar_fatura_atual(id_cliente: str) -> dict:
            """Busca os dados da fatura atual do cliente, incluindo valor e vencimento."""
            fatura: Fatura = await api.buscar_fatura_atual(id_cliente)
            return fatura.model_dump()

        @tool
        async def buscar_fatura_por_mes(id_cliente: str, mes: int, ano: int) -> dict:
            """Busca os dados de uma fatura de um mês e ano específicos."""
            fatura: Fatura = await api.buscar_fatura_por_mes(id_cliente, mes, ano)
            return fatura.model_dump()

        return [buscar_fatura_atual, buscar_fatura_por_mes]

    async def processar(self, mensagens: list[BaseMessage], id_cliente: str) -> RespostaAgente:
        """
        Processa a consulta de fatura usando tool calling iterativo.

        Args:
            mensagens: Histórico completo de mensagens da conversa.
            id_cliente: Identificador único do cliente.

        Returns:
            RespostaAgente com o texto final gerado pelo LLM.
        """
        mensagens_com_contexto = await _PROMPT_FATURA.aformat_messages(
            mensagens=mensagens,
            data_atual=datetime.now().strftime("%d/%m/%Y"),
            id_cliente=id_cliente,
        )

        resposta = await self._llm_com_ferramentas.ainvoke(mensagens_com_contexto)

        while resposta.tool_calls:
            mensagens_com_contexto.append(resposta)
            resultados_ferramentas = await self._executar_ferramentas(resposta.tool_calls)
            mensagens_com_contexto.extend(resultados_ferramentas)
            resposta = await self._llm_com_ferramentas.ainvoke(mensagens_com_contexto)

        return RespostaAgente(
            texto=resposta.content,
            intencao=IntencaoUsuario.CONSULTA_FATURA,
            confianca=1.0,
        )

    async def _executar_ferramentas(self, chamadas_ferramenta: list) -> list:
        from langchain_core.messages import ToolMessage

        resultados = []
        for chamada in chamadas_ferramenta:
            ferramenta = next(
                (f for f in self._ferramentas if f.name == chamada["name"]),
                None,
            )
            if not ferramenta:
                logger.warning("Ferramenta não encontrada: %s", chamada["name"])
                continue

            try:
                resultado = await ferramenta.ainvoke(chamada["args"])
                resultados.append(ToolMessage(content=str(resultado), tool_call_id=chamada["id"]))
            except Exception:
                logger.exception("Erro ao executar ferramenta %s", chamada["name"])
                resultados.append(
                    ToolMessage(
                        content="Erro ao buscar dados. Informe ao cliente que houve uma instabilidade.",
                        tool_call_id=chamada["id"],
                    )
                )

        return resultados
