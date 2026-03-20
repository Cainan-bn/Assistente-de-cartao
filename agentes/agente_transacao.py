"""
Agente especializado em consulta de transações e gastos do cartão.

Permite ao cliente consultar compras recentes, filtrar por período,
categoria e valor — com respostas inteligentes e sumarizadas.
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
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)

_PROMPT_TRANSACOES = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um assistente especializado em histórico de transações de cartão.
        Use as ferramentas para buscar os dados reais antes de responder.
        Sumarize os gastos de forma clara: agrupe por categoria quando possível.
        Formate valores em reais (R$). Ordene do mais recente ao mais antigo.
        Data atual: {data_atual} | ID do cliente: {id_cliente}
        """,
    ),
    MessagesPlaceholder(variable_name="mensagens"),
])


class AgenteTransacoes:
    """
    Agente que responde consultas sobre lançamentos e histórico de transações.

    Suporta filtros por período, categoria e valor, com sumarização
    inteligente dos gastos via LLM.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._api_cartao = ClienteCartaoAPI()
        self._ferramentas = self._registrar_ferramentas()
        self._llm_com_ferramentas = llm.bind_tools(self._ferramentas)

    def _registrar_ferramentas(self) -> list:
        api = self._api_cartao

        @tool
        async def buscar_transacoes_recentes(id_cliente: str, quantidade: int = 10) -> list[dict]:
            """Busca as transações mais recentes do cartão do cliente."""
            transacoes = await api.buscar_transacoes(id_cliente, quantidade=quantidade)
            return [t.model_dump() for t in transacoes]

        @tool
        async def buscar_transacoes_por_periodo(
            id_cliente: str,
            data_inicio: str,
            data_fim: str,
        ) -> list[dict]:
            """
            Busca transações em um período específico.

            Args:
                data_inicio: Data de início no formato YYYY-MM-DD.
                data_fim: Data de fim no formato YYYY-MM-DD.
            """
            transacoes = await api.buscar_transacoes_periodo(id_cliente, data_inicio, data_fim)
            return [t.model_dump() for t in transacoes]

        return [buscar_transacoes_recentes, buscar_transacoes_por_periodo]

    async def processar(self, mensagens: list[BaseMessage], id_cliente: str) -> RespostaAgente:
        """
        Processa a consulta de transações com tool calling iterativo.

        Args:
            mensagens: Histórico completo de mensagens da conversa.
            id_cliente: Identificador único do cliente.

        Returns:
            RespostaAgente com o texto final gerado pelo LLM.
        """
        mensagens_com_contexto = await _PROMPT_TRANSACOES.aformat_messages(
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
            intencao=IntencaoUsuario.CONSULTA_TRANSACOES,
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
                    ToolMessage(content="Erro ao buscar transações.", tool_call_id=chamada["id"])
                )
        return resultados
