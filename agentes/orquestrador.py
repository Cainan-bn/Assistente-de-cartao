"""
Orquestrador principal do assistente de cartões.

Responsável por coordenar o grafo de agentes via LangGraph,
roteando intenções do usuário para os agentes especializados corretos.
"""

from __future__ import annotations

import logging
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agentes.agente_fatura import AgenteConsultaFatura
from agentes.agente_limite import AgenteConsultaLimite
from agentes.agente_transacao import AgenteTransacoes
from agentes.classificador_intencao import ClassificadorIntencao
from modelos.estado import EstadoConversa, IntencaoUsuario
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)


class EstadoGrafo(TypedDict):
    """Estado compartilhado entre os nós do grafo LangGraph."""

    mensagens: Annotated[list[BaseMessage], add_messages]
    intencao: IntencaoUsuario | None
    resposta_final: RespostaAgente | None
    id_sessao: str
    id_cliente: str


class OrquestradorCartoes:
    """
    Orquestra o fluxo de conversação do assistente de cartões.

    Utiliza LangGraph para construir um grafo de estados que roteia
    mensagens para agentes especializados com base na intenção detectada.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm
        self._classificador = ClassificadorIntencao(llm)
        self._agente_fatura = AgenteConsultaFatura(llm)
        self._agente_limite = AgenteConsultaLimite(llm)
        self._agente_transacao = AgenteTransacoes(llm)
        self._grafo = self._construir_grafo()

    def _construir_grafo(self) -> StateGraph:
        grafo = StateGraph(EstadoGrafo)

        grafo.add_node("classificar", self._no_classificar_intencao)
        grafo.add_node("consultar_fatura", self._no_consultar_fatura)
        grafo.add_node("consultar_limite", self._no_consultar_limite)
        grafo.add_node("consultar_transacoes", self._no_consultar_transacoes)
        grafo.add_node("resposta_generica", self._no_resposta_generica)

        grafo.add_edge(START, "classificar")

        grafo.add_conditional_edges(
            "classificar",
            self._rotear_por_intencao,
            {
                IntencaoUsuario.CONSULTA_FATURA: "consultar_fatura",
                IntencaoUsuario.CONSULTA_LIMITE: "consultar_limite",
                IntencaoUsuario.CONSULTA_TRANSACOES: "consultar_transacoes",
                IntencaoUsuario.GENERICA: "resposta_generica",
            },
        )

        for no_agente in ["consultar_fatura", "consultar_limite", "consultar_transacoes", "resposta_generica"]:
            grafo.add_edge(no_agente, END)

        return grafo.compile()

    async def _no_classificar_intencao(self, estado: EstadoGrafo) -> dict:
        ultima_mensagem = estado["mensagens"][-1].content
        intencao = await self._classificador.classificar(ultima_mensagem)
        logger.info("Intenção classificada: %s para sessão %s", intencao, estado["id_sessao"])
        return {"intencao": intencao}

    def _rotear_por_intencao(self, estado: EstadoGrafo) -> IntencaoUsuario:
        return estado.get("intencao", IntencaoUsuario.GENERICA)

    async def _no_consultar_fatura(self, estado: EstadoGrafo) -> dict:
        resposta = await self._agente_fatura.processar(
            mensagens=estado["mensagens"],
            id_cliente=estado["id_cliente"],
        )
        return {"resposta_final": resposta, "mensagens": [AIMessage(content=resposta.texto)]}

    async def _no_consultar_limite(self, estado: EstadoGrafo) -> dict:
        resposta = await self._agente_limite.processar(
            mensagens=estado["mensagens"],
            id_cliente=estado["id_cliente"],
        )
        return {"resposta_final": resposta, "mensagens": [AIMessage(content=resposta.texto)]}

    async def _no_consultar_transacoes(self, estado: EstadoGrafo) -> dict:
        resposta = await self._agente_transacao.processar(
            mensagens=estado["mensagens"],
            id_cliente=estado["id_cliente"],
        )
        return {"resposta_final": resposta, "mensagens": [AIMessage(content=resposta.texto)]}

    async def _no_resposta_generica(self, estado: EstadoGrafo) -> dict:
        ultima_mensagem = estado["mensagens"][-1].content
        resposta_llm = await self._llm.ainvoke(
            [HumanMessage(content=ultima_mensagem)]
        )
        resposta = RespostaAgente(
            texto=resposta_llm.content,
            intencao=IntencaoUsuario.GENERICA,
            confianca=1.0,
        )
        return {"resposta_final": resposta, "mensagens": [AIMessage(content=resposta.texto)]}

    async def processar_mensagem(self, estado: EstadoConversa, mensagem: str) -> RespostaAgente:
        """
        Ponto de entrada principal para processar uma mensagem do usuário.

        Args:
            estado: Estado atual da conversa com histórico e dados do cliente.
            mensagem: Texto enviado pelo usuário.

        Returns:
            RespostaAgente com o texto e metadados da resposta.
        """
        estado_grafo: EstadoGrafo = {
            "mensagens": estado.historico_mensagens + [HumanMessage(content=mensagem)],
            "intencao": None,
            "resposta_final": None,
            "id_sessao": estado.id_sessao,
            "id_cliente": estado.id_cliente,
        }

        resultado = await self._grafo.ainvoke(estado_grafo)
        return resultado["resposta_final"]
