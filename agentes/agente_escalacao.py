"""
Agente de escalação para situações que requerem atendimento humano.

Detecta automaticamente quando o assistente não tem confiança suficiente
para responder e aciona o fluxo de transferência para um atendente.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from modelos.estado import IntencaoUsuario
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)

_CONFIANCA_MINIMA_PARA_RESPOSTA = 0.6
_LIMITE_TENTATIVAS_SEM_RESOLUCAO = 3


class MotivoEscalacao(StrEnum):
    """Motivos possíveis para escalação ao atendimento humano."""

    BAIXA_CONFIANCA = "BAIXA_CONFIANCA"
    SOLICITACAO_EXPLICITA = "SOLICITACAO_EXPLICITA"
    MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO = "MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO"
    ASSUNTO_FORA_DO_ESCOPO = "ASSUNTO_FORA_DO_ESCOPO"
    POSSIVEL_FRAUDE = "POSSIVEL_FRAUDE"


class DecisaoEscalacao(BaseModel):
    """Resultado da análise de necessidade de escalação."""

    deve_escalar: bool
    motivo: MotivoEscalacao | None = None
    mensagem_transicao: str = Field(default="")
    prioridade: int = Field(default=3, ge=1, le=5, description="1=urgente, 5=baixa")


_PROMPT_ANALISE_ESCALACAO = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um analisador de necessidade de escalação para atendimento humano.
        Analise a conversa e determine se o cliente precisa ser transferido para um atendente.

        SEMPRE escale quando detectar:
        - Suspeita de fraude ou transação não reconhecida
        - Cliente expressamente pede para falar com humano
        - Problema técnico com o cartão (bloqueio, perda, roubo)
        - Reclamação ou insatisfação grave
        - Contestação de cobrança
        - 3 ou mais turnos sem resolver o problema

        NÃO escale para:
        - Consultas simples de fatura, limite ou saldo
        - Dúvidas gerais sobre o cartão
        - Informações sobre benefícios
        """,
    ),
    MessagesPlaceholder(variable_name="mensagens"),
    ("human", "Com base na conversa acima, devo escalar para atendimento humano?"),
])


class AgenteEscalacao:
    """
    Detecta quando a conversa deve ser transferida para um atendente humano.

    Age como guardião do fluxo de atendimento, garantindo que situações
    sensíveis como fraude ou insatisfação sejam tratadas por humanos.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        llm_estruturado = llm.with_structured_output(DecisaoEscalacao)
        self._cadeia = _PROMPT_ANALISE_ESCALACAO | llm_estruturado

    async def analisar(
        self,
        mensagens: list[BaseMessage],
        resposta_atual: RespostaAgente,
        tentativas_sem_resolucao: int = 0,
    ) -> DecisaoEscalacao:
        """
        Analisa se a conversa deve ser escalada para atendimento humano.

        Args:
            mensagens: Histórico completo da conversa.
            resposta_atual: Última resposta gerada pelo assistente.
            tentativas_sem_resolucao: Número de turnos sem resolver o problema.

        Returns:
            DecisaoEscalacao com flag de escalação e contexto para o atendente.
        """
        if resposta_atual.confianca < _CONFIANCA_MINIMA_PARA_RESPOSTA:
            logger.info(
                "Escalação por baixa confiança: %.2f",
                resposta_atual.confianca,
            )
            return DecisaoEscalacao(
                deve_escalar=True,
                motivo=MotivoEscalacao.BAIXA_CONFIANCA,
                mensagem_transicao=self._gerar_mensagem_transicao(
                    MotivoEscalacao.BAIXA_CONFIANCA
                ),
                prioridade=2,
            )

        if tentativas_sem_resolucao >= _LIMITE_TENTATIVAS_SEM_RESOLUCAO:
            logger.info(
                "Escalação por múltiplas tentativas sem resolução: %d",
                tentativas_sem_resolucao,
            )
            return DecisaoEscalacao(
                deve_escalar=True,
                motivo=MotivoEscalacao.MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO,
                mensagem_transicao=self._gerar_mensagem_transicao(
                    MotivoEscalacao.MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO
                ),
                prioridade=2,
            )

        try:
            decisao: DecisaoEscalacao = await self._cadeia.ainvoke({"mensagens": mensagens})
            if decisao.deve_escalar and not decisao.mensagem_transicao:
                decisao = decisao.model_copy(
                    update={"mensagem_transicao": self._gerar_mensagem_transicao(decisao.motivo)}
                )
            return decisao
        except Exception:
            logger.exception("Falha na análise de escalação. Continuando sem escalar.")
            return DecisaoEscalacao(deve_escalar=False)

    def _gerar_mensagem_transicao(self, motivo: MotivoEscalacao | None) -> str:
        mensagens = {
            MotivoEscalacao.BAIXA_CONFIANCA: (
                "Vou transferir você para um de nossos especialistas que poderá "
                "ajudar melhor com sua solicitação. Um momento, por favor."
            ),
            MotivoEscalacao.SOLICITACAO_EXPLICITA: (
                "Claro! Vou conectar você com um atendente agora. "
                "Aguarde um instante."
            ),
            MotivoEscalacao.MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO: (
                "Percebo que ainda não conseguimos resolver sua solicitação. "
                "Vou transferir para um especialista que poderá dar atenção personalizada."
            ),
            MotivoEscalacao.POSSIVEL_FRAUDE: (
                "Para sua segurança, vou transferir você imediatamente para nossa "
                "equipe de segurança. Não compartilhe sua senha com ninguém."
            ),
            MotivoEscalacao.ASSUNTO_FORA_DO_ESCOPO: (
                "Essa solicitação requer atendimento especializado. "
                "Vou conectar você com o setor responsável."
            ),
        }
        return mensagens.get(motivo, "Transferindo para atendimento humano.")
