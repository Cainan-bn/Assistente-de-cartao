"""
Testes unitários do agente de escalação.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentes.agente_escalacao import (
    AgenteEscalacao,
    DecisaoEscalacao,
    MotivoEscalacao,
)
from modelos.estado import IntencaoUsuario
from modelos.resposta import RespostaAgente


@pytest.fixture
def agente_escalacao() -> AgenteEscalacao:
    return AgenteEscalacao(llm=MagicMock())


def _resposta(confianca: float = 1.0) -> RespostaAgente:
    return RespostaAgente(
        texto="resposta teste",
        intencao=IntencaoUsuario.CONSULTA_FATURA,
        confianca=confianca,
    )


class TestAgenteEscalacao:
    """Testes do agente que detecta necessidade de atendimento humano."""

    @pytest.mark.asyncio
    async def test_escala_por_baixa_confianca(self, agente_escalacao: AgenteEscalacao) -> None:
        decisao = await agente_escalacao.analisar(
            mensagens=[],
            resposta_atual=_resposta(confianca=0.3),
        )

        assert decisao.deve_escalar is True
        assert decisao.motivo == MotivoEscalacao.BAIXA_CONFIANCA

    @pytest.mark.asyncio
    async def test_escala_por_multiplas_tentativas(self, agente_escalacao: AgenteEscalacao) -> None:
        decisao = await agente_escalacao.analisar(
            mensagens=[],
            resposta_atual=_resposta(confianca=0.9),
            tentativas_sem_resolucao=3,
        )

        assert decisao.deve_escalar is True
        assert decisao.motivo == MotivoEscalacao.MULTIPLAS_TENTATIVAS_SEM_RESOLUCAO

    @pytest.mark.asyncio
    async def test_nao_escala_para_consulta_simples(self, agente_escalacao: AgenteEscalacao) -> None:
        decisao_mock = DecisaoEscalacao(deve_escalar=False)

        with patch.object(agente_escalacao, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=decisao_mock)
            decisao = await agente_escalacao.analisar(
                mensagens=[],
                resposta_atual=_resposta(confianca=0.95),
            )

        assert decisao.deve_escalar is False

    @pytest.mark.asyncio
    async def test_escalacao_tem_mensagem_de_transicao(self, agente_escalacao: AgenteEscalacao) -> None:
        decisao = await agente_escalacao.analisar(
            mensagens=[],
            resposta_atual=_resposta(confianca=0.2),
        )

        assert len(decisao.mensagem_transicao) > 0

    @pytest.mark.asyncio
    async def test_falha_no_llm_nao_escala(self, agente_escalacao: AgenteEscalacao) -> None:
        with patch.object(agente_escalacao, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(side_effect=Exception("LLM timeout"))
            decisao = await agente_escalacao.analisar(
                mensagens=[],
                resposta_atual=_resposta(confianca=0.8),
            )

        assert decisao.deve_escalar is False
