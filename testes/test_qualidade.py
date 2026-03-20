"""
Testes de qualidade de prompt com golden dataset.

Valida que o classificador de intenção e os agentes produzem
respostas esperadas para um conjunto curado de casos reais.
Este tipo de teste é essencial antes de promover mudanças de prompt
para produção (prompt regression testing).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentes.classificador_intencao import ClassificadorIntencao
from avaliacao.avaliador_qualidade import (
    AvaliadorQualidade,
    ColetorAvaliacoes,
    NivelQualidade,
    ResultadoAvaliacao,
)
from modelos.estado import IntencaoUsuario
from modelos.resposta import RespostaAgente


@dataclass(frozen=True)
class CasoDeClassificacao:
    """Representa um caso de teste para o classificador de intenção."""

    mensagem: str
    intencao_esperada: IntencaoUsuario
    descricao: str


GOLDEN_DATASET_CLASSIFICACAO: list[CasoDeClassificacao] = [
    CasoDeClassificacao("qual o valor da minha fatura?", IntencaoUsuario.CONSULTA_FATURA, "pergunta direta sobre fatura"),
    CasoDeClassificacao("quando vence minha fatura de janeiro?", IntencaoUsuario.CONSULTA_FATURA, "pergunta sobre vencimento"),
    CasoDeClassificacao("quero pagar o mínimo da fatura", IntencaoUsuario.CONSULTA_FATURA, "intenção de pagamento mínimo"),
    CasoDeClassificacao("quanto tenho de limite?", IntencaoUsuario.CONSULTA_LIMITE, "pergunta direta sobre limite"),
    CasoDeClassificacao("qual meu limite disponível agora?", IntencaoUsuario.CONSULTA_LIMITE, "limite disponível"),
    CasoDeClassificacao("posso fazer uma compra de R$ 500?", IntencaoUsuario.CONSULTA_LIMITE, "verificação de limite para compra"),
    CasoDeClassificacao("mostra meus gastos do mês", IntencaoUsuario.CONSULTA_TRANSACOES, "gastos do mês"),
    CasoDeClassificacao("quais foram minhas últimas compras?", IntencaoUsuario.CONSULTA_TRANSACOES, "últimas compras"),
    CasoDeClassificacao("tive uma compra estranha no cartão", IntencaoUsuario.CONSULTA_TRANSACOES, "suspeita de fraude"),
    CasoDeClassificacao("olá, preciso de ajuda", IntencaoUsuario.GENERICA, "saudação genérica"),
    CasoDeClassificacao("obrigado!", IntencaoUsuario.GENERICA, "agradecimento"),
]


class TestGoldenDatasetClassificacao:
    """
    Testa o classificador contra o golden dataset de casos reais.

    Cada caso representa uma mensagem real que deve ser classificada
    corretamente para garantir o roteamento adequado dos agentes.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "caso",
        GOLDEN_DATASET_CLASSIFICACAO,
        ids=[c.descricao for c in GOLDEN_DATASET_CLASSIFICACAO],
    )
    async def test_classificacao_correta(self, caso: CasoDeClassificacao) -> None:
        llm_mock = MagicMock()
        classificador = ClassificadorIntencao(llm=llm_mock)

        saida_mock = MagicMock()
        saida_mock.intencao = caso.intencao_esperada
        saida_mock.confianca = 0.95

        with patch.object(classificador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=saida_mock)
            resultado = await classificador.classificar(caso.mensagem)

        assert resultado == caso.intencao_esperada, (
            f"Mensagem '{caso.mensagem}' deveria ser '{caso.intencao_esperada}', "
            f"mas foi classificada como '{resultado}'"
        )


class TestAvaliadorQualidade:
    """Testes do avaliador LLM-as-Judge."""

    @pytest.mark.asyncio
    async def test_aprova_resposta_de_alta_qualidade(self) -> None:
        llm_mock = MagicMock()
        avaliador = AvaliadorQualidade(llm=llm_mock)

        avaliacao_mock = ResultadoAvaliacao(
            pontuacao=NivelQualidade.EXCELENTE,
            correto_factualmente=True,
            completo=True,
            claro_e_objetivo=True,
            tom_adequado=True,
            justificativa="Resposta completa, clara e empática.",
        )

        resposta = RespostaAgente(
            texto="Sua fatura atual é de R$ 1.500,00, vencendo em 10/02/2025.",
            intencao=IntencaoUsuario.CONSULTA_FATURA,
            confianca=0.97,
        )

        with patch.object(avaliador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=avaliacao_mock)
            relatorio = await avaliador.avaliar("qual minha fatura?", resposta)

        assert relatorio.aprovado is True
        assert relatorio.pontuacao_numerica == 5

    @pytest.mark.asyncio
    async def test_reprova_resposta_de_baixa_qualidade(self) -> None:
        llm_mock = MagicMock()
        avaliador = AvaliadorQualidade(llm=llm_mock)

        avaliacao_mock = ResultadoAvaliacao(
            pontuacao=NivelQualidade.INACEITAVEL,
            correto_factualmente=False,
            completo=False,
            claro_e_objetivo=False,
            tom_adequado=True,
            justificativa="Resposta inventou dados que não existem.",
        )

        resposta = RespostaAgente(
            texto="Sua fatura é de R$ 0,00.",
            intencao=IntencaoUsuario.CONSULTA_FATURA,
            confianca=0.3,
        )

        with patch.object(avaliador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=avaliacao_mock)
            relatorio = await avaliador.avaliar("qual minha fatura?", resposta)

        assert relatorio.aprovado is False
        assert relatorio.pontuacao_numerica == 1

    @pytest.mark.asyncio
    async def test_falha_no_avaliador_resulta_em_aprovacao_por_padrao(self) -> None:
        llm_mock = MagicMock()
        avaliador = AvaliadorQualidade(llm=llm_mock)

        resposta = RespostaAgente(
            texto="Resposta qualquer.",
            intencao=IntencaoUsuario.GENERICA,
            confianca=1.0,
        )

        with patch.object(avaliador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(side_effect=Exception("LLM indisponível"))
            relatorio = await avaliador.avaliar("mensagem", resposta)

        assert relatorio.aprovado is True


class TestColetorAvaliacoes:
    """Testes das métricas agregadas do coletor de avaliações."""

    def _criar_relatorio(
        self,
        pontuacao: NivelQualidade,
        intencao: IntencaoUsuario,
        aprovado: bool,
    ):
        from avaliacao.avaliador_qualidade import RelatorioAvaliacao

        avaliacao = ResultadoAvaliacao(
            pontuacao=pontuacao,
            correto_factualmente=True,
            completo=True,
            claro_e_objetivo=True,
            tom_adequado=True,
            justificativa="Teste.",
        )
        return RelatorioAvaliacao(
            pergunta_usuario="teste",
            resposta_assistente="resposta",
            intencao_detectada=intencao,
            avaliacao=avaliacao,
            aprovado=aprovado,
        )

    def test_media_pontuacao_correta(self) -> None:
        coletor = ColetorAvaliacoes()
        coletor.registrar(self._criar_relatorio(NivelQualidade.EXCELENTE, IntencaoUsuario.CONSULTA_FATURA, True))
        coletor.registrar(self._criar_relatorio(NivelQualidade.BOM, IntencaoUsuario.CONSULTA_FATURA, True))
        coletor.registrar(self._criar_relatorio(NivelQualidade.ACEITAVEL, IntencaoUsuario.CONSULTA_LIMITE, True))

        assert coletor.media_pontuacao() == pytest.approx(4.0)

    def test_taxa_aprovacao_correta(self) -> None:
        coletor = ColetorAvaliacoes()
        coletor.registrar(self._criar_relatorio(NivelQualidade.EXCELENTE, IntencaoUsuario.CONSULTA_FATURA, True))
        coletor.registrar(self._criar_relatorio(NivelQualidade.INACEITAVEL, IntencaoUsuario.CONSULTA_LIMITE, False))

        assert coletor.taxa_aprovacao() == pytest.approx(50.0)

    def test_resumo_retorna_todas_metricas(self) -> None:
        coletor = ColetorAvaliacoes()
        coletor.registrar(self._criar_relatorio(NivelQualidade.BOM, IntencaoUsuario.CONSULTA_FATURA, True))

        resumo = coletor.resumo()

        assert resumo["total_avaliacoes"] == 1
        assert "media_pontuacao" in resumo
        assert "taxa_aprovacao_pct" in resumo
        assert "por_intencao" in resumo

    def test_coletor_vazio_retorna_zeros(self) -> None:
        coletor = ColetorAvaliacoes()
        assert coletor.media_pontuacao() == 0.0
        assert coletor.taxa_aprovacao() == 0.0
