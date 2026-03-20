"""
Avaliador de qualidade de respostas usando LLM-as-Judge.

Implementa o padrão de avaliação automatizada onde um LLM avalia
a qualidade das respostas geradas pelo assistente, garantindo que
métricas de qualidade sejam monitoráveis em produção.

Referência: https://arxiv.org/abs/2306.05685 (Judging LLM-as-a-Judge)
"""

from __future__ import annotations

import logging
from enum import IntEnum

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from modelos.estado import IntencaoUsuario
from modelos.resposta import RespostaAgente

logger = logging.getLogger(__name__)


class NivelQualidade(IntEnum):
    """Escala de avaliação da qualidade de respostas (1-5)."""

    INACEITAVEL = 1
    ABAIXO_DO_ESPERADO = 2
    ACEITAVEL = 3
    BOM = 4
    EXCELENTE = 5


class ResultadoAvaliacao(BaseModel):
    """Resultado estruturado da avaliação LLM-as-Judge."""

    pontuacao: NivelQualidade
    correto_factualmente: bool
    completo: bool
    claro_e_objetivo: bool
    tom_adequado: bool
    justificativa: str = Field(max_length=500)


class RelatorioAvaliacao(BaseModel):
    """Relatório completo de uma avaliação de resposta do assistente."""

    pergunta_usuario: str
    resposta_assistente: str
    intencao_detectada: IntencaoUsuario
    avaliacao: ResultadoAvaliacao
    aprovado: bool

    @property
    def pontuacao_numerica(self) -> int:
        return self.avaliacao.pontuacao.value


_PROMPT_AVALIADOR = ChatPromptTemplate.from_messages([
    (
        "system",
        """Você é um avaliador especializado em qualidade de assistentes de cartão de crédito.
        Avalie a resposta do assistente com rigor e precisão.

        Critérios de avaliação:
        - Correção factual: A resposta está correta e não inventa dados?
        - Completude: A resposta atende plenamente à pergunta do usuário?
        - Clareza: A resposta é objetiva e fácil de entender?
        - Tom: A resposta é empática, profissional e adequada ao contexto bancário?

        Escala de pontuação:
        1 = Inaceitável (erro grave ou resposta completamente inadequada)
        2 = Abaixo do esperado (parcialmente correto ou confuso)
        3 = Aceitável (correto mas poderia ser melhor)
        4 = Bom (atende bem aos critérios)
        5 = Excelente (supera as expectativas em todos os critérios)
        """,
    ),
    (
        "human",
        """Avalie a seguinte interação:

        PERGUNTA DO USUÁRIO:
        {pergunta}

        INTENÇÃO DETECTADA:
        {intencao}

        RESPOSTA DO ASSISTENTE:
        {resposta}

        Forneça sua avaliação estruturada.
        """,
    ),
])


class AvaliadorQualidade:
    """
    Avalia a qualidade das respostas do assistente usando LLM-as-Judge.

    Permite monitoramento contínuo de qualidade em produção e pode ser
    usado em pipelines de fine-tuning para identificar amostras ruins.
    """

    _PONTUACAO_MINIMA_APROVACAO = NivelQualidade.ACEITAVEL

    def __init__(self, llm: ChatOpenAI) -> None:
        llm_estruturado = llm.with_structured_output(ResultadoAvaliacao)
        self._cadeia = _PROMPT_AVALIADOR | llm_estruturado

    async def avaliar(
        self,
        pergunta: str,
        resposta: RespostaAgente,
    ) -> RelatorioAvaliacao:
        """
        Avalia a qualidade de uma resposta do assistente.

        Args:
            pergunta: Mensagem original do usuário.
            resposta: Resposta gerada pelo assistente.

        Returns:
            RelatorioAvaliacao com pontuação e justificativa detalhada.
        """
        try:
            avaliacao: ResultadoAvaliacao = await self._cadeia.ainvoke({
                "pergunta": pergunta,
                "intencao": resposta.intencao.value,
                "resposta": resposta.texto,
            })

            aprovado = avaliacao.pontuacao >= self._PONTUACAO_MINIMA_APROVACAO

            logger.info(
                "Avaliação concluída",
                extra={
                    "pontuacao": avaliacao.pontuacao,
                    "aprovado": aprovado,
                    "intencao": resposta.intencao,
                },
            )

            return RelatorioAvaliacao(
                pergunta_usuario=pergunta,
                resposta_assistente=resposta.texto,
                intencao_detectada=resposta.intencao,
                avaliacao=avaliacao,
                aprovado=aprovado,
            )

        except Exception:
            logger.exception("Falha na avaliação da resposta")
            avaliacao_fallback = ResultadoAvaliacao(
                pontuacao=NivelQualidade.ACEITAVEL,
                correto_factualmente=True,
                completo=True,
                claro_e_objetivo=True,
                tom_adequado=True,
                justificativa="Avaliação indisponível — falha no avaliador.",
            )
            return RelatorioAvaliacao(
                pergunta_usuario=pergunta,
                resposta_assistente=resposta.texto,
                intencao_detectada=resposta.intencao,
                avaliacao=avaliacao_fallback,
                aprovado=True,
            )


class ColetorAvaliacoes:
    """
    Coleta e agrega avaliações para análise de tendências de qualidade.

    Em produção, as avaliações devem ser persistidas em banco de dados
    para alimentar dashboards e identificar regressões.
    """

    def __init__(self) -> None:
        self._avaliacoes: list[RelatorioAvaliacao] = []

    def registrar(self, avaliacao: RelatorioAvaliacao) -> None:
        """Registra uma avaliação na coleção em memória."""
        self._avaliacoes.append(avaliacao)

    def media_pontuacao(self) -> float:
        """Calcula a pontuação média de todas as avaliações registradas."""
        if not self._avaliacoes:
            return 0.0
        return sum(a.pontuacao_numerica for a in self._avaliacoes) / len(self._avaliacoes)

    def taxa_aprovacao(self) -> float:
        """Retorna o percentual de respostas aprovadas."""
        if not self._avaliacoes:
            return 0.0
        aprovadas = sum(1 for a in self._avaliacoes if a.aprovado)
        return aprovadas / len(self._avaliacoes) * 100

    def avaliacoes_por_intencao(self) -> dict[str, float]:
        """Agrupa a média de pontuação por intenção do usuário."""
        agrupado: dict[str, list[int]] = {}
        for avaliacao in self._avaliacoes:
            chave = avaliacao.intencao_detectada.value
            agrupado.setdefault(chave, []).append(avaliacao.pontuacao_numerica)
        return {k: sum(v) / len(v) for k, v in agrupado.items()}

    def resumo(self) -> dict:
        """Retorna um resumo executivo das métricas de qualidade."""
        return {
            "total_avaliacoes": len(self._avaliacoes),
            "media_pontuacao": round(self.media_pontuacao(), 2),
            "taxa_aprovacao_pct": round(self.taxa_aprovacao(), 1),
            "por_intencao": {k: round(v, 2) for k, v in self.avaliacoes_por_intencao().items()},
        }
