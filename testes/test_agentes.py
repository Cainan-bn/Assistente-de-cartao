"""
Testes unitários do classificador de intenção.

Cobre cenários felizes, casos de borda e falhas da LLM,
garantindo comportamento previsível em produção.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentes.classificador_intencao import ClassificadorIntencao
from modelos.estado import IntencaoUsuario


@pytest.fixture
def llm_mock() -> MagicMock:
    """LLM mockado para isolamento total dos testes."""
    return MagicMock()


@pytest.fixture
def classificador(llm_mock: MagicMock) -> ClassificadorIntencao:
    return ClassificadorIntencao(llm=llm_mock)


class TestClassificadorIntencao:
    """Testes do classificador de intenção via LLM."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("mensagem", "intencao_esperada"),
        [
            ("qual o valor da minha fatura?", IntencaoUsuario.CONSULTA_FATURA),
            ("quanto tenho de limite disponível?", IntencaoUsuario.CONSULTA_LIMITE),
            ("mostra meus gastos do mês", IntencaoUsuario.CONSULTA_TRANSACOES),
            ("olá, tudo bem?", IntencaoUsuario.GENERICA),
        ],
    )
    async def test_classifica_intencoes_conhecidas(
        self,
        classificador: ClassificadorIntencao,
        mensagem: str,
        intencao_esperada: IntencaoUsuario,
    ) -> None:
        saida_mock = MagicMock()
        saida_mock.intencao = intencao_esperada
        saida_mock.confianca = 0.95

        with patch.object(classificador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=saida_mock)
            resultado = await classificador.classificar(mensagem)

        assert resultado == intencao_esperada

    @pytest.mark.asyncio
    async def test_retorna_generica_quando_llm_falha(
        self,
        classificador: ClassificadorIntencao,
    ) -> None:
        with patch.object(classificador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(side_effect=Exception("Timeout da LLM"))
            resultado = await classificador.classificar("qualquer mensagem")

        assert resultado == IntencaoUsuario.GENERICA

    @pytest.mark.asyncio
    async def test_classifica_mensagem_vazia_como_generica(
        self,
        classificador: ClassificadorIntencao,
    ) -> None:
        saida_mock = MagicMock()
        saida_mock.intencao = IntencaoUsuario.GENERICA
        saida_mock.confianca = 0.5

        with patch.object(classificador, "_cadeia") as cadeia_mock:
            cadeia_mock.ainvoke = AsyncMock(return_value=saida_mock)
            resultado = await classificador.classificar("")

        assert resultado == IntencaoUsuario.GENERICA


class TestClienteCartaoAPI:
    """Testes do cliente HTTP para a API de cartões."""

    @pytest.mark.asyncio
    async def test_buscar_fatura_atual_retorna_fatura_valida(self) -> None:
        from datetime import date
        from decimal import Decimal
        from unittest.mock import AsyncMock, patch

        import httpx

        from ferramentas.cliente_cartao_api import ClienteCartaoAPI
        from modelos.fatura import Fatura

        fatura_esperada = Fatura(
            id_fatura="FAT-001",
            id_cliente="CLI-123",
            valor_total=Decimal("1500.00"),
            valor_minimo=Decimal("150.00"),
            data_vencimento=date(2025, 2, 10),
            data_fechamento=date(2025, 1, 28),
            status="FECHADA",
            mes_referencia=1,
            ano_referencia=2025,
        )

        resposta_http_mock = MagicMock(spec=httpx.Response)
        resposta_http_mock.json.return_value = fatura_esperada.model_dump(mode="json")
        resposta_http_mock.raise_for_status = MagicMock()

        async with ClienteCartaoAPI() as cliente:
            with patch.object(cliente._cliente_http, "get", new_callable=AsyncMock) as get_mock:
                get_mock.return_value = resposta_http_mock
                fatura = await cliente.buscar_fatura_atual("CLI-123")

        assert fatura.id_cliente == "CLI-123"
        assert fatura.valor_total == Decimal("1500.00")
        assert fatura.status == "FECHADA"

    @pytest.mark.asyncio
    async def test_buscar_fatura_levanta_erro_em_falha_http(self) -> None:
        import httpx

        from ferramentas.cliente_cartao_api import ClienteCartaoAPI, ErroAPICartao

        resposta_erro = MagicMock(spec=httpx.Response)
        resposta_erro.status_code = 503

        async with ClienteCartaoAPI() as cliente:
            with patch.object(cliente._cliente_http, "get", new_callable=AsyncMock) as get_mock:
                get_mock.side_effect = httpx.HTTPStatusError(
                    "Serviço indisponível",
                    request=MagicMock(),
                    response=resposta_erro,
                )
                with pytest.raises(ErroAPICartao):
                    await cliente.buscar_fatura_atual("CLI-123")


class TestOrquestradorCartoes:
    """Testes de integração do orquestrador de agentes."""

    @pytest.mark.asyncio
    async def test_processa_mensagem_de_fatura(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from agentes.orquestrador import OrquestradorCartoes
        from modelos.estado import EstadoConversa, IntencaoUsuario
        from modelos.resposta import RespostaAgente

        llm_mock = MagicMock()
        orquestrador = OrquestradorCartoes(llm=llm_mock)

        resposta_esperada = RespostaAgente(
            texto="Sua fatura atual é de R$ 1.500,00 com vencimento em 10/02/2025.",
            intencao=IntencaoUsuario.CONSULTA_FATURA,
            confianca=1.0,
        )

        with patch.object(orquestrador, "_grafo") as grafo_mock:
            grafo_mock.ainvoke = AsyncMock(
                return_value={"resposta_final": resposta_esperada}
            )

            estado = EstadoConversa(id_sessao="sess-001", id_cliente="CLI-123")
            resultado = await orquestrador.processar_mensagem(estado, "qual minha fatura?")

        assert resultado.intencao == IntencaoUsuario.CONSULTA_FATURA
        assert "R$ 1.500,00" in resultado.texto
