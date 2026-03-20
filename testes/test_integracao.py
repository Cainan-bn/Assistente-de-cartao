"""
Testes de integração da API REST do assistente de cartões.

Testa os endpoints FastAPI de ponta a ponta com mocks da LLM e da API
de cartões, garantindo que contratos HTTP e regras de negócio sejam mantidos.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from modelos.estado import IntencaoUsuario
from modelos.resposta import RespostaAgente
from servicos.api import app


@pytest.fixture
def resposta_fatura_mock() -> RespostaAgente:
    return RespostaAgente(
        texto="Sua fatura de janeiro é de R$ 1.500,00, vencendo em 10/02/2025.",
        intencao=IntencaoUsuario.CONSULTA_FATURA,
        confianca=0.97,
    )


@pytest.fixture
def cliente_api() -> TestClient:
    return TestClient(app)


class TestEndpointMensagem:
    """Testes de integração para POST /v1/mensagem."""

    def test_retorna_200_com_resposta_valida(
        self,
        cliente_api: TestClient,
        resposta_fatura_mock: RespostaAgente,
    ) -> None:
        with patch("servicos.api._orquestrador") as orquestrador_mock:
            orquestrador_mock.processar_mensagem = AsyncMock(return_value=resposta_fatura_mock)

            resposta = cliente_api.post(
                "/v1/mensagem",
                json={"mensagem": "qual minha fatura?", "id_cliente": "CLI-123"},
            )

        assert resposta.status_code == 200
        dados = resposta.json()
        assert "id_sessao" in dados
        assert dados["resposta"]["intencao"] == "CONSULTA_FATURA"

    def test_retorna_422_com_mensagem_vazia(self, cliente_api: TestClient) -> None:
        resposta = cliente_api.post(
            "/v1/mensagem",
            json={"mensagem": "", "id_cliente": "CLI-123"},
        )
        assert resposta.status_code == 422

    def test_retorna_422_sem_id_cliente(self, cliente_api: TestClient) -> None:
        resposta = cliente_api.post(
            "/v1/mensagem",
            json={"mensagem": "qual minha fatura?"},
        )
        assert resposta.status_code == 422

    def test_reutiliza_sessao_existente(
        self,
        cliente_api: TestClient,
        resposta_fatura_mock: RespostaAgente,
    ) -> None:
        id_sessao_existente = "sessao-abc-123"

        with patch("servicos.api._orquestrador") as orquestrador_mock:
            orquestrador_mock.processar_mensagem = AsyncMock(return_value=resposta_fatura_mock)

            resposta = cliente_api.post(
                "/v1/mensagem",
                json={
                    "mensagem": "minha fatura do mês",
                    "id_cliente": "CLI-456",
                    "id_sessao": id_sessao_existente,
                },
            )

        assert resposta.json()["id_sessao"] == id_sessao_existente

    def test_retorna_503_quando_orquestrador_indisponivel(
        self,
        cliente_api: TestClient,
    ) -> None:
        with patch("servicos.api._orquestrador", None):
            resposta = cliente_api.post(
                "/v1/mensagem",
                json={"mensagem": "olá", "id_cliente": "CLI-789"},
            )
        assert resposta.status_code == 503

    def test_retorna_500_quando_orquestrador_levanta_excecao(
        self,
        cliente_api: TestClient,
    ) -> None:
        with patch("servicos.api._orquestrador") as orquestrador_mock:
            orquestrador_mock.processar_mensagem = AsyncMock(
                side_effect=RuntimeError("Falha inesperada")
            )

            resposta = cliente_api.post(
                "/v1/mensagem",
                json={"mensagem": "minha fatura", "id_cliente": "CLI-999"},
            )

        assert resposta.status_code == 500


class TestEndpointHealthCheck:
    """Testes do endpoint de health check."""

    def test_retorna_200_e_status_ok(self, cliente_api: TestClient) -> None:
        resposta = cliente_api.get("/health")
        assert resposta.status_code == 200
        assert resposta.json()["status"] == "ok"


class TestRepositorioMemoriaEmMemoria:
    """Testes do repositório de memória em memória."""

    @pytest.mark.asyncio
    async def test_salva_e_carrega_mensagens_corretamente(self) -> None:
        from langchain_core.messages import HumanMessage

        from servicos.repositorio_memoria import RepositorioMemoriaEmMemoria

        repo = RepositorioMemoriaEmMemoria()
        mensagem = HumanMessage(content="Qual minha fatura?")

        await repo.salvar_mensagem("sessao-1", mensagem)
        historico = await repo.carregar_historico("sessao-1")

        assert len(historico) == 1
        assert historico[0].content == "Qual minha fatura?"

    @pytest.mark.asyncio
    async def test_limpar_sessao_remove_historico(self) -> None:
        from langchain_core.messages import HumanMessage

        from servicos.repositorio_memoria import RepositorioMemoriaEmMemoria

        repo = RepositorioMemoriaEmMemoria()
        await repo.salvar_mensagem("sessao-2", HumanMessage(content="teste"))
        await repo.limpar_sessao("sessao-2")
        historico = await repo.carregar_historico("sessao-2")

        assert historico == []

    @pytest.mark.asyncio
    async def test_sessoes_sao_isoladas(self) -> None:
        from langchain_core.messages import HumanMessage

        from servicos.repositorio_memoria import RepositorioMemoriaEmMemoria

        repo = RepositorioMemoriaEmMemoria()
        await repo.salvar_mensagem("sessao-A", HumanMessage(content="mensagem A"))
        await repo.salvar_mensagem("sessao-B", HumanMessage(content="mensagem B"))

        historico_a = await repo.carregar_historico("sessao-A")
        historico_b = await repo.carregar_historico("sessao-B")

        assert len(historico_a) == 1
        assert len(historico_b) == 1
        assert historico_a[0].content != historico_b[0].content
