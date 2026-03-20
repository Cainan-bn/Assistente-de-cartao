"""
Script de demonstração local do assistente de cartões.

Executa o assistente com dados mock sem necessidade de API externa,
permitindo validar o fluxo completo de orquestração de agentes.

Como usar:
    OPENAI_API_KEY=sua-chave python demo_local.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI

DADOS_MOCK = {
    "fatura": {
        "id_fatura": "FAT-2025-01",
        "id_cliente": "CLI-DEMO",
        "valor_total": "1847.30",
        "valor_minimo": "184.73",
        "data_vencimento": "2025-02-10",
        "data_fechamento": "2025-01-28",
        "status": "FECHADA",
        "mes_referencia": 1,
        "ano_referencia": 2025,
    },
    "limite": {
        "id_cliente": "CLI-DEMO",
        "limite_total": "10000.00",
        "limite_disponivel": "8152.70",
        "limite_adicional": "0",
    },
    "transacoes": [
        {"id_transacao": "TRX-001", "id_cliente": "CLI-DEMO", "descricao": "SUPERMERCADO EXTRA", "valor": "342.50", "data_hora": "2025-01-15T14:30:00", "categoria": "Alimentação", "parcela_atual": None, "total_parcelas": None, "status": "APROVADA"},
        {"id_transacao": "TRX-002", "id_cliente": "CLI-DEMO", "descricao": "NETFLIX", "valor": "55.90", "data_hora": "2025-01-12T00:00:00", "categoria": "Streaming", "parcela_atual": None, "total_parcelas": None, "status": "APROVADA"},
        {"id_transacao": "TRX-003", "id_cliente": "CLI-DEMO", "descricao": "FARMACIA DROGASIL", "valor": "89.40", "data_hora": "2025-01-20T11:15:00", "categoria": "Saúde", "parcela_atual": None, "total_parcelas": None, "status": "APROVADA"},
    ],
}

PERGUNTAS_DEMO = [
    ("Olá! Qual o valor da minha fatura de janeiro?", "CONSULTA_FATURA"),
    ("E quando vence essa fatura?", "CONSULTA_FATURA"),
    ("Quanto tenho de limite disponível?", "CONSULTA_LIMITE"),
    ("Quais foram minhas últimas compras?", "CONSULTA_TRANSACOES"),
    ("Obrigado, pode encerrar!", "GENERICA"),
]

VERDE = "\033[92m"
AZUL = "\033[94m"
AMARELO = "\033[93m"
CINZA = "\033[90m"
RESET = "\033[0m"
NEGRITO = "\033[1m"


def _configurar_mock_api():
    """Configura o mock da API de cartões com dados realistas."""
    from modelos.fatura import Fatura
    from modelos.limite import InformacaoLimite
    from modelos.transacao import Transacao

    mock_api = MagicMock()
    mock_api.buscar_fatura_atual = AsyncMock(
        return_value=Fatura.model_validate(DADOS_MOCK["fatura"])
    )
    mock_api.buscar_limite = AsyncMock(
        return_value=InformacaoLimite.model_validate(DADOS_MOCK["limite"])
    )
    mock_api.buscar_transacoes = AsyncMock(
        return_value=[Transacao.model_validate(t) for t in DADOS_MOCK["transacoes"]]
    )
    return mock_api


def _imprimir_cabecalho():
    print(f"\n{NEGRITO}{'='*60}{RESET}")
    print(f"{NEGRITO}  🃏 Assistente de Cartões IA — Demo Local{RESET}")
    print(f"{NEGRITO}  Getronics | LangGraph + LangChain + GPT-4o{RESET}")
    print(f"{NEGRITO}{'='*60}{RESET}\n")


def _imprimir_usuario(mensagem: str):
    print(f"\n{AZUL}{NEGRITO}👤 Usuário:{RESET} {mensagem}")


def _imprimir_assistente(resposta: str, intencao: str, tempo_ms: float):
    print(f"{VERDE}{NEGRITO}🤖 Assistente:{RESET} {resposta}")
    print(f"{CINZA}   ↳ Intenção: {intencao} | Tempo: {tempo_ms:.0f}ms{RESET}")


def _imprimir_rodape():
    print(f"\n{NEGRITO}{'='*60}{RESET}")
    print(f"{AMARELO}  Demo concluído. Ver código em: assistente_cartoes/{RESET}")
    print(f"{NEGRITO}{'='*60}{RESET}\n")


async def executar_demo():
    """Executa o fluxo de demonstração completo com dados mock."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{AMARELO}⚠️  OPENAI_API_KEY não definida. Configure antes de executar.{RESET}")
        sys.exit(1)

    _imprimir_cabecalho()

    from agentes.orquestrador import OrquestradorCartoes
    from modelos.estado import EstadoConversa

    mock_api = _configurar_mock_api()

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    orquestrador = OrquestradorCartoes(llm=llm)

    estado = EstadoConversa(id_sessao="demo-session-001", id_cliente="CLI-DEMO")

    import time

    with (
        patch("agentes.agente_fatura.ClienteCartaoAPI", return_value=mock_api),
        patch("agentes.agente_limite.ClienteCartaoAPI", return_value=mock_api),
        patch("agentes.agente_transacao.ClienteCartaoAPI", return_value=mock_api),
    ):
        for mensagem, intencao_esperada in PERGUNTAS_DEMO:
            _imprimir_usuario(mensagem)

            inicio = time.perf_counter()
            resposta = await orquestrador.processar_mensagem(estado, mensagem)
            duracao_ms = (time.perf_counter() - inicio) * 1000

            _imprimir_assistente(resposta.texto, resposta.intencao, duracao_ms)

            from langchain_core.messages import HumanMessage
            estado.historico_mensagens.append(HumanMessage(content=mensagem))
            estado.historico_mensagens.append(AIMessage(content=resposta.texto))

            await asyncio.sleep(0.5)

    _imprimir_rodape()


if __name__ == "__main__":
    asyncio.run(executar_demo())
