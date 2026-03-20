"""
Fixtures compartilhadas entre todos os módulos de teste.
"""
import pytest

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
