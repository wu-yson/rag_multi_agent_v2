import sys
from pathlib import Path
import pytest
from httpx import AsyncClient
from main import app

sys.path.append(str(Path(__file__).parent.parent))

@pytest.fixture(scope="session")
async def test_client():
    async with AsyncClient(app=app, base_url="http://test-server") as client:
        yield client
