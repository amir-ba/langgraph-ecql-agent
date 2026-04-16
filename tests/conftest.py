import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Ensure each test gets a fresh Settings instance."""
    from app.core.settings import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
