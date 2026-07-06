import sys
from pathlib import Path

# Allows `from app...` imports to work when this script is run directly
# (e.g. `python scripts/ingest_documents.py`) rather than as part of the
# installed `app` package, by putting the backend/ root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client import QdrantClient  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.utils.logger import get_logger, setup_logging  # noqa: E402

setup_logging()
logger = get_logger(__name__)

settings = get_settings()

client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key or None,
    check_compatibility=False,
)

client.delete_collection(settings.vector_collection_name)

logger.info("Collection '%s' deleted successfully", settings.vector_collection_name)
