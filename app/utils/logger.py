"""Application-wide logging configuration.

Centralizing this in one place means every module can just do
`logger = get_logger(__name__)` and get consistently formatted output
(timestamp, level, module name, message) without repeating config.
"""

import logging
import sys
from pathlib import Path

LOG_FILE_PATH = Path(__file__).resolve().parents[2] / "logs" / "app.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configures Python's root logger once, at process startup (called
    from `app/main.py` and `scripts/ingest_documents.py`). Logs go to
    stdout (rather than stderr or a file) so they show up in `uvicorn`'s
    console output and in `docker compose logs` without extra setup.
    Calling this more than once is harmless - `basicConfig` no-ops if the
    root logger already has handlers."""
    LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=(
            "[%(asctime)s] "
            "[%(levelname)-8s] "
            "[%(name)s] "
            "[%(filename)s:%(lineno)d] "
            "%(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE_PATH, encoding="utf-8"),
        ],
    )


def get_logger(name: str) -> logging.Logger:
    """Returns a named logger, conventionally called as
    `get_logger(__name__)` so log lines show which module they came from
    (e.g. `app.core.rag_pipeline`). Use this instead of `print()`
    everywhere so logs can be filtered/leveled/redirected consistently."""
    return logging.getLogger(name)
