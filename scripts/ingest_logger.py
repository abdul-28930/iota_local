"""Detailed per-document ingestion logging -- full path, timing, chunk/word
counts, embed success, metadata found. Writes to a timestamped .log file
alongside the usual console output."""
import time, logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger("ingest_detail")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
logger.addHandler(handler)


class DocTimer:
    """Wraps one document's ingestion, logs a full summary line at the end
    regardless of success/failure/skip."""
    def __init__(self, pdf_path: Path):
        self.path = pdf_path
        self.start = time.time()
        self.chunks = 0
        self.words = 0
        self.embedded = False
        self.metadata_found = False
        self.status = "unknown"

    def finish(self, status: str, chunks: int = 0, words: int = 0,
               embedded: bool = False, metadata_found: bool = False):
        elapsed = round(time.time() - self.start, 2)
        logger.info(
            f"path={self.path} status={status} elapsed={elapsed}s "
            f"chunks={chunks} words={words} embedded={embedded} metadata_found={metadata_found}"
        )
        print(f"  [log] {self.path.name}: {status} in {elapsed}s, {chunks} chunks, {words} words")


def get_log_path() -> Path:
    return LOG_FILE
