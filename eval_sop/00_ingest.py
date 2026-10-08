"""Step 0: run the original ingestion once and cache it. Prints chunk stats."""
import time
import common  # noqa: F401  (sets sys.path / HF_HOME)
from common import load_or_ingest

t = time.time()
sid, index, d2p = load_or_ingest()
from app.session import get_session
s = get_session(sid)
from collections import Counter
print("docs", len(d2p), "chunks", len(s["chunks"]), Counter(c["kind"] for c in s["chunks"]), f"{time.time()-t:.0f}s")
