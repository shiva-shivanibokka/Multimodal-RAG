# backend/app/config.py
import os
class Settings:
    # ponytail: property (not a frozen class attr) so BACKEND_TOKEN is read live —
    # tests set it via monkeypatch.setenv after this module is already imported.
    @property
    def backend_token(self):
        return os.getenv("BACKEND_TOKEN", "")
    text_model = "BAAI/bge-small-en-v1.5"
    clip_model = "clip-ViT-B-32"
    reranker_model = "BAAI/bge-reranker-base"
    nli_model = "cross-encoder/nli-deberta-v3-base"
    faithfulness_threshold = 0.5   # ponytail: tunable knob, not magic
    retrieval_min_score = 0.25     # below this → refuse
settings = Settings()
