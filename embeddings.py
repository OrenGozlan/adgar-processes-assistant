import os
import numpy as np
import threading

_model = None
_model_lock = threading.Lock()

BAKED_CACHE = "/app/model_cache"
CACHE_DIR = "/data/models" if os.path.isdir("/data") else "./models"


def _find_baked_model():
    if not os.path.isdir(BAKED_CACHE):
        return None
    for name in os.listdir(BAKED_CACHE):
        if "all-MiniLM-L6-v2" in name:
            path = os.path.join(BAKED_CACHE, name)
            if os.path.isdir(path):
                return path
    return None


def _get_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        from sentence_transformers import SentenceTransformer
        baked = _find_baked_model()
        if baked:
            _model = SentenceTransformer(baked)
        else:
            os.makedirs(CACHE_DIR, exist_ok=True)
            _model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=CACHE_DIR)
        return _model


def is_model_ready() -> bool:
    return _model is not None


def embed_text(text: str) -> list[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))
