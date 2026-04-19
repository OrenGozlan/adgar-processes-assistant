import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


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
