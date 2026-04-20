import os
import threading
import numpy as np
from tokenizers import Tokenizer
import onnxruntime as ort

_session = None
_tokenizer = None
_lock = threading.Lock()

MODEL_DIR = "/app/model_onnx"
FALLBACK_DIR = "./model_onnx"


def _get_model_dir():
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    return FALLBACK_DIR


def _init():
    global _session, _tokenizer
    if _session is not None:
        return
    with _lock:
        if _session is not None:
            return
        d = _get_model_dir()
        _tokenizer = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        _tokenizer.enable_padding(length=128)
        _tokenizer.enable_truncation(max_length=128)
        _session = ort.InferenceSession(
            os.path.join(d, "model.onnx"),
            providers=["CPUExecutionProvider"],
        )


def is_model_ready() -> bool:
    return _session is not None


def _mean_pool(token_embs, attention_mask):
    mask = np.expand_dims(attention_mask, -1).astype(np.float32)
    summed = np.sum(token_embs * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def _normalize(v):
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(norms, 1e-10, None)


def _encode_batch(texts):
    _init()
    encoded = _tokenizer.encode_batch(texts)
    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
    token_type_ids = np.zeros_like(input_ids)

    outputs = _session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )
    pooled = _mean_pool(outputs[0], attention_mask)
    return _normalize(pooled)


def embed_text(text: str) -> list[float]:
    return _encode_batch([text])[0].tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    return _encode_batch(texts).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))
