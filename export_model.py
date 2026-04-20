"""Download all-MiniLM-L6-v2 ONNX model and tokenizer at Docker build time."""
import os
import urllib.request
import json

MODEL_DIR = "/app/model_onnx"
BASE_URL = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main"

FILES = {
    "model.onnx": f"{BASE_URL}/onnx/model.onnx",
    "tokenizer.json": f"{BASE_URL}/tokenizer.json",
}


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, url in FILES.items():
        dest = os.path.join(MODEL_DIR, name)
        if os.path.exists(dest):
            print(f"  {name} already exists, skipping")
            continue
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        print(f"  {name}: {size_mb:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
