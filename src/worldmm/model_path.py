"""Where model weights are loaded from (EPIC-Morph port).

WORLDMM_MODEL_DIR, when set, is a folder of model folders named like the last part of the Hugging Face id
(e.g. <dir>/Qwen3-VL-8B-Instruct, <dir>/Qwen3-Embedding-4B, <dir>/VLM2Vec-V2.0). A model found there is loaded
from it; otherwise the Hugging Face id is used as before (the hub, or the local cache with HF_HUB_OFFLINE=1).
"""
import os


def resolve_model_path(name: str) -> str:
    root = os.environ.get("WORLDMM_MODEL_DIR")
    if root and not os.path.isabs(name):
        path = os.path.join(root, name.rstrip("/").split("/")[-1])
        if os.path.isdir(path):
            return path
    return name
