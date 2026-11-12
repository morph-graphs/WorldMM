import os
from typing import Union, List
import numpy as np
from sentence_transformers import SentenceTransformer

from ..model_path import resolve_model_path


class Qwen3EmbeddingModel:
    """Wrapper for Qwen3 Embedding Model"""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        
        self.model = SentenceTransformer(
            resolve_model_path(model_name),
            model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "auto", "device_map": device},
            tokenizer_kwargs={"padding_side": "left"},
        )
    
    def encode_text(self, texts: Union[str, List[str]], batch_size: int = 256) -> np.ndarray:
        """Encode text into embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        
        # WORLDMM_EMB_BATCH caps the batch (memory only, same embeddings): 256 OOMs a 16 GB GPU on long videos
        if os.environ.get("WORLDMM_EMB_BATCH"):
            batch_size = min(batch_size, int(os.environ["WORLDMM_EMB_BATCH"]))
        embeddings = self.model.encode(texts, batch_size=batch_size)
        return embeddings
    
    def encode(self, content: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Universal encode method for text"""
        return self.encode_text(content, **kwargs)
