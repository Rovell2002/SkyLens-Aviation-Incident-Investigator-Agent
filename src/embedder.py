import os
import torch
from sentence_transformers import SentenceTransformer

KEY = os.getenv("EMBED_MODEL", "bge-small")

CONFIGS = {
    "bge-small": {"name": "BAAI/bge-small-en-v1.5",
                  "query_prefix": "Represent this sentence for searching relevant passages: "},
    "bge-base": {"name": "BAAI/bge-base-en-v1.5",
                 "query_prefix": "Represent this sentence for searching relevant passages: "},
    "qwen3-0.6b": {"name": "Qwen/Qwen3-Embedding-0.6B", "query_prompt": "query"},
    "embeddinggemma": {"name": "google/embeddinggemma-300m", "dtype": "bfloat16",
                       "query_prompt": "query", "doc_prompt": "document"},
}
CFG = CONFIGS[KEY]

# Baseline keeps the original collection name; others get their own
COLLECTION = "asrs" if KEY == "bge-small" else "asrs_" + KEY.replace("-", "_").replace(".", "_")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[embedder] model={KEY} device={DEVICE}")

dtype = getattr(torch, CFG.get("dtype", "float16"))
model = SentenceTransformer(
    CFG["name"], device=DEVICE,
    model_kwargs={"torch_dtype": dtype} if DEVICE == "cuda" else {})
model.max_seq_length = 256


def embed_docs(texts, batch_size=16):
    kw = {"prompt_name": CFG["doc_prompt"]} if "doc_prompt" in CFG else {}
    return model.encode(texts, normalize_embeddings=True,
                        batch_size=batch_size, **kw).tolist()


def embed_query(text):
    if "query_prompt" in CFG:
        return model.encode([text], normalize_embeddings=True,
                            prompt_name=CFG["query_prompt"]).tolist()
    return model.encode([CFG.get("query_prefix", "") + text],
                        normalize_embeddings=True).tolist()