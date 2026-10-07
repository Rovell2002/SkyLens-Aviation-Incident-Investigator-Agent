import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
col = chromadb.PersistentClient(path="chroma_db").get_collection("asrs")

q = "landing gear failed to extend"
emb = model.encode(
    ["Represent this sentence for searching relevant passages: " + q],
    normalize_embeddings=True).tolist()
res = col.query(query_embeddings=emb, n_results=5)

for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
    print(meta["acn"], meta["date"], meta["aircraft"], "|", doc[:200], "\n")