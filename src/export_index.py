"""Copy one collection from chroma_db/ into a small, standalone deploy_db/ folder.

chroma_db/ holds every embedding model we tested (bge-small, bge-base, qwen3).
The live app only needs bge-small, so we export just that collection and upload it.

Run from the project root:  python src\\export_index.py
"""
import shutil
import sys

import chromadb

SRC, DST = "chroma_db", "deploy_db"
NAME = sys.argv[1] if len(sys.argv) > 1 else "asrs"   # "asrs" = the bge-small collection
PAGE = 1000

src = chromadb.PersistentClient(path=SRC).get_collection(NAME)
shutil.rmtree(DST, ignore_errors=True)
dst = chromadb.PersistentClient(path=DST).create_collection(NAME, metadata={"hnsw:space": "cosine"})

total, offset = src.count(), 0
while offset < total:
    b = src.get(include=["embeddings", "documents", "metadatas"], limit=PAGE, offset=offset)
    dst.add(ids=b["ids"], embeddings=b["embeddings"], documents=b["documents"],
            metadatas=b["metadatas"])
    offset += len(b["ids"])
    print(f"  {offset}/{total}")

print(f"Done. {dst.count()} chunks copied to {DST}/ (source had {total}).")
