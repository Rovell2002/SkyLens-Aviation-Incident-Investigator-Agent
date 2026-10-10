import glob
import re
import pandas as pd
import chromadb
from embedder import embed_docs, COLLECTION

RAW_GLOB = "data/**/*.csv"   # must match your data folder name exactly
DB_PATH = "chroma_db"
CHUNK_CHARS = 1500
OVERLAP = 200


def load_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=[0, 1], low_memory=False)
    cols = []
    for group, field in df.columns:
        field = str(field).strip()
        if str(group).startswith("Unnamed"):
            cols.append(field)
        else:
            cols.append(f"{str(group).strip()} | {field}")
    df.columns = cols
    df = df.loc[:, ~pd.Series(df.columns).str.startswith("Unnamed").values]
    return df


def find_col(df, *keywords):
    """First column whose name contains all keywords (case-insensitive)."""
    for c in df.columns:
        name = c.lower()
        if all(k.lower() in name for k in keywords):
            return c
    return None


def chunk(text: str):
    if len(text) <= CHUNK_CHARS:
        return [text]
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start + CHUNK_CHARS])
        start += CHUNK_CHARS - OVERLAP
    return out


def clean(text) -> str:
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def main():
    files = sorted(glob.glob(RAW_GLOB, recursive=True))
    assert files, f"No CSVs found at {RAW_GLOB}"
    df = pd.concat([load_file(f) for f in files], ignore_index=True)

    acn = find_col(df, "ACN")
    df = df.drop_duplicates(subset=acn)
    print(f"Loaded {len(df)} unique reports")
    print("\nAvailable columns:")
    for c in df.columns:
        print("  -", c)

    c_date = find_col(df, "Date")
    c_tod = find_col(df, "Local Time Of Day")
    c_state = find_col(df, "State")
    c_make = find_col(df, "Make Model")
    c_phase = find_col(df, "Flight Phase")
    c_anom = find_col(df, "Anomaly")
    c_syn = find_col(df, "Report 1", "Synopsis")
    c_n1 = find_col(df, "Report 1", "Narrative")
    c_n2 = find_col(df, "Report 2", "Narrative")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    ids, docs, metas = [], [], []
    for _, r in df.iterrows():
        synopsis = clean(r[c_syn]) if c_syn else ""
        narrative = clean(r[c_n1]) if c_n1 else ""
        if c_n2:
            narrative += " " + clean(r[c_n2])
        text = f"{synopsis} {narrative}".strip()
        if not text:
            continue

        date = clean(r[c_date]) if c_date else ""
        meta = {
            "acn": str(r[acn]),
            "date": date,
            "year": int(date[:4]) if date[:4].isdigit() else 0,
            "state": clean(r[c_state]) if c_state else "",
            "time_of_day": clean(r[c_tod]) if c_tod else "",
            "aircraft": clean(r[c_make]) if c_make else "",
            "flight_phase": clean(r[c_phase]) if c_phase else "",
            "anomaly": clean(r[c_anom]) if c_anom else "",
        }
        for i, piece in enumerate(chunk(text)):
            ids.append(f"{meta['acn']}_{i}")
            docs.append(piece)
            metas.append({**meta, "chunk": i})

    print(f"\nEmbedding {len(docs)} chunks...")
    BATCH = 256
    for i in range(0, len(docs), BATCH):
        b_docs = docs[i:i + BATCH]
        embs = embed_docs(b_docs)
        col.add(ids=ids[i:i + BATCH], documents=b_docs,
                embeddings=embs, metadatas=metas[i:i + BATCH])
        print(f"  {min(i + BATCH, len(docs))}/{len(docs)}")

    print("Done.")


if __name__ == "__main__":
    main()