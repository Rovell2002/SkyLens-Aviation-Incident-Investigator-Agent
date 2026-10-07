import chromadb
from collections import Counter
from sentence_transformers import SentenceTransformer

PREFIX = "Represent this sentence for searching relevant passages: "
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
_col = chromadb.PersistentClient(path="chroma_db").get_collection("asrs")


def _where(year_from=None, year_to=None, state=None):
    conds = []
    if year_from:
        conds.append({"year": {"$gte": year_from}})
    if year_to:
        conds.append({"year": {"$lte": year_to}})
    if state:
        conds.append({"state": state.upper()})
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def _has(meta, key, needle):
    if not needle:
        return True
    value = meta.get(key, "").lower()
    return any(opt.strip().lower() in value for opt in needle.split("|"))


def search(query, n=5, year_from=None, year_to=None, state=None,
           aircraft=None, flight_phase=None, anomaly=None):
    """Semantic search with metadata filters; one result per report."""
    emb = _model.encode([PREFIX + query], normalize_embeddings=True).tolist()
    has_post_filter = any([aircraft, flight_phase, anomaly])
    pool = 500 if has_post_filter else n * 8
    res = _col.query(query_embeddings=emb, n_results=pool,
                     where=_where(year_from, year_to, state))
    out = {}
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        acn = meta["acn"]
        if acn in out:
            continue
        if not (_has(meta, "aircraft", aircraft)
                and _has(meta, "flight_phase", flight_phase)
                and _has(meta, "anomaly", anomaly)):
            continue
        out[acn] = {**meta, "text": doc}
        if len(out) >= n:
            break
    return list(out.values())


def get_report(acn):
    res = _col.get(where={"acn": str(acn)})
    if not res["ids"]:
        return None
    parts = sorted(zip(res["metadatas"], res["documents"]), key=lambda x: x[0]["chunk"])
    return {**parts[0][0], "text": " ".join(d for _, d in parts)}


def stats(field="anomaly", year_from=None, year_to=None, state=None,
          aircraft=None, flight_phase=None, anomaly=None, top=10):
    res = _col.get(where=_where(year_from, year_to, state), include=["metadatas"])
    seen = {}
    for m in res["metadatas"]:
        seen[m["acn"]] = m
    rows = [m for m in seen.values()
            if _has(m, "aircraft", aircraft)
            and _has(m, "flight_phase", flight_phase)
            and _has(m, "anomaly", anomaly)]
    counts = Counter()
    for m in rows:
        raw = m.get(field, "") or "unknown"
        parts = [p.strip() for p in raw.split(";") if p.strip()] or ["unknown"]
        counts.update(set(parts))   # count each label once per report
    return {"total_matching_reports": len(rows),
            "top": counts.most_common(top)}