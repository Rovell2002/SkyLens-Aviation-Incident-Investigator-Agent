import re
from collections import Counter
from functools import lru_cache

import chromadb
from embedder import embed_query, COLLECTION

_col = chromadb.PersistentClient(path="chroma_db").get_collection(COLLECTION)


# ---------------------------------------------------------------- helpers

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


def _aircraft_variants(opt):
    """ASRS writes 'B787 Dreamliner', 'A320', not 'Boeing 787' / 'Airbus A320'."""
    out = [opt]
    m = re.search(r"boeing\s*-?\s*(\d{3})", opt)
    if m:
        out.append("b" + m.group(1))
    m = re.search(r"airbus\s*-?\s*a?\s*(\d{3})", opt)
    if m:
        out.append("a" + m.group(1))
    if "piper" in opt:          # most Piper models are listed only as PA-28, PA-34, ...
        out.append("pa-")
    return out


def _has(meta, key, needle):
    """Substring match; use | for alternatives, e.g. 'Piper|PA-'."""
    if not needle:
        return True
    value = str(meta.get(key, "")).lower()
    opts = [o.strip().lower() for o in needle.split("|") if o.strip()]
    if key == "aircraft":
        opts = [v for o in opts for v in _aircraft_variants(o)]
    return any(o in value for o in opts)


@lru_cache(maxsize=1)
def _all_metas(page=2000):
    """Load every chunk's metadata once, in pages (Chroma fails on one huge get())."""
    out, offset = [], 0
    while True:
        res = _col.get(include=["metadatas"], limit=page, offset=offset)
        batch = res["metadatas"]
        out.extend(batch)
        if len(batch) < page:
            break
        offset += page
    return out


def _rows(year_from=None, year_to=None, state=None,
          aircraft=None, flight_phase=None, anomaly=None):
    """One metadata row per report (chunks deduplicated by ACN), filtered."""
    seen = {}
    for m in _all_metas():
        yr = m.get("year", 0)
        if year_from and yr < year_from:
            continue
        if year_to and yr > year_to:
            continue
        if state and str(m.get("state", "")).upper() != state.upper():
            continue
        seen[m["acn"]] = m
    return [m for m in seen.values()
            if _has(m, "aircraft", aircraft)
            and _has(m, "flight_phase", flight_phase)
            and _has(m, "anomaly", anomaly)]


def _count_by(field, **filters):
    """Return (number of reports, Counter of labels). Multi-label fields split on ';'."""
    rows = _rows(**filters)
    counts = Counter()
    for m in rows:
        raw = str(m.get(field, "") or "unknown")
        parts = [p.strip() for p in raw.split(";") if p.strip()] or ["unknown"]
        counts.update(set(parts))   # count each label once per report
    return len(rows), counts


# ---------------------------------------------------------------- search

def search(query, n=5, year_from=None, year_to=None, state=None,
           aircraft=None, flight_phase=None, anomaly=None):
    """Semantic search with metadata filters; one result per report."""
    emb = embed_query(query)
    where = _where(year_from, year_to, state)
    pool = n * 8
    if any([aircraft, flight_phase, anomaly]):
        # Find matching reports from metadata first, then search only inside them.
        # (Filtering after a top-500 search missed rare aircraft such as the B787.)
        acns = sorted({m["acn"] for m in _rows(year_from, year_to, state,
                                               aircraft, flight_phase, anomaly)})
        if not acns:
            return []
        if len(acns) <= 3000:
            acn_cond = {"acn": {"$in": acns}}
            where = acn_cond if where is None else {"$and": [where, acn_cond]}
        else:
            pool = 500   # very common filter: search wide, then filter below
    res = _col.query(query_embeddings=emb, n_results=pool, where=where)
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


# ---------------------------------------------------------------- stats

def stats(field="anomaly", year_from=None, year_to=None, state=None,
          aircraft=None, flight_phase=None, anomaly=None, top=10):
    total, counts = _count_by(field, year_from=year_from, year_to=year_to, state=state,
                              aircraft=aircraft, flight_phase=flight_phase,
                              anomaly=anomaly)
    return {"total_matching_reports": total,
            "top": counts.most_common(top)}


# ---------------------------------------------------------------- charts

def _group_filters(filter_type, value, year_from, year_to):
    f = {"year_from": year_from, "year_to": year_to}
    if filter_type == "year":
        f.update(year_from=int(value), year_to=int(value))
    elif filter_type != "all":
        f[filter_type] = value
    return f


def compare(field, filter_type, values, year_from=None, year_to=None, top=8):
    """Count `field` inside each group. filter_type: aircraft, flight_phase,
    anomaly, state or year."""
    totals, counts = {}, {}
    for v in values:
        totals[v], counts[v] = _count_by(
            field, **_group_filters(filter_type, v, year_from, year_to))
    overall = Counter()
    for c in counts.values():
        overall.update(c)
    labels = [l for l, _ in overall.most_common(top)]
    rows = [{"group": v, "label": l, "count": counts[v][l],
             "pct": round(100 * counts[v][l] / max(totals[v], 1), 1)}
            for v in values for l in labels]
    return {"kind": "compare", "totals": totals, "rows": rows,
            "title": f"{field} by {filter_type}: " + " vs ".join(values)}


@lru_cache(maxsize=1)
def _covered_months():
    return sorted({m["date"] for m in _all_metas() if m.get("date")})


def trend(filter_type="all", values=("All",), year_from=None, year_to=None):
    """Monthly report counts per group. Months with no data at all stay empty (None)."""
    covered = _covered_months()
    covered_set = set(covered)
    first, last = covered[0], covered[-1]
    months, y, m = [], int(first[:4]), int(first[4:])
    while f"{y}{m:02d}" <= last:
        months.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    rows = []
    for v in values:
        _, c = _count_by("date", **_group_filters(filter_type, v, year_from, year_to))
        for mo in months:
            rows.append({"group": v, "month": mo,
                         "count": c.get(mo, 0) if mo in covered_set else None})
    return {"kind": "trend", "rows": rows, "totals": {},
            "title": "Monthly reports: " + " vs ".join(values)}