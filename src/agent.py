import json
import os
import uuid
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import retriever as R

SYSTEM = """You are SkyLens, an aviation safety analyst working over NASA ASRS
incident reports from 2022 to 2026. 2022-2025 are full years; only 2026 is partial,
so do not compare 2026 with full years without saying so. Rules:
- Always ground claims in retrieved reports and cite them as [ACN 1234567], including
  inside tables (write "ACN 1234567", never the bare number).
- Your only tools are search_reports, get_report, stats_summary, compare_groups and
  trend_chart. There is no browser, file or code tool; never call any other tool.
- Search first. If results are thin or off-topic, rephrase and search again, but use
  at most 3 tool calls per question and answer from what you have.
- Use get_report when a snippet is not enough, and stats_summary for counts or trends.
- If the evidence does not support an answer, say so. Never invent report details.
- Only attribute a cause to a report if that report states it. If a snippet does not say
  why something happened, write "cause not stated" or call get_report. Keep general
  aviation knowledge separate and label it "General background (not from the reports)".
- In stats results, one report can carry several anomaly labels. When a filter is used,
  the other labels are co-occurring labels, not sub-types, and their counts do not add up
  to the total. Say this when you present them.
- ASRS is a voluntary incident-reporting system, not a crash database. For accidents,
  suggest the NTSB database.
- ASRS reports are voluntary and anecdotal, so counts are not true accident rates.
- For comparisons or trends, call compare_groups or trend_chart. The chart is shown to
  the user automatically, so give a 2-3 sentence interpretation and do not reprint the data.
- Mention that 2026 is a partial year whenever you describe a trend across years.
- Filters exclude reports with missing fields, so prefer a plain semantic query first and
  add filters only when the user names a specific aircraft, phase, state or year."""


CHART_STORE = {}   # full chart data is kept here; the LLM only sees a short summary


def _store_chart(payload, summary):
    cid = uuid.uuid4().hex[:8]
    CHART_STORE[cid] = payload
    return f"CHART:{cid}\n{summary}"


@tool
def compare_groups(filter_type: str, values: list[str], field: str = "anomaly",
                   year_from: int = None, year_to: int = None) -> str:
    """Compare 2-4 groups side by side and draw a bar chart for the user.
    filter_type is one of: aircraft, flight_phase, state, year.
    values are the groups, e.g. ['Cessna', 'Piper|PA-'] or ['2022', '2024'].
    field is what to count inside each group: anomaly, flight_phase or aircraft."""
    p = R.compare(field, filter_type, values, year_from, year_to, top=5)
    lines = [f"Chart shown to the user. Reports per group: {p['totals']}."]
    for v in values:
        top = sorted((r for r in p["rows"] if r["group"] == v), key=lambda r: -r["pct"])[:3]
        lines.append(f"{v}: " + "; ".join(f"{r['label']} {r['pct']}%" for r in top))
    return _store_chart(p, "\n".join(lines))


@tool
def trend_chart(filter_type: str = "all", values: list[str] = None,
                year_from: int = None, year_to: int = None) -> str:
    """Draw monthly report counts as a line chart, optionally one line per group.
    Use filter_type 'anomaly' to trend a topic, e.g. values ['Smoke / Fire / Fumes'],
    or 'aircraft' / 'flight_phase' to compare groups over time."""
    p = R.trend(filter_type, tuple(values or ["All"]), year_from, year_to)
    yearly = {}
    for r in p["rows"]:
        if r["count"] is not None:
            key = (r["group"], r["month"][:4])
            yearly[key] = yearly.get(key, 0) + r["count"]
    summary = "Chart shown to the user. Reports per year: " + "; ".join(
        f"{g} {y}: {n}" for (g, y), n in sorted(yearly.items()))
    return _store_chart(p, summary)


@tool
def search_reports(query: str, n: int = 5, year_from: int = None, year_to: int = None,
                   state: str = None, aircraft: str = None,
                   flight_phase: str = None, anomaly: str = None) -> str:
    """Semantic search over incident reports. Optional filters: year_from/year_to
    (e.g. 2024), state (2-letter code), aircraft (substring; ASRS uses codes such as
    'B737', 'B787', 'A320', 'PA-28', 'Cessna 172'; use | for alternatives),
    flight_phase (e.g. 'Landing'), anomaly (substring)."""
    hits = R.search(query, n=n, year_from=year_from, year_to=year_to, state=state,
                    aircraft=aircraft, flight_phase=flight_phase, anomaly=anomaly)
    if not hits:
        return "No matching reports. Try a broader query or fewer filters."
    return "\n\n".join(
        f"[ACN {h['acn']}] {h['date']} | {h['aircraft']} | phase: {h['flight_phase']}\n{h['text'][:400]}"
        for h in hits)


@tool
def get_report(acn: str) -> str:
    """Fetch the full text of one report by its ACN."""
    r = R.get_report(acn)
    if not r:
        return "Report not found."
    return f"[ACN {r['acn']}] {r['date']} | {r['aircraft']}\n{r['text'][:2500]}"


@tool
def stats_summary(field: str = "anomaly", year_from: int = None, year_to: int = None,
                  aircraft: str = None, flight_phase: str = None,
                  anomaly: str = None, state: str = None) -> str:
    """Count reports grouped by a field (anomaly, flight_phase, aircraft, state, date,
    time_of_day), with optional filters. Use for 'how many' or 'most common' questions."""
    r = R.stats(field=field, year_from=year_from, year_to=year_to,
                aircraft=aircraft, flight_phase=flight_phase,
                anomaly=anomaly, state=state)
    return (f"Reports matching the filters: {r['total_matching_reports']}.\n"
            f"Most common '{field}' labels among those reports (a report can have several "
            f"labels, so counts overlap and do not sum to the total): {r['top']}")


llm = ChatGroq(model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
               temperature=0, max_tokens=1500, max_retries=5,
               reasoning_effort="low")   # less hidden "thinking" = faster answers
agent = create_react_agent(llm, [search_reports, get_report, stats_summary, compare_groups, trend_chart], prompt=SYSTEM)

if __name__ == "__main__":
    print("SkyLens ready. Type 'quit' to exit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [("user", q)]})
        print("\nSkyLens:", result["messages"][-1].content)