from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import retriever as R

SYSTEM = """You are SkyLens, an aviation safety analyst working over NASA ASRS
incident reports from 2022. Rules:
- Always ground claims in retrieved reports and cite them as [ACN 1234567].
- Search first. If results are thin or off-topic, rephrase and search again.
- Use get_report when a snippet is not enough, and stats_summary for counts or trends.
- If the evidence does not support an answer, say so. Never invent report details.
- ASRS reports are voluntary and anecdotal, so counts are not true accident rates."""


@tool
def search_reports(query: str, n: int = 5, state: str = None, aircraft: str = None,
                   flight_phase: str = None, anomaly: str = None) -> str:
    """Semantic search over incident reports. Optional filters: state (2-letter code),
    aircraft (substring, e.g. 'Cessna'), flight_phase (e.g. 'Landing'), anomaly (substring)."""
    hits = R.search(query, n=n, state=state, aircraft=aircraft,
                    flight_phase=flight_phase, anomaly=anomaly)
    if not hits:
        return "No matching reports. Try a broader query or fewer filters."
    return "\n\n".join(
        f"[ACN {h['acn']}] {h['date']} | {h['aircraft']} | phase: {h['flight_phase']}\n{h['text'][:600]}"
        for h in hits)


@tool
def get_report(acn: str) -> str:
    """Fetch the full text of one report by its ACN."""
    r = R.get_report(acn)
    if not r:
        return "Report not found."
    return f"[ACN {r['acn']}] {r['date']} | {r['aircraft']}\n{r['text']}"


@tool
def stats_summary(field: str = "anomaly", aircraft: str = None,
                  flight_phase: str = None, anomaly: str = None, state: str = None) -> str:
    """Count reports grouped by a field (anomaly, flight_phase, aircraft, state, date,
    time_of_day), with optional filters. Use for 'how many' or 'most common' questions."""
    return str(R.stats(field=field, aircraft=aircraft, flight_phase=flight_phase,
                       anomaly=anomaly, state=state))


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
agent = create_react_agent(llm, [search_reports, get_report, stats_summary], prompt=SYSTEM)

if __name__ == "__main__":
    print("SkyLens ready. Type 'quit' to exit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [("user", q)]})
        print("\nSkyLens:", result["messages"][-1].content)