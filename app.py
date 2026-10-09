import re
import sys
import streamlit as st

sys.path.insert(0, "src")
st.set_page_config(page_title="SkyLens", page_icon="✈️", layout="wide")


@st.cache_resource(show_spinner="Loading SkyLens (first start only)...")
def load():
    import retriever as R
    from agent import agent
    R._all_metas()          # warm the metadata cache so the Compare tab is instant
    return R, agent


R, agent = load()
from charts import extract_charts, render_chart, compare_tab

ACN_RE = re.compile(r"ACN\s*(\d{6,8})")

st.title("✈️ SkyLens: Aviation Incident Investigator")
st.caption("Answers grounded in NASA ASRS reports. ASRS data is voluntary and "
           "anecdotal, so counts are not accident rates and this is not for real flight decisions.")


def text_of(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def render_citations(acns):
    if not acns:
        return
    st.markdown("**Cited reports**")
    for acn in acns:
        r = R.get_report(acn)
        if r:
            with st.expander(f"ACN {acn} | {r['date']} | {r['aircraft']}"):
                st.write(r["text"])
        else:
            st.warning(f"ACN {acn} was cited but not found in the database.")


# ---- Session state ----
if "messages" not in st.session_state:
    st.session_state.messages = []   # each: {"role", "content", "acns", "charts"}

# ---- Sidebar filters ----
st.sidebar.header("Filters (optional)")
years = st.sidebar.slider("Year range", 2022, 2026, (2022, 2026))
aircraft = st.sidebar.text_input("Aircraft contains", placeholder="Cessna")
phase = st.sidebar.text_input("Flight phase contains", placeholder="Landing")
st.sidebar.caption("Data covers 2022 to 2026. 2026 is a partial year.")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []

# ---- Tabs ----
tab_chat, tab_compare = st.tabs(["Chat", "Compare & Trends"])

with tab_compare:
    compare_tab()

with tab_chat:
    # Show history
    for idx, m in enumerate(st.session_state.messages):
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant":
                render_citations(m.get("acns", []))
                for i, p in enumerate(m.get("charts", [])):
                    render_chart(p, key=f"h{idx}_{i}")

    # New question
    if prompt := st.chat_input("Ask about aviation incidents..."):
        st.session_state.messages.append(
            {"role": "user", "content": prompt, "acns": [], "charts": []})
        with st.chat_message("user"):
            st.markdown(prompt)

        hints = []
        if years != (2022, 2026):   # only send a year filter if the user narrowed it
            hints.append(f"year_from={years[0]}, year_to={years[1]}")
        if aircraft:
            hints.append(f"aircraft='{aircraft}'")
        if phase:
            hints.append(f"flight_phase='{phase}'")
        # only the last 4 messages, so requests stay under the free-tier token limit
        # skip failed turns, so an error message does not confuse the next answer
        good = [m for m in st.session_state.messages[:-1] if not m.get("error")]
        history = [(m["role"], m["content"][:1500]) for m in good][-4:]
        note = f"\n\n[User-selected filters to apply in tool calls: {', '.join(hints)}]" if hints else ""
        history.append(("user", prompt + note))

        with st.chat_message("assistant"):
            labels = {"search_reports": "Searching reports", "get_report": "Reading a full report",
                      "stats_summary": "Counting reports", "compare_groups": "Building comparison chart",
                      "trend_chart": "Building trend chart"}
            error = False
            for attempt in range(2):   # gpt-oss sometimes invents a tool name; one retry fixes it
                try:
                    with st.status("Thinking...", expanded=False) as status:
                        result = None
                        # stream shows each step as it happens instead of one long wait
                        for state in agent.stream({"messages": history}, stream_mode="values",
                                                  config={"recursion_limit": 10}):
                            result = state
                            for c in getattr(state["messages"][-1], "tool_calls", None) or []:
                                label = labels.get(c["name"], c["name"])
                                status.update(label=label + "...")
                                status.write(f"{label}: {c['args']}")
                        status.update(label="Done", state="complete")
                    answer = text_of(result["messages"][-1].content)
                    charts = extract_charts(result["messages"])
                    error = False
                    break
                except Exception as e:
                    answer = f"Something went wrong: {e}"
                    charts = []
                    error = True
                    if "tool_use_failed" not in str(e):
                        break
            st.markdown(answer)
            acns = list(dict.fromkeys(ACN_RE.findall(answer)))
            render_citations(acns)
            for i, p in enumerate(charts):
                render_chart(p, key=f"h{len(st.session_state.messages)}_{i}")  # same key as history, so toggles keep their state

        if error:   # mark the failed question too, so neither is sent to the model again
            st.session_state.messages[-1]["error"] = True
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "acns": acns, "charts": charts,
             "error": error})