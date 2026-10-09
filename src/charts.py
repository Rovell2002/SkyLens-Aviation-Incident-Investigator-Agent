import pandas as pd
import plotly.express as px
import streamlit as st
import retriever as R


def extract_charts(messages):
    """Find chart ids in the agent's tool results and fetch the full chart data."""
    from agent import CHART_STORE
    out = []
    for m in messages:
        if m.type == "tool" and isinstance(m.content, str) and m.content.startswith("CHART:"):
            cid = m.content[6:].split("\n", 1)[0].strip()
            if cid in CHART_STORE:
                out.append(CHART_STORE.pop(cid))
    return out


def render_chart(p, key):
    df = pd.DataFrame(p["rows"])
    if p["kind"] == "compare":
        mode = st.radio("Show", ["% of group", "Raw counts"], horizontal=True, key=f"mode_{key}")
        y = "pct" if mode.startswith("%") else "count"
        fig = px.bar(df, x="label", y=y, color="group", barmode="group", title=p["title"])
        fig.update_layout(xaxis_title="", xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True, key=f"fig_{key}")
        st.caption("Reports per group: " + ", ".join(f"{g}: {n}" for g, n in p["totals"].items())
                   + ". Groups differ in size, so % of group is the fairer comparison.")
    else:
        df["month"] = pd.to_datetime(df["month"], format="%Y%m")
        fig = px.line(df, x="month", y="count", color="group", markers=True, title=p["title"])
        st.plotly_chart(fig, use_container_width=True, key=f"fig_{key}")
        st.caption("Gaps mean no data was loaded for those months. They are not zero incidents. "
                   "2026 is a partial year.")


def compare_tab():
    st.subheader("Compare groups")
    c1, c2, c3 = st.columns(3)
    ftype = c1.selectbox("Compare by", ["aircraft", "flight_phase", "state", "year"])
    field = c2.selectbox("Count", ["anomaly", "flight_phase", "aircraft"])
    kind = c3.radio("Chart", ["Bar", "Trend"], horizontal=True)
    hint = {"year": "2022, 2023, 2024, 2025", "aircraft": "Cessna, Piper|PA-", "flight_phase": "Landing, Takeoff",
            "state": "CA, TX"}[ftype]
    raw = st.text_input("Values, separated by commas", value=hint,
                        help="Use | for alternatives, like Piper|PA-")
    if st.button("Compare"):
        values = [v.strip() for v in raw.split(",") if v.strip()]
        if len(values) < 2:
            st.warning("Enter at least two values.")
            return
        with st.spinner("Counting..."):
            p = R.compare(field, ftype, values) if kind == "Bar" else R.trend(ftype, tuple(values))
        st.session_state.compare_chart = p   # keep the chart when the page reruns
    if st.session_state.get("compare_chart"):
        render_chart(st.session_state.compare_chart, key="tab")