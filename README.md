# SkyLens: Aviation Incident Investigator Agent

SkyLens is a RAG + agentic AI project that answers questions about aviation safety incidents. It runs over NASA ASRS (Aviation Safety Reporting System) reports and returns answers grounded in retrieved reports, with cited report IDs (ACNs).

## What it does

Ask something like *"What are common causes of landing gear failures in small aircraft?"* and the agent will:

1. Interpret the question and pick optional filters (aircraft, flight phase, state).
2. Run **hybrid retrieval**: semantic search over report text plus metadata filters.
3. Re-search with a refined query if the first results are thin or off-topic.
4. Return a synthesized answer with citations like `[ACN 1884889]`.

### Agent tools

| Tool | Purpose |
|---|---|
| `search_reports(query, filters)` | Semantic + metadata search, one result per report |
| `get_report(acn)` | Fetch the full text of a single report |
| `stats_summary(field, filters)` | Counts and "most common" questions over the data |

## Dataset

- **Source:** [NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html), exported manually as CSV.
- **Coverage:** calendar year 2022, four CSV files of three months each (`ASRS_01_03_22`, and so on).
- **Size:** 5,452 unique reports after deduplicating by ACN.
- **Fields used:** ACN, date, aircraft make/model, flight phase, anomaly labels, state, local time of day, synopsis, and narratives (Report 1 and Report 2).

> ASRS reports are voluntary and anecdotal. Counts from this data are **not** true accident rates, and the agent is instructed to say so.

## Architecture

```
ASRS CSVs (DATA/)
      |
      v
 ingest.py  -> clean, chunk (1500 chars, 200 overlap), embed (bge-small-en-v1.5)
      |
      v
 ChromaDB (chroma_db/)  <-- cosine similarity + metadata
      |
      v
 retriever.py  -> search / get_report / stats
      |
      v
 agent.py (LangGraph ReAct agent + 3 tools + LLM)
      |
      v
 Cited answer
```

## Tech stack

- **Python**, pandas
- **ChromaDB** for vector storage
- **sentence-transformers** (`BAAI/bge-small-en-v1.5`) for embeddings, running locally
- **LangGraph** for the agent loop
- **LLM:** configurable. Works with free options (Groq, Gemini, Ollama) or the Claude API.

## Project structure

```
GEN AI Project/
├── DATA/                 # raw ASRS CSV exports (not committed)
├── chroma_db/            # vector database, created by ingest.py (not committed)
├── eval/
│   ├── questions.json    # evaluation questions
│   ├── run_eval.py       # evaluation script
│   └── results.json      # output of the last run
├── src/
│   ├── ingest.py         # load, clean, chunk, embed, store
│   ├── retriever.py      # search, get_report, stats
│   ├── agent.py          # LangGraph agent and tools
│   └── test.py           # quick retrieval sanity check
├── requirements.txt
└── README.md
```

## Getting started

Run every command from the **project root**, because the scripts use relative paths.

1. **Create and activate a virtual environment** (Windows PowerShell):
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```powershell
   python -m pip install -r requirements.txt
   ```
   Suggested `requirements.txt`:
   ```
   chromadb
   sentence-transformers
   langgraph
   langchain-core
   pandas
   # plus one LLM provider:
   langchain-groq            # or langchain-google-genai, langchain-ollama, langchain-anthropic
   ```
3. **Add data:** export ASRS results as CSV and place the files in `DATA/`.
4. **Build the index** (takes a few minutes on CPU; prints `Done.` when finished):
   ```powershell
   python src\ingest.py
   ```
5. **Check retrieval:**
   ```powershell
   python src\test.py
   ```
6. **Run the agent** (set the API key for your chosen provider first):
   ```powershell
   $env:GROQ_API_KEY="your-key-here"
   python src\agent.py
   ```

Never commit API keys. Add `.env`, `chroma_db/`, and `DATA/` to `.gitignore`.

### Example questions

- What are common causes of landing gear failures in small aircraft?
- What are the most common anomaly categories in the data?
- Show me incidents where the crew went around because of gear indication problems.

## Evaluation

Run:

```powershell
python eval\run_eval.py --no-agent   # retrieval only, free and instant
python eval\run_eval.py              # full run including the agent
```

The evaluation measures:

- **Keyword precision@5:** of the top 5 retrieved reports, the share containing an expected keyword.
- **Fabricated citations:** whether the agent cites an ACN it never retrieved (target: 0).
- **Stats correctness:** whether stats answers match ground truth computed directly from the data.

### Results

| Change | Keyword precision@5 |
|---|---|
| Baseline (post-filter on top 40 candidates) | 0.89 |
| Wider candidate pool (500) + alternative filter terms (`Piper\|PA-`) | 0.97 |

Agent-level results (citation rate, fabricated citations, stats accuracy): *to be added after the full agent run.*

### Engineering notes

- **Filter recall bug:** an aircraft filter for "Piper" returned nothing, because many ASRS entries list Piper aircraft by model name (for example `PA-28`) and the filter ran on only the top 40 candidates. Fixed by widening the candidate pool and supporting `|`-separated alternative terms.
- **Multi-label anomaly field:** ASRS stores several anomaly labels per report, separated by `;`. The stats tool splits them and counts each label once per report, so the counts are meaningful.
- **Chunk deduplication:** long reports are split into chunks, but search returns one result per report.

## Limitations

- Keyword precision is a lenient proxy for relevance and is measured on a small question set (13 retrieval questions).
- Filters exclude reports where the field is empty, which is common in ASRS (for example, ground or maintenance events).
- Only 2022 data is indexed, so year filters are not yet useful.
- Free-tier LLMs have rate limits and can be less careful with citations.

## Roadmap

- [x] Data ingestion and embedding
- [x] Hybrid retrieval with metadata filters
- [x] Agent with three tools
- [x] Evaluation harness (retrieval precision)
- [ ] Full agent evaluation (citations and stats accuracy)
- [ ] Expand the question set to 30+
- [ ] Streamlit UI
- [ ] Deployment (Hugging Face Spaces or Streamlit Community Cloud)
- [ ] Architecture diagram and a 2-minute demo video
- [ ] Stretch: add NTSB accident data, try a larger embedding model (`bge-base-en-v1.5`)

## Data and license note

ASRS data is published by NASA. Review the terms on the ASRS website before redistributing any raw data. This repository should not include the raw CSVs.
