# SkyLens: Aviation Incident Investigator Agent

SkyLens is a RAG + agentic AI project that answers questions about aviation safety incidents. It runs over NASA ASRS (Aviation Safety Reporting System) reports and returns answers grounded in retrieved reports, with cited report IDs (ACNs). It also draws comparison and trend charts across aircraft, flight phases, states and years.

## What it does

Ask something like *"What are common causes of landing gear failures in small aircraft?"* and the agent will:

1. Interpret the question and pick optional filters (year range, aircraft, flight phase, state).
2. Run **hybrid retrieval**: when a filter is set, it first finds the matching reports from metadata, then runs semantic search only inside them.
3. Re-search with a refined query if the first results are thin or off-topic.
4. Return a synthesized answer with citations like `[ACN 1884889]`, each expandable to the full report in the UI.

Ask *"Compare engine problems in Cessna vs Piper aircraft"* and it also draws a grouped bar chart, with a toggle between raw counts and % of group so differently sized groups compare fairly.

### Agent tools

| Tool | Purpose |
|---|---|
| `search_reports(query, filters)` | Semantic + metadata search (year range, state, aircraft, phase, anomaly), one result per report |
| `get_report(acn)` | Fetch the full text of a single report |
| `stats_summary(field, filters)` | Counts and "most common" questions over the data |
| `compare_groups(filter_type, values, field)` | Side-by-side comparison of 2-4 groups, rendered as a bar chart |
| `trend_chart(filter_type, values)` | Monthly report counts as a line chart, one line per group |

## Dataset

- **Source:** [NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html), exported manually as CSV into the `Dataset/` folder.
- **Coverage:** 2022 to 2026. 2026 is a partial year, so compare full years only when judging trends. Months with no loaded data would show as gaps in trend charts, not as zero incidents.
- **Size:** 24,464 unique reports in the index (chunked for embedding): 2022: 5,451, 2023: 4,590, 2024: 5,427, 2025: 6,127, 2026 (partial): about 2,869.
- **Fields used:** ACN, date, aircraft make/model, flight phase, anomaly labels, state, local time of day, synopsis, and narratives (Report 1 and Report 2).

> ASRS reports are voluntary and anecdotal. A higher count means more reports, not more danger, and counts are **not** accident rates. The agent and the charts say so.

## Architecture

```
ASRS CSVs (Dataset/)
      |
      v
 ingest.py  -> clean, chunk (1500 chars, 200 overlap), embed in batches
      |
      v
 embedder.py  -> configurable embedding model (EMBED_MODEL), GPU if available
      |
      v
 ChromaDB (chroma_db/)  <-- one collection per embedding model
      |
      v
 retriever.py  -> search / get_report / stats / compare / trend
      |
      v
 agent.py (LangGraph ReAct agent + 5 tools + LLM)
      |
      v
 app.py (Streamlit: Chat tab + Compare & Trends tab)
```

## Tech stack

- **Python**, pandas
- **ChromaDB** for vector storage
- **sentence-transformers** for embeddings, running on an NVIDIA GPU (CUDA PyTorch) with CPU fallback
- **LangGraph** for the agent loop
- **Streamlit** and **Plotly** for the UI and charts
- **LLM:** Groq `openai/gpt-oss-120b` (free tier), set with `LLM_MODEL`. The `llm` line can be swapped for Gemini, Ollama or the Claude API.

## Project structure

```
GEN AI Project/
├── Dataset/              # raw ASRS CSV exports (not committed)
├── chroma_db/            # vector database, created by ingest.py (not committed)
├── eval/
│   ├── questions.json    # evaluation questions
│   ├── run_eval.py       # evaluation script
│   └── results.json      # output of the last run
├── src/
│   ├── embedder.py       # embedding model config (model, prefixes, device, batch size)
│   ├── ingest.py         # load, clean, chunk, embed, store
│   ├── retriever.py      # search, get_report, stats, compare, trend
│   ├── charts.py         # Plotly chart rendering and the Compare tab
│   ├── agent.py          # LangGraph agent and tools
│   └── test.py           # quick retrieval sanity check
├── app.py                # Streamlit UI
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
   Suggested `requirements.txt` (generate yours with `python -m pip freeze > requirements.txt`):
   ```
   chromadb
   sentence-transformers
   torch
   langgraph
   langchain-core
   pandas
   plotly
   streamlit
   # plus one LLM provider:
   langchain-groq            # or langchain-google-genai, langchain-ollama, langchain-anthropic
   ```
3. **Optional, for GPU embedding on Windows:** a plain install usually gives CPU-only PyTorch. Install the CUDA build from [pytorch.org](https://pytorch.org/get-started/locally), then check:
   ```powershell
   python -c "import torch; print(torch.cuda.is_available())"
   ```
4. **Add data:** export ASRS results as CSV and place the files in `Dataset/`.
5. **Choose an embedding model** (default is `bge-small`) and build the index (prints `Done.` when finished):
   ```powershell
   $env:EMBED_MODEL="bge-small"
   python src\ingest.py
   ```
   Options: `bge-small`, `bge-base`, `qwen3-0.6b`, `embeddinggemma`. Each model gets its own collection. Use the **same** `EMBED_MODEL` in every terminal that runs the agent, evaluation or app.
6. **Check retrieval:**
   ```powershell
   python src\test.py
   ```
7. **Run the app** (set the API key for your LLM provider first):
   ```powershell
   $env:GROQ_API_KEY="your-key-here"
   $env:LLM_MODEL="openai/gpt-oss-120b"   # optional, this is the default
   streamlit run app.py
   ```
   To see which Groq models your key can use:
   ```powershell
   python -c "from groq import Groq; print(sorted(m.id for m in Groq().models.list().data))"
   ```
   Or use the terminal agent with `python src\agent.py`.

Never commit API keys. `.gitignore` should contain `.venv/`, `chroma_db/`, `Dataset/`, `__pycache__/`, `*.pyc` and `.env`.

### Example questions

- Why do landing gears fail to extend?
- What weather-related incidents happened in 2023?
- Show me Boeing 787 incidents involving landing gear.
- What are the most common anomaly categories in the data?
- Compare the most common anomalies in Cessna vs Piper aircraft.
- How did gear-related reports change between 2022 and 2025?
- Compare landing vs approach incidents by year.

## Features

- **Chat with citations:** every `[ACN ...]` expands into the full report. A cited ACN that is not in the database triggers a warning, which makes fabrication visible.
- **Conversation memory:** follow-ups such as "now only in 2025" work.
- **Sidebar filters:** year range, aircraft and flight phase are passed to the agent as hints.
- **Compare & Trends tab:** pick groups (aircraft, flight phase, state, year) and get bar or trend charts with no LLM call, so it is free and never rate-limited.
- **% of group toggle:** fair comparison between groups of different sizes.
- **Honest gaps:** months with no loaded data show as gaps, not zeros, and 2026 is flagged as a partial year.
- **Live progress:** each agent step ("Searching reports...", "Building comparison chart...") is shown as it happens, with the tool arguments in an expandable box.
- **Fast reloads:** the embedding model, database and report metadata load once per app start, so tab switches and the Compare tab are instant.
- **Grounding rules:** the agent only states causes that a report states ("cause not stated" otherwise), labels general aviation knowledge separately, and points to NTSB for accidents, since ASRS is not a crash database.

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

2022-only dataset, `bge-small`:

| Change | Keyword precision@5 |
|---|---|
| Baseline (post-filter on top 40 candidates) | 0.89 |
| Wider candidate pool (500) + alternative filter terms (`Piper\|PA-`) | 0.97 |

Expanded dataset (2022 to 2026, 24,464 reports), measured on a larger index, so **not directly comparable** to the rows above:

| Embedding model | Precision@5 | Embed time | Notes |
|---|---|---|---|
| bge-small | 0.98 (17 questions, 13 retrieval) | *to add* | baseline, 384-dim |
| bge-base | *to add* | *to add* | |
| qwen3-0.6b | *to add* | *to add* | |
| embeddinggemma | *to add* | *to add* | gated model, needs Hugging Face license |

Agent-level results (citation rate, fabricated citations, stats accuracy): *to be added after the full agent run.*

### Engineering notes

- **Filter recall bug:** an aircraft filter for "Piper" returned nothing, because many ASRS entries list Piper aircraft by model name (for example `PA-28`) and the filter ran on only the top 40 candidates. Fixed by widening the candidate pool and supporting `|`-separated alternative terms.
- **Multi-label anomaly field:** ASRS stores several anomaly labels per report, separated by `;`. The stats tool splits them and counts each label once per report, so the counts are meaningful.
- **Chunk deduplication:** long reports are split into chunks, but search returns one result per report.
- **Scaling bug ("too many SQL variables"):** after the dataset grew to about 44k chunks, fetching all metadata in one Chroma `get()` failed. Fixed by paging the metadata in batches of 2,000 and caching it in memory, which also makes the Compare tab and trend charts fast.
- **CPU-only PyTorch:** the default Windows install ran embedding on the CPU (GPU at 0% utilization). Installing the CUDA build of PyTorch moved embedding to the GPU (RTX 3050, 4 GB VRAM), using small batches and a 256-token max length to fit in memory.
- **Configurable embeddings:** model, query/document prompts, dtype and collection name live in one file (`embedder.py`), so models can be compared on the same evaluation set without touching the rest of the code.

**Answer-quality fixes found by testing the chat**

- **Rare aircraft were missed:** searching the top 500 chunks across all aircraft and then filtering for B787 could return nothing. Search now pre-filters by metadata (a `$in` list of matching ACNs) and only falls back to the wide search for very common filters (over 3,000 reports).
- **Aircraft naming:** users write "Boeing 787" or "Airbus A320", but ASRS stores `B787` and `A320`. The filter now expands these aliases.
- **Invented causes (hallucination):** asking the same question twice gave two different causes for the same report (ACN 1966647), because the snippet never stated a cause. The prompt now forbids attributing causes a report does not state.
- **Misleading counts:** filtered stats listed co-occurring anomaly labels as if they were sub-types, and their counts summed past the total. The stats tool now says that labels overlap, and the prompt requires explaining this.
- **Wrong data coverage:** the agent called 2023 a partial year. The prompt now states that 2022 to 2025 are full years and only 2026 is partial.
- **Citations in tables:** bare report numbers in tables were not recognised as citations. The prompt now requires the `ACN` prefix everywhere.
- **Hallucinated tool calls:** gpt-oss sometimes calls built-in tools it was trained with (for example `repo_browser.open_file`), which Groq rejects. Fixed with an explicit tool allow-list in the prompt and one automatic retry.
- **Error poisoning the history:** after a failed turn, the next question was answered as if it were the previous one. Failed turns are now kept out of the history sent to the model.

**Working within free-tier limits (Groq, 8,000 tokens per minute)**

- **Model access:** `llama-3.3-70b-versatile` returned 404 for this account, so the app moved to `openai/gpt-oss-120b`, chosen from the models the key could list.
- **Request too large (413):** chart tools used to return the full chart data to the LLM. They now store it on the side and return a 2-4 line summary. Search snippets were cut to 400 characters, full reports to 2,500, and history to the last 4 messages.
- **Speed:** `reasoning_effort="low"`, at most 3 tool calls per question (hard stop at 10 steps), `max_tokens=1500` and automatic retries on rate limits.
- **Streamlit reruns:** the Compare chart disappeared when the % / raw toggle was clicked. Chart data is now kept in session state, and chat charts use stable keys.

## Limitations

- Keyword precision is a lenient proxy for relevance and is measured on a small question set.
- Filters exclude reports where the field is empty, which is common in ASRS (for example, ground or maintenance events).
- 2026 is a partial year, so cross-year trends need care.
- Free-tier LLMs have rate limits and can be less careful with citations. On Groq's free tier, back-to-back questions can pause while the per-minute token limit resets.
- Filters work by year, not by month.
- Free hosting has no GPU, so a deployed app embeds each query on CPU. A smaller model may be the better deployment choice even if a larger one scores higher locally.

## Roadmap

- [x] Data ingestion and embedding
- [x] Hybrid retrieval with metadata filters
- [x] Agent with tools
- [x] Evaluation harness (retrieval precision)
- [x] Expanded dataset (2022 to 2026, 24,464 reports)
- [x] Streamlit UI with clickable citations, conversation memory and live progress
- [x] Comparison and trend charts (tested in the UI)
- [x] GPU embedding and configurable embedding models
- [x] Rebuilt index on GPU, updated eval ground truth, retrieval precision 0.98
- [x] Chat tested end to end; grounding, citation and filter fixes applied
- [ ] Re-test chat after the latest fixes (B787 search, tool retry)
- [ ] Embedding model comparison table
- [ ] Full agent evaluation (citations and stats accuracy)
- [ ] Month filter (`date_from` / `date_to`)
- [ ] Reranker (cross-encoder), measured in the eval table
- [ ] Hybrid BM25 search, "find similar incidents", PDF briefing export
- [ ] Deployment (Hugging Face Spaces or Streamlit Community Cloud)
- [ ] Architecture diagram and a 2-minute demo video
- [ ] Stretch: NTSB data, feedback logging, topic map

## Data and license note

ASRS data is published by NASA. Review the terms on the ASRS website before redistributing any raw data. This repository should not include the raw CSVs.
