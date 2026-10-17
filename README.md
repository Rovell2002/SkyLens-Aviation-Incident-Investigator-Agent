# SkyLens: Aviation Incident Investigator Agent

SkyLens is a RAG + agentic AI project that answers questions about aviation safety incidents. It runs over NASA ASRS (Aviation Safety Reporting System) reports and returns answers grounded in retrieved reports, with cited report IDs (ACNs). It also draws comparison and trend charts across aircraft, flight phases, states and years.

## What it does

Ask something like *"Why do landing gears fail to extend?"* and the agent will:

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

- **Source:** [NASA ASRS Database Online](https://asrs.arc.nasa.gov/search/database.html), exported manually as CSV into the `data/` folder.
- **Coverage:** 2022 to 2026. 2026 is a partial year, so compare full years only when judging trends. Months with no loaded data would show as gaps in trend charts, not as zero incidents.
- **Size:** 24,464 unique reports in the index (chunked for embedding): 2022: 5,451, 2023: 4,590, 2024: 5,427, 2025: 6,127, 2026 (partial): about 2,869.
- **Fields used:** ACN, date, aircraft make/model, flight phase, anomaly labels, state, local time of day, synopsis, and narratives (Report 1 and Report 2).

> ASRS reports are voluntary and anecdotal. A higher count means more reports, not more danger, and counts are **not** accident rates. The agent and the charts say so.

## Architecture

```
ASRS CSVs (data/)
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
 retriever.py  -> search (+ cross-encoder rerank of top 30) / get_report / stats / compare / trend
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
- **sentence-transformers** for embeddings (`bge-small-en-v1.5`) and a cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`), running on an NVIDIA GPU (CUDA PyTorch) with CPU fallback
- **LangGraph** for the agent loop
- **Streamlit** and **Plotly** for the UI and charts
- **LLM:** Groq `openai/gpt-oss-120b` (free tier), set with `LLM_MODEL`. The `llm` line can be swapped for Gemini, Ollama or the Claude API.

## Project structure

```
GEN AI Project/
├── data/                 # raw ASRS CSV exports (not committed)
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
   Suggested `requirements.txt`. On Windows, generate yours with `python -m pip freeze | Out-File -Encoding utf8 requirements.txt`, because PowerShell's `>` saves UTF-16, which Git treats as binary:
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
4. **Add data:** export ASRS results as CSV and place the files in `data/`.
5. **Choose an embedding model** (default is `bge-small`) and build the index (prints `Done.` when finished):
   ```powershell
   $env:EMBED_MODEL="bge-small"
   python src\ingest.py
   ```
   Options: `bge-small`, `bge-base`, `qwen3-0.6b`, `embeddinggemma`. Each model gets its own collection. Use the **same** `EMBED_MODEL` in every terminal that runs the agent, evaluation or app.

   The reranker is on by default. Choose it with `$env:RERANKER="minilm"` (default), `"bge"` or `"none"`. It needs no re-indexing.
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

Never commit API keys. Keep them only in environment variables, never in a file inside the project. `.gitignore` should contain `.venv/`, `chroma_db/`, `data/`, `__pycache__/`, `*.pyc`, `.env`, `API.txt` and `*.key`.

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
- **Grounding rules:** the agent only states causes that a report states ("cause not stated" otherwise), labels general aviation knowledge separately, copies field values exactly, flags small samples (under 50 reports), gives no fleet recommendations, treats each question as standalone, and points to NTSB for accidents, since ASRS is not a crash database.

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

The first 13 retrieval questions saturated (bge-small scored 0.98), so 8 paraphrased questions were added that describe an event without its usual keyword (for example "wrong barometric pressure" instead of "altimeter setting"). The set was then frozen and every model was tested on the same 21 retrieval questions.

| Embedding model | Precision@5 (original 13) | Precision@5 (all 21) | Embed time | Notes |
|---|---|---|---|---|
| bge-small | 0.98 | 0.90 | 4.7 min (RTX 3050) | baseline, 384-dim |
| bge-base | 1.00 | 0.93 | 9.0 min (RTX 3050) | 768-dim; hard questions 18/19/25: 0.6/0.8/0.2 (bge-small: 0.4/0.6/0.0) |
| qwen3-0.6b | 1.00 | 0.94 | 51.7 min (RTX 3050) | 1024-dim; hard questions 18/19/25: 0.6/0.6/0.8, but question 22 dropped to 0.8 |

**Reading the table:** with 21 questions, one question moving by 0.2 changes the average by only about 0.01, so bge-base (0.93) and Qwen3 (0.94) are effectively tied overall. The clear difference is question 25 ("wrong barometric pressure"): Qwen3 understood it (0.8) where bge-small (0.0) and bge-base (0.2) matched the word "pressure". That gain cost 11 times the build time of bge-small, and a 0.6B model is slow to embed queries on a CPU-only free host. So Qwen3 is not used for deployment. The choice between bge-small and bge-base is made after testing a cross-encoder reranker, which may fix the same kind of miss far more cheaply.
| embeddinggemma | not run | not run | not run | gated model, needs Hugging Face license |

**Reranking.** A cross-encoder reads the question and each candidate report together, which embeddings cannot do. It re-orders the top 30 candidates from the embedding search. Same 21 questions, bge-small embeddings, RTX 3050:

| Setup | Original 13 | All 21 | Q18 / Q19 / Q22 / Q23 / Q25 | Avg search time |
|---|---|---|---|---|
| bge-small, no reranker | 0.98 | 0.90 | 0.4 / 0.6 / 1.0 / 1.0 / 0.0 | 254 ms |
| bge-small + `ms-marco-MiniLM-L-6-v2` (22M) | 1.00 | **0.95** | 0.8 / 1.0 / 1.0 / 0.8 / 0.4 | 458 ms |
| bge-small + `bge-reranker-base` (278M) | 1.00 | 0.91 | 0.8 / 0.6 / 0.8 / 1.0 / 0.0 | about 5 s* |

\*Measured while another evaluation shared the GPU, and the 1.1 GB model in full precision is heavy for a 4 GB card, so this time is inflated. The accuracy result stands regardless.

**Decision:** bge-small + MiniLM reranker is the default. It scores the best overall (0.95, above Qwen3's 0.94), costs about 200 ms per search, builds in 4.7 minutes, and is small enough for a CPU-only free host. The 12 times larger reranker was *not* better here, and no setup fully fixes question 25 except Qwen3 (0.8). The baseline run reproduced last week's scores exactly (0.90, same per-question values), so the evaluation is deterministic.

Agent-level results (Groq `openai/gpt-oss-120b`, bge-small, original 17 questions):

| Metric | Result |
|---|---|
| Retrieval precision@5 (keyword) | 0.98 |
| Answers with at least one citation | 13/13 |
| Answers with a citation not found in the retrieved reports | 1/13 (the model wrote `ACN 202305` for a 2023 Piper fuel report: a mangled ACN, caught by the check and by the app's "not found in the database" warning) |
| Stats answers matching ground truth | 4/4 |

The fabricated-citation check only verifies that each cited ACN was actually retrieved by a tool. It does not check that the claim matches the report text, so it is a lower bound on hallucination, not a full faithfulness score.

### Engineering notes

- **Filter recall bug:** an aircraft filter for "Piper" returned nothing, because many ASRS entries list Piper aircraft by model name (for example `PA-28`) and the filter ran on only the top 40 candidates. Fixed by widening the candidate pool and supporting `|`-separated alternative terms.
- **Multi-label anomaly field:** ASRS stores several anomaly labels per report, separated by `;`. The stats tool splits them and counts each label once per report, so the counts are meaningful.
- **Chunk deduplication:** long reports are split into chunks, but search returns one result per report.
- **Scaling bug ("too many SQL variables"):** after the dataset grew to about 44k chunks, fetching all metadata in one Chroma `get()` failed. Fixed by paging the metadata in batches of 2,000 and caching it in memory, which also makes the Compare tab and trend charts fast.
- **CPU-only PyTorch:** the default Windows install ran embedding on the CPU (GPU at 0% utilization). Installing the CUDA build of PyTorch moved embedding to the GPU (RTX 3050, 4 GB VRAM), using small batches and a 256-token max length to fit in memory. A full bge-small rebuild went from about 35-40 minutes on the CPU (about 1,200 chunks per minute) to 4.7 minutes on the GPU, about 8 times faster.
- **Evaluation saturation:** the first question set scored 0.98, which could not separate models. Paraphrased questions were added, the set was frozen, and only then were models compared.
- **Semantic confusion found by the hard questions:** for "pilot set the wrong barometric pressure and flew lower than intended", bge-small returned cabin-pressure, engine-pressure and fuel reports (precision 0.0). It matched the word "pressure" instead of the meaning (altimeter setting). This is the kind of miss a stronger embedding model or a reranker should fix.
- **Configurable embeddings:** model, query/document prompts, dtype and collection name live in one file (`embedder.py`), so models can be compared on the same evaluation set without touching the rest of the code.

**Answer-quality fixes found by testing the chat**

- **Rare aircraft were missed:** asking for Boeing 787 landing gear incidents returned "no reports", even though the index holds 256 B787 reports. Searching the top 500 chunks across all aircraft and then filtering for B787 left nothing. Search now pre-filters by metadata (a `$in` list of matching ACNs) and only falls back to the wide search for very common filters (over 3,000 reports). After the fix, the same query returns five relevant B787 gear reports, including nose and main gear failing to retract after takeoff (ACN 1892506, ACN 1893025) and main gear hardware findings (ACN 2222769, ACN 2315568).
- **Aircraft naming:** users write "Boeing 787", "Airbus A320" or "Piper", but ASRS stores `B787`, `A320` and `PA-28`. A Cessna vs Piper comparison first counted only 21 Piper reports for this reason. The filter now expands these aliases.
- **Invented causes (hallucination):** asking the same question twice gave two different causes for the same report (ACN 1966647), because the snippet never stated a cause. The prompt now forbids attributing causes a report does not state.
- **Misleading counts:** filtered stats listed co-occurring anomaly labels as if they were sub-types, and their counts summed past the total. The stats tool now says that labels overlap, and the prompt requires explaining this.
- **Wrong data coverage:** the agent called 2023 a partial year. The prompt now states that 2022 to 2025 are full years and only 2026 is partial.
- **Citations in tables:** bare report numbers in tables were not recognised as citations. The prompt now requires the `ACN` prefix everywhere.
- **Hallucinated tool calls:** gpt-oss sometimes calls built-in tools it was trained with (for example `repo_browser.open_file`), which Groq rejects. Fixed with an explicit tool allow-list in the prompt and one automatic retry.
- **Error poisoning the history:** after a failed turn, the next question was answered as if it were the previous one. Failed turns are now kept out of the history sent to the model.
- **Context carry-over:** a general question asked right after a B787 question was answered only for the B787. The prompt now treats each question as standalone unless the user clearly refers back.
- **Over-claiming:** answers excluded 2026 from lookups, claimed outcomes like "non-fatal" that no report stated, called something "most common" from 2 reports, and gave fleet advice from 21 reports. The prompt now limits the 2026 caveat to year comparisons and requires matching summaries, small-sample warnings and no recommendations.

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
- [x] Re-tested chat after the fixes (B787 search, grounding, Piper alias)
- [x] Full agent evaluation: citations 13/13, fabricated 1/13, stats 4/4
- [x] Harder evaluation questions and embedding comparison (bge-small vs bge-base)
- [x] Qwen3-0.6b result (0.94, fixes the "pressure" confusion, 51.7 min build)
- [ ] Citation guardrail: mark cited ACNs that were not retrieved as unverified
- [ ] Month filter (`date_from` / `date_to`)
- [x] Reranker (cross-encoder): MiniLM lifts bge-small from 0.90 to 0.95
- [ ] Hybrid BM25 search, "find similar incidents", PDF briefing export
- [ ] Deployment (Hugging Face Spaces or Streamlit Community Cloud)
- [ ] Architecture diagram and a 2-minute demo video
- [ ] Stretch: NTSB data, feedback logging, topic map

## Data and license note

ASRS data is published by NASA. Review the terms on the ASRS website before redistributing any raw data. This repository should not include the raw CSVs.
