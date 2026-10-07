import json, re, sys, time
sys.path.insert(0, "src")
import retriever as R

RUN_AGENT = "--no-agent" not in sys.argv
if RUN_AGENT:
    from agent import agent

ACN_RE = re.compile(r"ACN\s*(\d{6,8})")


def text_of(content):
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def retrieval_precision(q):
    hits = R.search(q["question"], n=5, **q.get("filters", {}))
    if not hits:
        return 0.0
    kws = [k.lower() for k in q["keywords"]]
    ok = sum(any(k in h["text"].lower() for k in kws) for h in hits)
    return ok / len(hits)


def run_agent(q):
    result = agent.invoke({"messages": [("user", q["question"])]})
    msgs = result["messages"]
    answer = text_of(msgs[-1].content)
    retrieved = set()
    for m in msgs:
        if m.type == "tool":
            retrieved |= set(ACN_RE.findall(text_of(m.content)))
    cited = set(ACN_RE.findall(answer))
    return answer, cited, retrieved


results = []
questions = json.load(open("eval/questions.json"))
for q in questions:
    row = {"id": q["id"], "type": q["type"], "question": q["question"]}
    if q["type"] in ("semantic", "filtered"):
        row["retrieval_precision@5"] = retrieval_precision(q)
    if RUN_AGENT:
        try:
            answer, cited, retrieved = run_agent(q)
            row["answer"] = answer
            if q["type"] != "stats":
                row["cited"] = len(cited)
                row["fabricated_citations"] = sorted(cited - retrieved)
                row["has_citation"] = len(cited) > 0
            else:
                row["answer_correct"] = all(k.lower() in answer.lower()
                                            for k in q["answer_keywords"])
        except Exception as e:
            row["error"] = str(e)
        time.sleep(3)  # stay under free-tier rate limits
    results.append(row)
    print(row["id"], {k: v for k, v in row.items() if k not in ("answer", "question")})

json.dump(results, open("eval/results.json", "w"), indent=2)

# summary
ret = [r["retrieval_precision@5"] for r in results if "retrieval_precision@5" in r]
print("\n=== SUMMARY ===")
if ret:
    print(f"Retrieval precision@5 (keyword): {sum(ret)/len(ret):.2f}")
if RUN_AGENT:
    cit = [r for r in results if "has_citation" in r]
    if cit:
        print(f"Answers with citations: {sum(r['has_citation'] for r in cit)}/{len(cit)}")
        print(f"Answers with fabricated citations: {sum(bool(r['fabricated_citations']) for r in cit)}/{len(cit)}")
    st = [r for r in results if "answer_correct" in r]
    if st:
        print(f"Stats questions correct: {sum(r['answer_correct'] for r in st)}/{len(st)}")