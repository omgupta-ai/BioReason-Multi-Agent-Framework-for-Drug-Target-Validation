# BioReason-Multi-Agent-Framework-for-Drug-Target-Validation
An autonomous multi-agent AI framework using LangGraph and Groq (Llama 3.3) to validate medical drug targets, featuring an automated LLM-as-a-Judge evaluator that scores reports for hallucination and precision against raw tool data.

Built with **LangGraph** and **Groq (Llama 3.3 70B)**, BioReason tackles one of the biggest challenges in medical AI: **hallucinations**. It enforces strict tool-calling, real-time API grounding, and uses an automated "LLM-as-a-Judge" evaluation loop that scores generated reports against raw tool output on hallucination and precision dimensions, flagging any report that falls below a configurable pass threshold.

![Architecture](Architecture.png)
---

## 🚀 The Problem & The Solution
Standard LLMs often hallucinate citations or invent molecular binding affinities when asked complex biological questions. 

**BioReason** addresses this by constraining the LLM to act as a reasoning engine over verified data. It queries real scientific databases, extracts exact data points, formats a clinical summary, and then grades its own output against the raw API data, surfacing any cited PMIDs or numeric values that do not appear in the underlying tool responses.

## 🧠 System Architecture



The pipeline is built on a directed acyclic graph (DAG) using LangGraph, consisting of the following core components:

1. **The Researcher (Agent Node):** Receives the biological query, plans the research strategy, and autonomously triggers parallel tool calls.
2. **The Tools (API Connectors):**
   - `pubmed_search`: Dynamically fetches recent medical literature and PMIDs using BioPython.
   - `chembl_search`: Retrieves molecular bioactivity data (IC50, binding affinity) via the ChEMBL API.
4. **The Summarizer (Formatting Node):** Synthesizes the raw API outputs into a highly structured, professional Markdown clinical report.
5. **The Evaluator (LLM-as-a-Judge):** An independent scoring script utilizing **Pydantic** structured outputs. It compares the final report against the raw API data, penalizing the system if any PMIDs or data points are hallucinated.

---

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain (`langchain-core`, `langchain-groq`)
* **LLM:** Groq (Llama-3.3-70b-versatile) for ultra-fast, reliable JSON tool execution
* **Data Validation:** Pydantic (Strict schema enforcement)
* **External APIs:** NCBI PubMed, ChEMBL (via `biopython` and `chembl_webresource_client`)
* **Environment:** Python 3.11+

---

## 📊 Evaluation

After the initial implementation, I built a deterministic evaluation framework on the `eval-framework` branch to measure pipeline quality without relying on the LLM-as-a-Judge alone. The goal: separate verifiable structural correctness (cited PMIDs and numeric values appear in actual tool output) from harder-to-measure judge opinion.

### Methodology

The framework runs against each completed pipeline run and computes two deterministic checks plus the existing LLM-as-a-Judge:

1. **PMID grounding rate** — every PMID cited in the final report must appear in the run's actual `raw_tool_data`. A cited PMID that does not appear in any tool response is a fabricated citation. This directly tests "did the LLM cite something its tools returned?"

2. **Numeric grounding rate** — every numeric+unit claim in the final report (IC50 values, binding affinities like "41 nM", "9.7 µM") is extracted via regex and matched against numeric values in the raw tool data, with relative tolerance. Catches fabricated quantitative claims.

3. **LLM-as-a-Judge** — the existing Pydantic-structured `AgentScorecard` continues to score hallucination and precision, run alongside the deterministic checks for calibration analysis.

The benchmark suite (`eval/benchmarks/queries.jsonl`) contains 25 curated queries across mechanism-of-action, binding-affinity, target-validation, and adversarial categories. Adversarial queries (made-up entities like XYZQRT-9999) test whether the system fabricates citations when the input has no real grounding.

### Design pivot worth noting

An earlier version of the framework compared cited PMIDs against a static `expected_pmids` set populated by direct `Entrez.esearch` on cleaned queries. That approach failed: the agent's LLM dynamically reformulates queries before calling tools, so the static ground-truth set diverged systematically from what the pipeline actually retrieved. On q001 (imatinib mechanism), grounding rate was 0.00 despite cited PMIDs being real PubMed results.

The pivot — comparing cited PMIDs against the run's actual `raw_tool_data` — gave a direct hallucination signal without external-oracle dependencies. After the pivot, q001 grounding rate moved from 0.00 to 1.00 with no pipeline change.

### Results across both runs (17 of 25 queries succeeded)

Run artifacts: `eval/runs/20260504_181524/results.json` (7 queries) and `eval/runs/20260505_181731/results.json` (10 queries).

| Metric | Value | Notes |
|---|---|---|
| Queries succeeded | 17 / 25 | 6 blocked by Groq free-tier daily token ceiling, 2 by Llama 3.3 tool-call format errors |
| PMID grounding rate (avg) | **1.00** | No PMID fabrication on any query that cited PMIDs |
| Numeric grounding rate (avg) | 0.64 | Caught hallucinated binding-affinity values on multiple queries |
| Judge-vs-deterministic agreement | Surfaced on q014 (nilotinib) and q018 (dasatinib): both methods independently flagged hallucinated IC50 values | |
| Avg latency (per query) | 27 s | Multi-step LangGraph DAG with up to 4 tool-calling rounds |

### Notable findings

- **PMID grounding stayed at 1.00 across all runs.** Hard tool-grounding (the agent loop is structured so the LLM cannot answer without invoking PubMed or ChEMBL) prevents PMID fabrication at the structural level.
- **Numeric grounding caught real hallucinations.** On q014 (nilotinib IC50) and q018 (dasatinib SRC kinase binding affinity), the synthesizer cited specific numeric values that did not appear anywhere in the raw ChEMBL tool output. The LLM-as-a-Judge independently scored these reports as hallucinated, giving us a calibration data point: judge and deterministic checks agreed on the failures.
- **Llama 3.3 occasionally produces invalid tool-call XML.** Two queries (q006, q021) failed with `tool_use_failed` errors where the model emitted malformed `<function=...>` syntax. Per-query failure isolation in the runner kept these from cascading.

### How to run

```bash
python -m eval.run_suite                        # full 25-query benchmark
python -m eval.run_suite --limit 3              # smoke test on first 3
python -m eval.run_suite --ids q001,q010,q024   # specific queries
```

Each run writes a timestamped artifact to `eval/runs/{timestamp}/results.json` containing per-query scores, full reports, judge scorecards, deterministic grounding details, and aggregate metrics.

### What I'd extend next

- **Run-to-run consistency check.** Repeat the same query 3–5 times and measure PMID set overlap (Jaccard) and numeric stability across runs. Surfaces non-determinism in tool-selection and answer formulation.
- **Configurable rubric.** Refactor `AgentScorecard` from hardcoded Pydantic to a YAML-driven schema via `pydantic.create_model()` so new evaluation dimensions are added via configuration.
- **Judge calibration study at scale.** With paid Groq tier (free-tier daily ceiling blocked our full benchmark), run all 25 queries × 3 reps to compute proper confusion matrix between judge scores and deterministic checks.
