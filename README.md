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


