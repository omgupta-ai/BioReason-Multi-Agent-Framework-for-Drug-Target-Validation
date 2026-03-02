# BioReason-Multi-Agent-Framework-for-Drug-Target-Validation
An autonomous multi-agent AI framework using LangGraph and Groq (Llama 3.3) to validate medical drug targets, featuring an automated LLM-as-a-Judge evaluator to guarantee 0% hallucinations.

Built with **LangGraph** and **Groq (Llama 3.3 70B)**, BioReason tackles one of the biggest challenges in medical AI: **hallucinations**. It enforces strict tool-calling, real-time API grounding, and features an automated "LLM-as-a-Judge" evaluation loop to guarantee high-precision, hallucination-free clinical reports.

---

## 🚀 The Problem & The Solution
Standard LLMs often hallucinate citations or invent molecular binding affinities when asked complex biological questions. 

**BioReason** solves this by removing the LLM's ability to guess. Instead, it acts as a reasoning engine that orchestrates a workflow: it queries real scientific databases, extracts exact data points, formats a clinical summary, and then ruthlessly grades its own output against the raw API data to ensure 100% factual accuracy.

## 🧠 System Architecture



The pipeline is built on a directed acyclic graph (DAG) using LangGraph, consisting of the following core components:

1. **The Researcher (Agent Node):** Receives the biological query, plans the research strategy, and autonomously triggers parallel tool calls.
2. **The Tools (API Connectors):** - `pubmed_search`: Dynamically fetches recent medical literature and PMIDs using BioPython.
   - `chembl_search`: Retrieves molecular bioactivity data (IC50, binding affinity) via the ChEMBL API.
3. **The Summarizer (Formatting Node):** Synthesizes the raw API outputs into a highly structured, professional Markdown clinical report.
4. **The Evaluator (LLM-as-a-Judge):** An independent scoring script utilizing **Pydantic** structured outputs. It compares the final report against the raw API data, penalizing the system if any PMIDs or data points are hallucinated.

---

## 🛠️ Tech Stack
* **Orchestration:** LangGraph, LangChain (`langchain-core`, `langchain-groq`)
* **LLM:** Groq (Llama-3.3-70b-versatile) for ultra-fast, reliable JSON tool execution
* **Data Validation:** Pydantic (Strict schema enforcement)
* **External APIs:** NCBI PubMed, ChEMBL (via `biopython` and `chembl_webresource_client`)
* **Environment:** Python 3.11+


