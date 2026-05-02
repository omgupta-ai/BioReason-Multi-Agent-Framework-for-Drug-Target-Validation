"""
For each benchmark query, call Entrez.esearch directly to get the PMID list
that PubMed would return for that query. Store this as ground truth in
expected_pmids. Used by eval/grounding.py to verify whether reports cite
PMIDs that actually exist in the search results.

Usage:
    python -m eval.populate_expected
"""
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from Bio import Entrez

load_dotenv()

# Use same email as tools.py (NCBI requires a real contact)
Entrez.email = "gupta.om@northeastern.edu"

# Path setup so we can run as a module from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QUERIES_PATH = Path(__file__).parent / "benchmarks" / "queries.jsonl"

import re

# Common English stopwords + biomedical query meta-phrasing
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "as",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "do", "does", "did", "doing",
    "this", "that", "these", "those",
    "and", "or", "but", "if", "then",
    "have", "has", "had", "having",
    "can", "could", "may", "might", "should", "would", "will",
    "i", "you", "he", "she", "it", "we", "they",
    "valid", "validated", "good", "values", "against",  # query-specific filler
}

ADVERSARIAL_IDS = {"q024", "q025"}

def clean_query_for_pubmed(query: str) -> str:
    """Strip stopwords and conversational phrasing for PubMed keyword search.

    PubMed's esearch is a keyword index, not a semantic search. Long natural-
    language queries like 'What is the mechanism of action of imatinib...'
    return zero results because the verb/preposition tokens dilute the
    keyword signal. We strip stopwords and punctuation to leave only the
    biomedical content terms.
    """
    # Strip apostrophes (Alzheimer's -> Alzheimers)
    cleaned = query.replace("'", "").replace("'", "")
    # Lowercase for stopword matching, then split on word boundaries
    tokens = re.findall(r"[A-Za-z0-9\-]+", cleaned)
    # Keep only non-stopword tokens, preserving original case from query where possible
    kept = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join(kept)


def fetch_pmids(query: str, retmax: int = 10) -> list[str]:
    """Call Entrez.esearch and return PMIDs as strings."""
    cleaned = clean_query_for_pubmed(query)
    handle = Entrez.esearch(db="pubmed", term=cleaned, retmax=retmax)
    record = Entrez.read(handle)
    handle.close()
    return list(record.get("IdList", []))


def main():
    queries = []
    with open(QUERIES_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    print(f"Loaded {len(queries)} queries.\n")

    for q in queries:
        if q.get("expected_pmids"):
            print(f"[skip] {q['id']} already populated")
            continue

        if q["id"] in ADVERSARIAL_IDS:
            q["expected_pmids"] = []
            print(f"[adv]  {q['id']}: marked as adversarial, expected_pmids=[]")
            continue

        print(f"[run]  {q['id']}: {q['query'][:70]}...")
        try:
            pmids = fetch_pmids(q["query"], retmax=10)
            q["expected_pmids"] = pmids
            preview = pmids[:5]
            suffix = "..." if len(pmids) > 5 else ""
            print(f"       -> {len(pmids)} PMIDs: {preview}{suffix}")
        except Exception as e:
            print(f"       -> FAILED: {e}")
            q["expected_pmids"] = []
        time.sleep(0.4)

    with open(QUERIES_PATH, "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print(f"\nDone. Wrote {len(queries)} queries to {QUERIES_PATH}")


if __name__ == "__main__":
    main()