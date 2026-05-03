"""
Deterministic grounding checks for BioReason reports.

Two checks:
  1. pmid_grounding_rate: of the PMIDs cited in the report, what fraction
     appear in the expected_pmids ground-truth set for that query?
  2. numeric_grounding_rate: of the numeric+unit claims (e.g., '41 nM',
     '7820.0 nM') cited in the report, what fraction appear in the raw
     tool output?

These run alongside the LLM-as-Judge to validate its scoring.
"""
import re

PMID_PATTERN = re.compile(r"\bPMID[:\s-]*(\d{6,9})\b", re.IGNORECASE)

NUMERIC_PATTERN = re.compile(
    r"(-?\d+\.?\d*)\s*(kcal/mol|nM|μM|uM|µM|pM|mM|%)",
    re.IGNORECASE,
)

CHEMBL_VALUE_UNIT_PATTERN = re.compile(
    r"['\"]?standard_value['\"]?\s*[:=]\s*['\"]?(-?\d+\.?\d*)['\"]?"
    r".*?"
    r"['\"]?standard_units['\"]?\s*[:=]\s*['\"]?(kcal/mol|nM|μM|uM|µM|pM|mM|%)['\"]?",
    re.IGNORECASE | re.DOTALL,
)


def extract_pmids(text: str) -> set[str]:
    """Pull all PMIDs cited in text, deduplicated."""
    return set(PMID_PATTERN.findall(text or ""))


def extract_numeric_claims(text: str) -> list[tuple[float, str]]:
    """Pull all (value, unit) numeric claims from text.

    Handles two formats:
      1. Adjacent: '41.0 nM', '300 nM' (typical in LLM-generated reports)
      2. ChEMBL field format: 'standard_value=41.0 ... standard_units=nM'
         (typical in raw tool output)
    """
    claims = []

    # Format 1: adjacent value + unit
    for value, unit in NUMERIC_PATTERN.findall(text or ""):
        try:
            normalized_unit = unit.lower().replace("μ", "u").replace("µ", "u")
            claims.append((float(value), normalized_unit))
        except ValueError:
            continue

    # Format 2: ChEMBL field-separated
    for value, unit in CHEMBL_VALUE_UNIT_PATTERN.findall(text or ""):
        try:
            normalized_unit = unit.lower().replace("μ", "u").replace("µ", "u")
            claims.append((float(value), normalized_unit))
        except ValueError:
            continue

    return claims


def pmid_grounding_rate(report: str, raw_tool_data: str) -> dict:
    """How many PMIDs cited in the report appear in the raw tool data.

    A PMID in the report that does not appear anywhere in raw_tool_data is
    a fabricated citation: the LLM is asserting a paper exists that its
    tools never returned.
    """
    cited = extract_pmids(report)
    available = extract_pmids(raw_tool_data)

    if not cited:
        return {
            "cited": 0,
            "grounded": 0,
            "rate": 1.0,
            "ungrounded": [],
        }

    ungrounded = sorted(cited - available)
    grounded = len(cited) - len(ungrounded)
    return {
        "cited": len(cited),
        "grounded": grounded,
        "rate": grounded / len(cited),
        "ungrounded": ungrounded,
    }


def numeric_grounding_rate(
    report: str,
    raw_tool_data: str,
    rel_tolerance: float = 0.01,
) -> dict:
    """How many cited numeric+unit claims appear in raw_tool_data."""
    cited = extract_numeric_claims(report)
    available = extract_numeric_claims(raw_tool_data)

    if not cited:
        return {
            "cited": 0,
            "grounded": 0,
            "rate": 1.0,
            "ungrounded": [],
        }

    available_by_unit: dict[str, list[float]] = {}
    for value, unit in available:
        available_by_unit.setdefault(unit, []).append(value)

    ungrounded: list[str] = []
    grounded = 0
    for value, unit in cited:
        candidates = available_by_unit.get(unit, [])
        tol = max(rel_tolerance * abs(value), rel_tolerance)
        if any(abs(c - value) <= tol for c in candidates):
            grounded += 1
        else:
            ungrounded.append(f"{value} {unit}")

    return {
        "cited": len(cited),
        "grounded": grounded,
        "rate": grounded / len(cited),
        "ungrounded": ungrounded,
    }