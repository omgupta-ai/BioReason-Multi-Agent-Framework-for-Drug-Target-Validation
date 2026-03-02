# evaluator.py
import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# 1. DEFINE THE GRADING RUBRIC (The JSON Schema)
class AgentScorecard(BaseModel):
    hallucination_score: int = Field(
        description="Score from 0 to 10. 10 means NO hallucinations (perfectly grounded). 0 means completely made up PMIDs or facts."
    )
    precision_score: int = Field(
        description="Score from 0 to 10. 10 means exact numbers (e.g., -6.7 kcal/mol) and PMIDs were cited perfectly."
    )
    passed: bool = Field(
        description="True if both scores are 8 or higher, False otherwise."
    )
    evaluator_reasoning: str = Field(
        description="A short, harsh explanation of why these scores were given."
    )

# 2. SETUP THE JUDGE
# We use Llama 3.3 again, but we force it to output our Pydantic schema!
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
structured_judge = llm.with_structured_output(AgentScorecard)

# 3. THE EVALUATION PROMPT
eval_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a ruthless Scientific Peer Reviewer. 
    Compare the generated FINAL REPORT against the RAW TOOL DATA.
    
    RULES:
    - If a PMID is in the report but NOT in the raw data, tank the hallucination score.
    - If the report correctly quotes molecular affinities (like -6.7 kcal/mol) from the raw data, give a high precision score.
    - Be strict. Do not forgive made-up data.
    
    🚨 CRITICAL JSON FORMATTING RULES 🚨
    - hallucination_score MUST be a raw integer (e.g., 10), NEVER a string (e.g., "10").
    - precision_score MUST be a raw integer (e.g., 9), NEVER a string (e.g., "9").
    - passed MUST be a raw boolean (true or false), NEVER a string ("true" or "false")."""),
    ("human", "RAW TOOL DATA:\n{raw_data}\n\nFINAL REPORT:\n{report}")
])

# 4. THE EVALUATOR FUNCTION
def grade_report(raw_data: str, final_report: str):
    print("--- ⚖️ JUDGE IS EVALUATING ---")
    chain = eval_prompt | structured_judge
    scorecard = chain.invoke({
        "raw_data": raw_data, 
        "report": final_report
    })
    return scorecard

