from typing import TypedDict, List, Annotated, Union
import operator
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from tools import pubmed_search, chembl_search

load_dotenv()

'''from langchain_community.chat_models import ChatOllama
# Swapping out Groq for local MedGemma
llm = ChatOllama(model="medgemma", temperature=0)'''

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
tools = [pubmed_search, chembl_search]
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List[str], operator.add]
    #raw_data: str
    verdict: str

def researcher_node(state: AgentState):
    print("--- RESEARCHER IS THINKING ---")
    response = llm_with_tools.invoke(state["messages"])
    # We return the AI's response (which might contain tool calls)
    return {"messages": [response]}
    
    # Use the last message as the query
    result = executor.invoke({"input": state["messages"][-1]})
    return {"messages": [result["output"]], "raw_data": result["output"]}

def tool_node(state: AgentState):
    print("--- EXECUTING TOOLS ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    results = []
    # Loop through the tool calls Gemini requested
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        
        # Connect to your actual functions in tools.py
        if tool_name == "pubmed_search":
            content = pubmed_search.invoke(args)
        elif tool_name == "chembl_search":
            content = chembl_search.invoke(args)
        else:
            content = f"Tool {tool_name} not found."
            
        # Wrap the result in a ToolMessage so LangGraph understands it
        results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(content)))
    
    return {"messages": results}

# --- CRITIC NODE (The New Part) ---
def critic_node(state: AgentState):
    print("--- EXECUTING TOOLS ---")
    messages = state["messages"]
    last_message = messages[-1]
    
    results = []
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        args = tool_call["args"]
        
        # Mapping string names to actual functions
        if tool_name == "pubmed_search":
            content = pubmed_search.invoke(args)
        elif tool_name == "chembl_search":
            content = chembl_search.invoke(args)
            
        results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(content)))
    
    return {"messages": results}

def summarizer_node(state: AgentState):
    print("--- GENERATING FINAL REPORT ---")
    
    # 1. Grab the Researcher's final summary
    researcher_summary = state["messages"][-1].content
    
    # 2. Extract the raw data from the Tool Messages in the state history
    raw_tool_data = "\n".join(
        [msg.content for msg in state["messages"] if msg.type == "tool"]
    )
    
    # 3. Combine them so the Summarizer has the full picture
    full_context = f"RESEARCHER SUMMARY:\n{researcher_summary}\n\nRAW TOOL DATA:\n{raw_tool_data}"
    
    report_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Lead Bioinformatician. 
        Take the provided research data and format it into a professional, highly-structured Markdown report.
        
        🚨 CRITICAL CONSTRAINTS 🚨
        1. DO NOT invent, guess, or hallucinate any PMIDs, IC50 values, or scientific claims.
        2. You must ONLY cite PMIDs that are explicitly written in the raw data provided.
        3. If specific numerical data or PMIDs are missing, explicitly write "Data not available in current literature pull."
        
        Include these sections:
        # 🧬 BioReason Target Validation Report
        ## 🎯 Target Overview
        ## 📚 Literature Evidence (Cite PMIDs)
        ## 🧪 Molecular Bioactivity
        ## ⚖️ Final Verdict (Is it a good target?)
        """),
        ("human", "{data}") # Pass the combined context here
    ])
    
    chain = report_prompt | llm
    final_report = chain.invoke({"data": full_context})
    
    return {"messages": [final_report]}
    
    chain = report_prompt | llm
    final_report = chain.invoke({"data": verified_data})
    
    return {"messages": [final_report]}
