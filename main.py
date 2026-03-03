# main.py
from langgraph.graph import StateGraph, END
from nodes import AgentState, researcher_node, tool_node, critic_node, summarizer_node
from evaluator import grade_report

workflow = StateGraph(AgentState)

# 1. Add Nodes
workflow.add_node("researcher", researcher_node)
workflow.add_node("tools", tool_node)
workflow.add_node("critic", critic_node)
workflow.add_node("summarizer", summarizer_node)

# 2. Define the Routing Logic
def route_researcher(state: AgentState):
    last_message = state["messages"][-1]
    # If the LLM wants to use a tool, go to 'tools' node
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, go to the Critic
    return "critic"

def route_critic(state: AgentState):
    # If we haven't implemented the strict FAIL logic yet, default to passing
    # In a full build, this checks if state["verdict"] == "FAIL"
    return "summarizer"

# 3. Connect the Edges
workflow.set_entry_point("researcher")

workflow.add_conditional_edges(
    "researcher",
    route_researcher,
    {"tools": "tools", "critic": "critic"}
)

# After tools are run, always go back to researcher to summarize
workflow.add_edge("tools", "researcher") 

workflow.add_conditional_edges(
    "critic",
    route_critic,
    {"summarizer": "summarizer", "researcher": "researcher"}
) 

workflow.add_edge("summarizer", END)

# 4. Compile and Run
app = workflow.compile()

if __name__ == "__main__":
    from langchain_core.messages import HumanMessage

    print("\n" + "="*50)
    print("🧬 Welcome to BioReason: Autonomous Target Validator")
    print("="*50)
    
    # Ask the user for input
    print("\nWhat drug target or disease would you like to research?")
    print("(Example: 'Is MAPK1 a target for Alzheimer's?')")
    user_query = input("\n> ")
    
    # Handle empty inputs safely
    if not user_query.strip():
        print("No query provided. Exiting BioReason...")
        sys.exit(0)
        
    inputs = {"messages": [HumanMessage(content=user_query)]}
    
    print("\n🚀 STARTING BIOREASON PIPELINE...\n")
    final_state = app.invoke(inputs)
    
    # Extract the Final Report
    final_report = final_state["messages"][-1].content
    
    # Extract the Raw Tool Data that the agent found
    raw_tool_data = "\n".join(
        [msg.content for msg in final_state["messages"] if msg.type == "tool"]
    )
    
    # Print the Report
    print("\n" + "="*50)
    print(final_report)
    print("="*50 + "\n")
    
    # Grade the Report!
    print("⚖️ GRADING THE REPORT...\n")
    scorecard = grade_report(raw_tool_data, final_report)
    
    print("--- 📊 BIOREASON FINAL SCORECARD ---")
    print(f"Hallucination Score: {scorecard.hallucination_score}/10")
    print(f"Precision Score:     {scorecard.precision_score}/10")
    print(f"Verdict PASSED:      {scorecard.passed}")
    print(f"Reasoning:           {scorecard.evaluator_reasoning}")
