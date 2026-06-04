
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import planner_node, researcher_node, critic_node, writer_node


def should_continue_research(state: AgentState) -> str:
   
    print(f"🔄 EVALUATING: Revision {state['revision_number']}/{state['max_revisions']}")
    
    # Check if we've exceeded max revisions
    if state["revision_number"] >= state["max_revisions"]:
        print("⏹️ Max revisions reached - proceeding to write report")
        return "writer"
    
    # Check if the plan was modified (indicating critic rejection)
    # If plan has only 1 item, it's likely a new query from critic
    if len(state["plan"]) == 1 and state["revision_number"] > 0:
        print("🔄 Critic requested more research - continuing loop")
        return "researcher"
    
    # If we have original plan (3 items) and no revisions, proceed
    if state["revision_number"] == 0:
        print(" Initial research complete - proceeding to write report")
        return "writer"
    
    # Default to writer
    print(" Research complete - proceeding to write report")
    return "writer"


def create_research_graph() -> StateGraph:

    print("🚀 Initializing DeepReason Research Graph...")
    
    # Initialize the StateGraph with our AgentState
    workflow = StateGraph(AgentState)
    
    # Add all nodes to the workflow
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("writer", writer_node)
    
    # Define the entry point
    workflow.set_entry_point("planner")
    
    # Add edges for the main workflow
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    
    # Add conditional edge from critic - this is the core loop logic
    workflow.add_conditional_edges(
        "critic",
        should_continue_research,
        {
            "researcher": "researcher",  # Loop back for more research
            "writer": "writer"          # Proceed to final report
        }
    )
    
    # Final edge to END
    workflow.add_edge("writer", END)
    
    # Compile the graph
    app = workflow.compile()
    
    print("Research Graph compiled successfully")
    print("   Graph Structure:")
    print("   planner → researcher → critic ──┐")
    print("                ↑                    │")
    print("                └─────(conditional)───┘")
    print("                                   ↓")
    print("                                writer → END")
    
    return app


def run_research(query: str) -> Dict[str, Any]:
    """
    Executes the research workflow for a given query.
    
    Args:
        query: User research query
        
    Returns:
        Final state containing the research report
    """
    print(f"\n🎯 Starting DeepReason Research for: '{query}'")
    print("=" * 60)
    
    # Initialize the research graph
    app = create_research_graph()
    
    # Create initial state
    initial_state = {
        "query": query,
        "plan": [],
        "content": [],
        "revision_number": 0,
        "max_revisions": 2,
        "final_report": ""
    }
    
    try:
        # Execute the workflow
        print("\n Executing research workflow...")
        final_state = app.invoke(initial_state)
        
        print("\n" + "=" * 60)
        print("RESEARCH COMPLETE!")
        print("=" * 60)
        
        return final_state
        
    except Exception as e:
        print(f"\n❌ Research workflow failed: {e}")
        return {
            **initial_state,
            "final_report": f"Research failed due to error: {str(e)}"
        }


def main():
  
    print(" DeepReason AI Research Agent")
    print("=" * 40)
    print("Autonomous Research Engine powered by LangGraph")
    print("Enter your research query or 'quit' to exit\n")
    
    while True:
        try:
            # Get user input
            query = input("🔍 Research Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print(" Goodbye!")
                break
            
            if not query:
                print(" Please enter a valid query")
                continue
            
            # Run research
            result = run_research(query)
            
            # Display results
            print(f"\n📄 FINAL REPORT:")
            print("-" * 40)
            print(result.get("final_report", "No report generated"))
            print("-" * 40)
            print(f"\n📊 Research Stats:")
            print(f"   • Total revisions: {result.get('revision_number', 0)}")
            print(f"   • Content pieces: {len(result.get('content', []))}")
            print(f"   • Max revisions: {result.get('max_revisions', 2)}")
            
        except KeyboardInterrupt:
            print("\nResearch interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            print("Please try again or check your configuration.")


if __name__ == "__main__":
    main()
