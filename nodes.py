
import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from state import AgentState

# Load environment variables
load_dotenv()

# Initialize LLM and Search Client
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def planner_node(state: AgentState) -> AgentState:
  
    print("🎯 PLANNER: Breaking down query into search tasks...")
    
    planner_prompt = f"""
    You are a research planner. Given the user query below, break it down into 
    exactly 3 distinct, specific Google search queries that will help gather 
    comprehensive information to answer the original query.
    
    User Query: {state['query']}
    
    Return only the 3 search queries, one per line, without numbering or bullets.
    Make them specific and targeted for effective web search.
    """
    
    try:
        response = llm.invoke(planner_prompt)
        plan = [query.strip() for query in response.content.strip().split('\n') if query.strip()]
        
        # Ensure we have exactly 3 queries
        while len(plan) < 3:
            plan.append(f"additional search for {state['query']}")
        plan = plan[:3]  # Limit to exactly 3
        
        print(f"📋 Generated search plan: {plan}")
        
        return {
            **state,
            "plan": plan,
            "revision_number": 0,  # Initialize revision counter
            "max_revisions": 2,    # Set max revisions limit
            "content": []          # Initialize content list
        }
        
    except Exception as e:
        print(f"❌ Planner error: {e}")
        # Fallback plan
        fallback_plan = [
            f"What is {state['query']}",
            f"How does {state['query']} work",
            f"Examples of {state['query']}"
        ]
        return {
            **state,
            "plan": fallback_plan,
            "revision_number": 0,
            "max_revisions": 2,
            "content": []
        }


def researcher_node(state: AgentState) -> AgentState:
   
    print("🔍 RESEARCHER: Executing search queries...")
    
    content = state.get("content", [])
    
    for i, query in enumerate(state["plan"]):
        print(f"🌐 Searching: {query}")
        
        try:
            # Execute search using Tavily
            search_result = tavily.search(query=query, search_depth="basic")
            
            # Extract and format search results
            if search_result.get("results"):
                search_content = f"\n=== SEARCH QUERY {i+1}: {query} ===\n"
                for result in search_result["results"][:3]:  # Top 3 results
                    search_content += f"\nTitle: {result.get('title', 'N/A')}\n"
                    search_content += f"URL: {result.get('url', 'N/A')}\n"
                    search_content += f"Content: {result.get('content', 'N/A')[:500]}...\n"
                
                content.append(search_content)
                print(f"✅ Found {len(search_result['results'])} results")
            else:
                content.append(f"\n=== SEARCH QUERY {i+1}: {query} ===\nNo results found.")
                print(f"⚠️ No results found for: {query}")
                
        except Exception as e:
            print(f"❌ Search error for '{query}': {e}")
            content.append(f"\n=== SEARCH QUERY {i+1}: {query} ===\nSearch failed: {str(e)}")
    
    print(f"📚 Total content pieces: {len(content)}")
    
    return {
        **state,
        "content": content
    }


def critic_node(state: AgentState) -> AgentState:
   
    print("🔎 CRITIC: Evaluating research quality...")
    
    # Combine all content for evaluation
    combined_content = "\n".join(state["content"])
    
    critic_prompt = f"""
    You are a research quality controller. Review the accumulated research data below
    and determine if it fully and comprehensively answers the user's original query.
    
    Original Query: {state['query']}
    
    Research Data:
    {combined_content}
    
    CRITICAL EVALUATION:
    - Does this data completely answer the original query?
    - Is the information comprehensive and sufficient?
    - Are there critical gaps or missing information?
    
    RESPOND IN ONE OF TWO WAYS:
    1. If the research is complete and sufficient, respond with exactly: "PROCEED"
    2. If more information is needed, respond with a single, specific new search query
       that will fill the most important gap in the research.
    
    Be strict in your evaluation. Only PROCEED if you're confident the research is comprehensive.
    """
    
    try:
        response = llm.invoke(critic_prompt)
        decision = response.content.strip()
        
        print(f"🤖 Critic decision: {decision}")
        
        if decision.upper() == "PROCEED":
            print("✅ Research approved - proceeding to write final report")
            return state  # No changes, proceed to writer
        else:
            # Generate new search query and increment revision
            print(f"🔄 Research insufficient - new query: {decision}")
            return {
                **state,
                "plan": [decision],  # New search query
                "revision_number": state["revision_number"] + 1
            }
            
    except Exception as e:
        print(f"❌ Critic error: {e}")
        print("⚠️ Defaulting to proceed due to critic error")
        return state  # Default to proceeding on error


def writer_node(state: AgentState) -> AgentState:
   
    print("✍️ WRITER: Synthesizing final report...")
    
    # Combine all content for report generation
    combined_content = "\n".join(state["content"])
    
    writer_prompt = f"""
    You are an expert research analyst. Based on the accumulated research data below,
    write a comprehensive, professional report that fully answers the user's original query.
    
    Original Query: {state['query']}
    
    Research Data:
    {combined_content}
    
    REQUIREMENTS:
    1. Write a well-structured, professional report
    2. Include proper citations and references to sources
    3. Provide clear, actionable insights
    4. Use markdown formatting for readability
    5. Include a summary and key findings
    6. Acknowledge any limitations or gaps in the research
    7. Structure with headings, bullet points, and clear organization
    
    The report should be comprehensive yet concise, focusing on the most relevant information.
    """
    
    try:
        response = llm.invoke(writer_prompt)
        final_report = response.content
        
        print("📄 Final report generated successfully")
        
        return {
            **state,
            "final_report": final_report
        }
        
    except Exception as e:
        print(f"❌ Writer error: {e}")
        # Fallback report
        fallback_report = f"""
        # Research Report: {state['query']}
        
        ## Error During Report Generation
        
        Unfortunately, an error occurred while generating the final report:
        {str(e)}
        
        ## Raw Research Data
        
        {combined_content}
        """
        
        return {
            **state,
            "final_report": fallback_report
        }
