import os
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from state import AgentState

# ld env
load_dotenv()

# init llm + srch clnt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def planner_node(st: AgentState) -> AgentState:

    print(" PLANNER: brk qry 2 srch tasks...")

    p_prompt = f"""
    You are a research planner. Given the user query below, break it down into 
    exactly 3 distinct, specific Google search queries that will help gather 
    comprehensive information to answer the original query.
    
    User Query: {st['query']}
    
    Return only the 3 search queries, one per line, without numbering or bullets.
    Make them specific and targeted for effective web search.
    """

    try:
        res = llm.invoke(p_prompt)
        plan = [q.strip() for q in res.content.strip().split('\n') if q.strip()]

        # mk sure exactly 3
        while len(plan) < 3:
            plan.append(f"add srch for {st['query']}")
        plan = plan[:3]  # cap 3

        print(f" plan gen: {plan}")

        return {
            **st,
            "plan": plan,
            "revision_number": 0,  # init rev cnt
            "max_revisions": 2,    # set max rev
            "content": []          # init content lst
        }

    except Exception as e:
        print(f" planner err: {e}")
        # fbk plan
        fbk_plan = [
            f"What is {st['query']}",
            f"How does {st['query']} work",
            f"Examples of {st['query']}"
        ]
        return {
            **st,
            "plan": fbk_plan,
            "revision_number": 0,
            "max_revisions": 2,
            "content": []
        }


def researcher_node(st: AgentState) -> AgentState:

    print(" RESEARCHER: run srch qrys...")

    content = st.get("content", [])

    for i, q in enumerate(st["plan"]):
        print(f" srch: {q}")

        try:
            # run srch via tavily
            res = tavily.search(query=q, search_depth="basic")

            # fmt res
            if res.get("results"):
                s_cont = f"\n=== SEARCH QUERY {i+1}: {q} ===\n"
                for r in res["results"][:3]:  # top 3
                    s_cont += f"\nTitle: {r.get('title', 'N/A')}\n"
                    s_cont += f"URL: {r.get('url', 'N/A')}\n"
                    s_cont += f"Content: {r.get('content', 'N/A')[:500]}...\n"

                content.append(s_cont)
                print(f" fnd {len(res['results'])} res")
            else:
                content.append(f"\n=== SEARCH QUERY {i+1}: {q} ===\nNo results found.")
                print(f" no res for: {q}")

        except Exception as e:
            print(f" srch err for '{q}': {e}")
            content.append(f"\n=== SEARCH QUERY {i+1}: {q} ===\nSearch failed: {str(e)}")

    print(f"📚 tot content pcs: {len(content)}")

    return {
        **st,
        "content": content
    }


def critic_node(st: AgentState) -> AgentState:

    print(" CRITIC: chk res qlty...")

    # comb content 4 eval
    comb = "\n".join(st["content"])

    c_prompt = f"""
    You are a research quality controller. Review the accumulated research data below
    and determine if it fully and comprehensively answers the user's original query.
    
    Original Query: {st['query']}
    
    Research Data:
    {comb}
    
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
        res = llm.invoke(c_prompt)
        dec = res.content.strip()

        print(f"🤖 critic dec: {dec}")

        if dec.upper() == "PROCEED":
            print(" appr - go writer")
            return st  # no chg, go writer
        else:
            # new qry + inc rev
            print(f" not enuf - new qry: {dec}")
            return {
                **st,
                "plan": [dec],  # new qry
                "revision_number": st["revision_number"] + 1
            }

    except Exception as e:
        print(f" critic err: {e}")
        print(" def 2 proceed on err")
        return st  # def proceed on err


def writer_node(st: AgentState) -> AgentState:

    print(" WRITER: mk fin report...")

    # comb content 4 report
    comb = "\n".join(st["content"])

    w_prompt = f"""
    You are an expert research analyst. Based on the accumulated research data below,
    write a comprehensive, professional report that fully answers the user's original query.
    
    Original Query: {st['query']}
    
    Research Data:
    {comb}
    
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
        res = llm.invoke(w_prompt)
        fin_rep = res.content

        print("📄 fin report gen ok")

        return {
            **st,
            "final_report": fin_rep
        }

    except Exception as e:
        print(f" writer err: {e}")
        # fbk report
        fbk_rep = f"""
        # Research Report: {st['query']}
        
        ## Error During Report Generation
        
        Unfortunately, an error occurred while generating the final report:
        {str(e)}
        
        ## Raw Research Data
        
        {comb}
        """

        return {
            **st,
            "final_report": fbk_rep
        }
