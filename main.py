from typing import Dict, Any
from langgraph.graph import StateGraph, END
from state import AgentState
from nodes import planner_node, researcher_node, critic_node, writer_node


def chk_cont(st: AgentState) -> str:

    print(f" chk rev {st['revision_number']}/{st['max_revisions']}")

    # max rev hit -> go writer
    if st["revision_number"] >= st["max_revisions"]:
        print(" max rev done -> writer")
        return "writer"

    # plan len 1 + rev>0 -> critic sent bk
    if len(st["plan"]) == 1 and st["revision_number"] > 0:
        print(" critic sent bk -> researcher")
        return "researcher"

    # rev 0 -> 1st run done
    if st["revision_number"] == 0:
        print(" 1st run done -> writer")
        return "writer"

    # def case
    print(" done -> writer")
    return "writer"


def mk_graph() -> StateGraph:

    print(" init graph...")

    g = StateGraph(AgentState)

    # ad nd
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("critic", critic_node)
    g.add_node("writer", writer_node)

    # entry pt
    g.set_entry_point("planner")

    # ad ed 2 nd
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "critic")

    # cond ed frm critic - loop logic
    g.add_conditional_edges(
        "critic",
        chk_cont,
        {
            "researcher": "researcher",  # loop bk
            "writer": "writer"          # go end
        }
    )

    # fin ed
    g.add_edge("writer", END)

    # compile
    app = g.compile()

    print("graph rdy")
    print("   struct:")
    print("   planner → researcher → critic ──┐")
    print("                ↑                    │")
    print("                └─────(cond)───┘")
    print("                                   ↓")
    print("                                writer → END")

    return app


def run_res(q: str) -> Dict[str, Any]:

    print(f"\n start res for: '{q}'")
    print("=" * 60)

    app = mk_graph()

    # init st
    init_st = {
        "query": q,
        "plan": [],
        "content": [],
        "revision_number": 0,
        "max_revisions": 2,
        "final_report": ""
    }

    try:
        # run flow
        print("\n running...")
        fin_st = app.invoke(init_st)

        print("\n" + "=" * 60)
        print("DONE!")
        print("=" * 60)

        return fin_st

    except Exception as e:
        print(f"\n failed: {e}")
        return {
            **init_st,
            "final_report": f"failed due 2 err: {str(e)}"
        }


def main():

    print(" DeepReason Agent")
    print("=" * 40)
    print("LangGraph res engine")
    print("enter q or 'quit' 2 exit\n")

    while True:
        try:
            q = input(" Query: ").strip()

            if q.lower() in ['quit', 'exit', 'q']:
                print(" bye!")
                break

            if not q:
                print(" enter valid q")
                continue

            res = run_res(q)

            print(f"\n REPORT:")
            print("-" * 40)
            print(res.get("final_report", "no report gen"))
            print("-" * 40)
            print(f"\n stats:")
            print(f"   • tot rev: {res.get('revision_number', 0)}")
            print(f"   • content pcs: {len(res.get('content', []))}")
            print(f"   • max rev: {res.get('max_revisions', 2)}")

        except KeyboardInterrupt:
            print("\nstopped. bye!")
            break
        except Exception as e:
            print(f"\n err: {e}")
            print("try again or chk cfg.")


if __name__ == "__main__":
    main()
