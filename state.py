from typing import TypedDict, List

class AgentState(TypedDict):
    query: str
    plan: List[str]
    content: List[str]
    revision_number: int
    max_revisions: int
    final_report: str   
