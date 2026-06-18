from typing import TypedDict, List, Annotated
import operator

class AgentState(TypedDict):
    """
    The central memory store for the AI Dev Agent's ReAct cycle.
    
    Fields:
    - requirement: The original natural language prompt provided by the user.
    - plan: An ordered list of atomic implementation steps to fulfill the requirement.
    - files: A registry of file paths generated or modified by the agent.
    - logs: An append-only list of execution outputs, errors, and system status messages.
    - current_step: An integer pointer indicating the active index in the 'plan'.
    - is_complete: A global flag that, when True, triggers the graph to terminate.
    """
    requirement: str
    plan: List[str]
    files: List[str]
    logs: Annotated[List[str], operator.add]
    current_step: int
    is_complete: bool