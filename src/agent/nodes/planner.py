
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import AgentState
from src.agent.nodes.models import ArchitecturePlan
from src.agent.config import llm
from src.agent.prompts import PLANNER_SYSTEM_PROMPT

def planner_node(state: AgentState) -> dict:
    """
    Analyzes the requirement (and any plan feedback) and updates the state with a fresh/revised plan.
    """
    print("---  PLANNER NODE EXECUTING ---")
    requirement = state.get("requirement", "")
    feedback = state.get("plan_feedback")
    existing_plan = state.get("plan", [])
    
    # 1. Bind the Pydantic schema to force Mistral to return structured JSON
    structured_llm = llm.with_structured_output(ArchitecturePlan)
    
    # 2. Adjust prompt if there is feedback/replanning needed
    if feedback:
        system_content = PLANNER_SYSTEM_PROMPT + (
            "\nREPLANNING CONTEXT:\n"
            "An execution error or feedback was encountered for the previous plan.\n"
            f"Previous Plan: {existing_plan}\n"
            f"Feedback/Error: {feedback}\n"
            "Please adjust the plan accordingly to resolve this feedback/error."
        )
    else:
        system_content = PLANNER_SYSTEM_PROMPT
        
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_content),
        ("human", "{requirement}")
    ])
    
    # 3. Create the chain and invoke it
    chain = prompt | structured_llm
    
    try:
        result: ArchitecturePlan = chain.invoke({"requirement": requirement})
        plan_steps = result.steps
        log_message = f"Planner Node: Successfully generated a plan with {len(plan_steps)} steps."
    except Exception as e:
        # Fallback in case of an API or formatting error
        plan_steps = []
        log_message = f"Planner Node: FAILED to generate plan. Error: {str(e)}"
    
    # 4. Return the dictionary of state updates, resetting loop/error indicators
    return {
        "plan": plan_steps,
        "current_step": 0,
        "logs": [log_message],
        "is_complete": False,
        "last_error": None,
        "retry_count": 0,
        "plan_feedback": None
    }

# --- Local Testing Block ---
if __name__ == "__main__":
    # A quick mock state to test the node in isolation
    mock_state = AgentState(
        requirement="Create a simple Python FastAPI for weather endpoints.",
        plan=[],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False,
        last_error=None,
        retry_count=0,
        plan_feedback=None,
        user_feedback=None
    )
    
    print("Testing Planner Node...")
    new_state = planner_node(mock_state)
    print("\nGenerated Plan:")
    for i, step in enumerate(new_state["plan"]):
        print(f"{i + 1}. {step}")
    print("\nLogs:")
    for log in new_state["logs"]:
        print(f" - {log}")
