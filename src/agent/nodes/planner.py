
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import AgentState
from src.agent.nodes.models import ArchitecturePlan
from src.agent.config import llm
from src.agent.prompts import PLANNER_SYSTEM_PROMPT

def planner_node(state: AgentState) -> dict:
    """
    Analyzes the requirement and updates the state with a fresh implementation plan.
    """
    print("---  PLANNER NODE EXECUTING ---")
    requirement = state.get("requirement", "")
    
    # 1. Bind the Pydantic schema to force Mistral to return structured JSON
    structured_llm = llm.with_structured_output(ArchitecturePlan)
    
    # 2. Construct the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", PLANNER_SYSTEM_PROMPT),
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
    
    # 4. Return the dictionary of state updates
    return {
        "plan": plan_steps,
        "current_step": 0,
        "logs": [log_message],
        "is_complete": False
    }

# --- Local Testing Block ---
if __name__ == "__main__":
    # A quick mock state to test the node in isolation
    mock_state = AgentState(
        requirement="Create a simple Python script that prints 'Hello World', then run it.",
        plan=[],
        files=[],
        logs=[],
        current_step=0,
        is_complete=False
    )
    
    print("Testing Planner Node...")
    new_state = planner_node(mock_state)
    print("\nGenerated Plan:")
    for i, step in enumerate(new_state["plan"]):
        print(f"{i + 1}. {step}")