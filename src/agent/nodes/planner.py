
from langchain_core.prompts import ChatPromptTemplate
from src.agent.state import AgentState
from src.agent.nodes.models import ArchitecturePlan
from src.agent.config import llm
from src.agent.prompts import PLANNER_SYSTEM_PROMPT
from src.utils.json_parser import extract_json_object


PLANNER_OUTPUT_FORMAT_PROMPT = """
Return only a valid JSON object with this exact schema:
{{"steps": ["clear actionable step"], "thought_process": "brief rationale"}}
Do not wrap the JSON in markdown or add extra commentary.
"""

def planner_node(state: AgentState) -> dict:
    """
    Analyzes the requirement (and any plan feedback) and updates the state with a fresh/revised plan.
    
    Handles edge cases like:
    - Malformed feedback from previous errors
    - Missing or empty requirements
    - LLM failures
    """
    print("---  PLANNER NODE EXECUTING ---")
    requirement = state.get("requirement", "")
    feedback = state.get("plan_feedback")
    existing_plan = state.get("plan", [])
    
    # 1. Sanitize feedback to prevent template injection/parsing errors
    if feedback:
        # Clean feedback: escape curly braces that could break the template
        feedback_safe = str(feedback).replace("{", "{{").replace("}", "}}")
        existing_plan_safe = str(existing_plan).replace("{", "{{").replace("}", "}}")
        
        system_content = PLANNER_SYSTEM_PROMPT + (
            "\nREPLANNING CONTEXT:\n"
            "An execution error or feedback was encountered for the previous plan.\n"
            f"Previous Plan: {existing_plan_safe}\n"
            f"Feedback/Error: {feedback_safe}\n"
            "Please adjust the plan accordingly to resolve this feedback/error."
        )
    else:
        system_content = PLANNER_SYSTEM_PROMPT

    system_content += PLANNER_OUTPUT_FORMAT_PROMPT
        
    # 2. Create the prompt and chain
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_content),
            ("human", "{requirement}")
        ])
        
        # Create the chain and invoke it
        chain = prompt | llm
        llm_response = chain.invoke({"requirement": requirement})
        result = _parse_architecture_plan(llm_response.content)
        plan_steps = result.steps
        log_message = f"Planner Node: Successfully generated a plan with {len(plan_steps)} steps."
        print(f"✓ {log_message}")
    
    except Exception as e:
        # Handle template errors, API errors, or parsing errors
        error_msg = str(e)
        print(f"✗ Planner Node FAILED: {error_msg}")
        
        # If it's a template error, try a simplified version
        if "ChatPromptTemplate" in error_msg or "variables" in error_msg:
            print("[FALLBACK] Attempting simplified replanning...")
            try:
                # Simplified prompt without complex feedback
                simple_prompt = ChatPromptTemplate.from_messages([
                    ("system", PLANNER_SYSTEM_PROMPT + PLANNER_OUTPUT_FORMAT_PROMPT),
                    ("human", "{requirement}")
                ])
                chain = simple_prompt | llm
                llm_response = chain.invoke({"requirement": requirement})
                result = _parse_architecture_plan(llm_response.content)
                plan_steps = result.steps
                log_message = f"Planner Node: Generated fallback plan with {len(plan_steps)} steps."
                print(f"✓ {log_message}")
            except Exception as e2:
                plan_steps = []
                log_message = f"Planner Node: FAILED to generate plan (fallback also failed). Error: {str(e2)}"
                print(f"✗ {log_message}")
        else:
            plan_steps = []
            log_message = f"Planner Node: FAILED to generate plan. Error: {error_msg}"
    
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


def _parse_architecture_plan(response_text: str) -> ArchitecturePlan:
    plan_data = extract_json_object(response_text)
    plan_data.setdefault("thought_process", "No rationale provided.")
    return ArchitecturePlan.model_validate(plan_data)

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
