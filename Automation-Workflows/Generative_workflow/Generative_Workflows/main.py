from agents.workflow_graph import graph
from models.workflow_state import WorkflowState
from services.history_service import HistoryService
from utils.logger import logger


def main():
    user_id = "user1"  # Replace with auth in production
    history_service = HistoryService()
    state = WorkflowState(user_id=user_id, history=history_service.load(user_id))
    print("Agent: Hey there! I’m here to help with Unizo workflows or SCM tools. What’s on your mind?")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input and state.awaiting_input:
                print(f"Agent: {state.next_question}")
                continue
            elif not user_input:
                print("Agent: Give me something to work with—what’s your next move?")
                continue

            state.prompt = user_input
            state.awaiting_input = False

            # Invoke the graph and get the updated state
            result = graph.invoke(state)

            # Convert LangGraph's dictionary output back to WorkflowState
            state = WorkflowState(**{k: v for k, v in result.items() if k in WorkflowState.__dataclass_fields__})

            print(f"Agent: {state.history[-1][1]}")
            history_service.save(user_id, state.history)
        except KeyboardInterrupt:
            print("\nAgent: Goodbye!")
            break
        except Exception as e:
            logger.error(f"Graph invocation failed: {e}")
            print("Agent: Oops, something went wrong. Let’s try again—what’s your question?")
            state.intent = None


if __name__ == "__main__":
    main()