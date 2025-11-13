from typing import List, Dict, Optional

class WorkflowState:
    """Legacy state class for workflow (replaced by TypedDict in workflow_graph)."""
    def __init__(self):
        """Initialize an empty workflow state."""
        self.prompt: str = ""
        self.history: List[tuple[str, str]] = []
        self.workflow: Dict = {"structure": [], "data": []}
        self.partial_workflow: Dict = {"structure": [], "data": []}
        self.awaiting_input: bool = False
        self.next_question: str = ""
        self.intent: Optional[str] = None
        self.user_id: str = "default_user"