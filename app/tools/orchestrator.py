"""
Legacy entry-point kept for backward compatibility.
All pipeline logic now runs through the LangGraph StateGraph.
"""
from app.tools.langgraph_orchestrator import run_langgraph_pipeline


def run_pipeline(image_path: str) -> dict:
    """
    Executes the plant & disease identification pipeline via LangGraph.

    Flow (managed by LangGraph StateGraph):
        Image → identify_plant → identify_disease
                → [conditional] → get_remedies → END
                               └─ (uncertain)  → END

    Args:
        image_path: Path to the leaf image.

    Returns:
        dict containing 'plant', 'disease', and 'remedies'.
    """
    return run_langgraph_pipeline(image_path)
