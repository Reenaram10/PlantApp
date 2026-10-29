"""
LangGraph-based orchestrator for the plant disease detection pipeline.

Graph flow:
  identify_plant
       │
  identify_disease
       │
  ┌────┴──────────────────────┐
  │ status == "uncertain"      │ status == "detected" / "healthy"
  ▼                            ▼
 END                      get_remedies ──► END
"""

from __future__ import annotations

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.tools.plant_identifier import identify_plant
from app.tools.disease_identifier import identify_disease
from app.tools.remedy_retriever import get_remedies


# ---------------------------------------------------------------------------
# Shared pipeline state
# ---------------------------------------------------------------------------

class PipelineState(TypedDict, total=False):
    """Shared state that flows through all LangGraph nodes."""
    image_path: str
    plant: dict          # output of identify_plant
    disease: dict        # output of identify_disease
    remedies: dict       # output of get_remedies
    error: Optional[str]


# ---------------------------------------------------------------------------
# Nodes  (each receives the full state dict and returns a partial update)
# ---------------------------------------------------------------------------

def node_identify_plant(state: PipelineState) -> PipelineState:
    """Node 1 – identify the plant species from the image."""
    try:
        plant_info = identify_plant(state["image_path"])
        return {"plant": plant_info}
    except Exception as exc:
        return {"plant": {"name": "Unknown", "scientific_name": "", "confidence": 0.0,
                          "image_type": "leaf", "status": "error"},
                "error": str(exc)}


def node_identify_disease(state: PipelineState) -> PipelineState:
    """Node 2 – identify the disease using image + plant name from Node 1."""
    plant_name = state.get("plant", {}).get("name", "Unknown")
    try:
        disease_info = identify_disease(
            image_path=state["image_path"],
            plant_name=plant_name
        )
        return {"disease": disease_info}
    except Exception as exc:
        return {"disease": {"plant_name": plant_name, "disease_name": "Unknown Disease",
                            "confidence": 0.0, "status": "uncertain"},
                "error": str(exc)}


def node_get_remedies(state: PipelineState) -> PipelineState:
    """Node 3 – fetch treatment / prevention / care tips."""
    disease = state.get("disease", {})
    try:
        remedies = get_remedies(
            plant_name=disease.get("plant_name", "Unknown"),
            disease_name=disease.get("disease_name", "Unknown Disease"),
            disease_confidence=disease.get("confidence", 0.0),
            status=disease.get("status", "uncertain")
        )
        return {"remedies": remedies}
    except Exception as exc:
        return {"remedies": {"treatment": [], "prevention": [], "care_tips": []},
                "error": str(exc)}


# ---------------------------------------------------------------------------
# Conditional edge: after disease identification decide whether to get remedies
# ---------------------------------------------------------------------------

def _should_get_remedies(state: PipelineState) -> str:
    """
    If disease is uncertain (or an error occurred) we skip remedies entirely.
    Otherwise we proceed to the remedy node.
    """
    status = state.get("disease", {}).get("status", "uncertain")
    if status == "uncertain":
        return "skip"
    return "get_remedies"


# ---------------------------------------------------------------------------
# Build and compile the graph (done once at module import)
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    g = StateGraph(PipelineState)

    # Register nodes
    g.add_node("identify_plant",   node_identify_plant)
    g.add_node("identify_disease", node_identify_disease)
    g.add_node("get_remedies",     node_get_remedies)

    # Entry point
    g.set_entry_point("identify_plant")

    # Edges
    g.add_edge("identify_plant", "identify_disease")
    g.add_conditional_edges(
        "identify_disease",
        _should_get_remedies,
        {
            "get_remedies": "get_remedies",
            "skip": END,
        }
    )
    g.add_edge("get_remedies", END)

    return g.compile()


_graph = _build_graph()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_langgraph_pipeline(image_path: str) -> dict:
    """
    Execute the LangGraph disease detection pipeline.

    Args:
        image_path: Absolute or relative path to the leaf/plant image.

    Returns:
        dict with keys: 'plant', 'disease', 'remedies'
        Mirrors the shape returned by the original run_pipeline().
    """
    initial_state: PipelineState = {
        "image_path": image_path,
        "plant": {},
        "disease": {},
        "remedies": {"treatment": [], "prevention": [], "care_tips": []},
        "error": None,
    }

    final_state = _graph.invoke(initial_state)

    # Log to history vector store (best-effort, same as original orchestrator)
    try:
        from app.history.history_retriever import history_manager
        history_manager.add_analysis_history(
            plant_name=final_state["plant"].get("name", "Unknown"),
            disease_name=final_state["disease"].get("disease_name", "Unknown"),
            plant_confidence=float(final_state["plant"].get("confidence", 1.0)),
            disease_confidence=float(final_state["disease"].get("confidence", 1.0)),
        )
    except Exception as exc:
        print(f" [LangGraph] History log notice: {exc}")

    return {
        "plant":    final_state.get("plant", {}),
        "disease":  final_state.get("disease", {}),
        "remedies": final_state.get("remedies", {"treatment": [], "prevention": [], "care_tips": []}),
    }
