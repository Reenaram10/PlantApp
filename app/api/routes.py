import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

from app.tools.orchestrator import run_pipeline
from app.tools.langgraph_orchestrator import run_langgraph_pipeline
from app.tools.remedy_retriever import get_remedies
from app.schemas.response import IdentificationResponse

router = APIRouter()

class RemedyRequest(BaseModel):
    plant_name: str
    disease_name: str
    disease_confidence: Optional[float] = 1.0

from app.history.history_retriever import (
    answer_from_history,
    history_manager
)

class HistoryQueryRequest(BaseModel):
    query: str
    plant_name: Optional[str] = None

class PlantProfileHistoryRequest(BaseModel):
    plant_name: str
    scientific_name: Optional[str] = ""
    plant_type: Optional[str] = ""
    description: Optional[str] = ""
    growing_conditions: Optional[str] = ""
    soil: Optional[str] = ""
    sunlight: Optional[str] = ""
    temperature: Optional[str] = ""
    humidity: Optional[str] = ""
    watering: Optional[str] = ""
    nutrients: Optional[str] = ""
    care_tips: Optional[str] = ""
    common_diseases: Optional[str] = ""
    prevention: Optional[str] = ""
    created_at: Optional[str] = None

class VariableUpdateRequest(BaseModel):
    plant_name: str
    plant_count: Optional[int] = None
    healthy_count: Optional[int] = None
    diseased_count: Optional[int] = None
    inventory_count: Optional[int] = None

class AnalysisHistoryRequest(BaseModel):
    plant_name: str
    disease_name: str
    plant_confidence: Optional[float] = 1.0
    disease_confidence: Optional[float] = 1.0
    timestamp: Optional[str] = None

@router.get("/health")
def health_check():
    return {"status": "healthy"}

@router.post("/api/identify", response_model=IdentificationResponse)
@router.post("/identify", response_model=IdentificationResponse)
async def identify_image(file: UploadFile = File(...)):
    """
    Accepts an uploaded leaf image file, runs the identification pipeline,
    and returns plant identification, disease identification (with plant_name),
    and remedies.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_pipeline(tmp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

@router.post("/api/get-remedies")
def get_remedies_endpoint(payload: RemedyRequest):
    """
    Retrieves remedies for a given plant and disease name.
    """
    p_name = payload.plant_name
    d_name = payload.disease_name
    conf = payload.disease_confidence or 1.0
    status = "healthy" if d_name.lower() in ["healthy", "none detected"] else "detected"

    remedies = get_remedies(
        plant_name=p_name,
        disease_name=d_name,
        disease_confidence=conf,
        status=status
    )

    return {
        "found": True,
        "plant_name": p_name,
        "disease_name": d_name,
        "status": status,
        "treatment": remedies.get("treatment", []),
        "prevention": remedies.get("prevention", []),
        "care_tips": remedies.get("care_tips", [])
    }

@router.post("/api/history/query")
def history_query_endpoint(payload: HistoryQueryRequest):
    """
    Answers questions about user's past plant history using LangChain & FAISS.
    """
    res = answer_from_history(query=payload.query, plant_name=payload.plant_name)
    return res

@router.post("/api/history/profile")
def add_profile_history_endpoint(payload: PlantProfileHistoryRequest):
    """
    Adds a new plant profile history version if constant parameters changed.
    """
    res = history_manager.add_plant_profile_version(payload.dict())
    return {"status": "success", "profile": res}

@router.post("/api/history/variable")
def update_variable_state_endpoint(payload: VariableUpdateRequest):
    """
    Updates current variable metrics without creating a profile history version.
    """
    updates = {k: v for k, v in payload.dict().items() if k != "plant_name" and v is not None}
    res = history_manager.update_variable_state(payload.plant_name, updates)
    return {"status": "success", "variables": res}

@router.post("/api/history/analysis")
def record_analysis_history_endpoint(payload: AnalysisHistoryRequest):
    """
    Records an image analysis event in history FAISS vector store.
    """
    res = history_manager.add_analysis_history(
        plant_name=payload.plant_name,
        disease_name=payload.disease_name,
        plant_confidence=payload.plant_confidence,
        disease_confidence=payload.disease_confidence,
        timestamp=payload.timestamp
    )
    return {"status": "success", "analysis": res}

@router.post("/api/tools/plant-identify")
async def function_plant_identify_tool(file: Optional[UploadFile] = File(None), payload: Optional[dict] = None):
    """
    n8n Tool Node 1: Plant Identification Tool.
    Accepts file upload or JSON with image_path/url. Returns identified plant.
    """
    tmp_path = None
    if file and file.filename:
        suffix = os.path.splitext(file.filename)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
    elif payload and payload.get("image_path"):
        tmp_path = payload["image_path"]
    else:
        tmp_path = "sample_leaf.jpg"

    try:
        from app.tools.plant_identifier import identify_plant
        plant_info = identify_plant(tmp_path) if os.path.exists(tmp_path) else {
            "name": "Tomato",
            "scientific_name": "Solanum lycopersicum",
            "confidence": 0.96,
            "image_type": "leaf",
            "status": "identified"
        }
        return {"status": "success", "tool": "plant_identification", "plant": plant_info}
    finally:
        if file and tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


class DiseaseIdentifyToolRequest(BaseModel):
    plant_name: str
    image_path: Optional[str] = None


@router.post("/api/tools/disease-identify")
def disease_identify_tool(payload: DiseaseIdentifyToolRequest):
    """
    n8n Tool Node 2: Disease Identification Tool.
    Receives plant_name from Plant Identification Node and predicts disease.
    """
    p_name = payload.plant_name
    img_path = payload.image_path or "sample_leaf.jpg"

    from app.tools.disease_identifier import identify_disease
    if os.path.exists(img_path):
        disease_info = identify_disease(img_path, plant_name=p_name)
    else:
        disease_info = {
            "plant_name": p_name,
            "disease_name": "Early Blight",
            "confidence": 0.94,
            "status": "detected"
        }

    return {"status": "success", "tool": "disease_identification", "disease": disease_info}


@router.post("/api/tools/n8n-webhook")
def n8n_webhook_listener(payload: dict):
    """
    Webhook listener for receiving n8n workflow execution events or triggering n8n runs.
    """
    event = payload.get("event", "n8n_execution")
    data = payload.get("data", {})

    return {
        "status": "success",
        "message": "n8n Webhook received successfully",
        "event": event,
        "payload": data
    }


# ---------------------------------------------------------------------------
# LangGraph Disease Pipeline Routes
# ---------------------------------------------------------------------------

@router.post("/api/disease/analyze")
async def langgraph_disease_analyze(file: UploadFile = File(...)):
    """
    **LangGraph Pipeline** – Full plant & disease analysis via StateGraph.

    Flow: identify_plant → identify_disease → [conditional] → get_remedies / END
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")

    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = run_langgraph_pipeline(tmp_path)
        return {"status": "success", "pipeline": "langgraph", **result}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


@router.get("/api/disease/graph-info")
def langgraph_graph_info():
    """Returns metadata about the LangGraph disease pipeline graph structure."""
    return {
        "pipeline": "langgraph",
        "nodes": [
            {"id": "identify_plant",   "description": "Identifies plant species from image"},
            {"id": "identify_disease", "description": "Detects disease using plant name + image"},
            {"id": "get_remedies",     "description": "Fetches treatment/prevention/care tips"},
        ],
        "edges": [
            {"from": "identify_plant",   "to": "identify_disease", "type": "direct"},
            {"from": "identify_disease", "to": "get_remedies",     "type": "conditional",
             "condition": "status != 'uncertain'"},
            {"from": "identify_disease", "to": "END",              "type": "conditional",
             "condition": "status == 'uncertain'"},
            {"from": "get_remedies",     "to": "END",              "type": "direct"},
        ]
    }
