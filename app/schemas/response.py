from typing import List, Optional
from pydantic import BaseModel, Field

class PlantInfo(BaseModel):
    name: str = Field(..., description="Common name of the identified plant")
    scientific_name: str = Field(..., description="Scientific name of the identified plant")
    confidence: float = Field(..., description="Identification confidence score (0.0 to 1.0)")
    image_type: str = Field(default="leaf", description="Type of image analyzed (leaf, flower, fruit)")
    status: str = Field(default="identified", description="Status of plant identification")

class DiseaseInfo(BaseModel):
    plant_name: str = Field(..., description="Plant name from Plant Identification Tool")
    disease_name: str = Field(..., description="Name of detected disease, 'None Detected', or 'Unknown Disease'")
    confidence: float = Field(..., description="Disease confidence score (0.0 to 1.0)")
    status: str = Field(..., description="Status of disease identification ('detected', 'healthy', or 'uncertain')")

class RemediesInfo(BaseModel):
    treatment: List[str] = Field(default_factory=list, description="Treatment steps")
    prevention: List[str] = Field(default_factory=list, description="Prevention tips")
    care_tips: List[str] = Field(default_factory=list, description="General care recommendations")

class IdentificationResponse(BaseModel):
    plant: PlantInfo
    disease: DiseaseInfo
    remedies: RemediesInfo
