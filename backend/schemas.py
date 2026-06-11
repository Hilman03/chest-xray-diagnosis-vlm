"""
backend/schemas.py
==================
Pydantic data models for FastAPI request and response validation.
"""

from pydantic import BaseModel
from typing import Optional, List


class UploadResponse(BaseModel):
    image_id        : str
    filename        : str
    status          : str
    message         : str
    is_dicom        : bool = False


class DiseaseScore(BaseModel):
    disease : str
    score   : float
    percent : str


class AnalyzeResponse(BaseModel):
    image_id        : str
    filename        : str
    status          : str
    disease_label   : str
    top_diseases    : list
    vlm_caption     : str
    llm_report      : str
    llm_backend     : str
    vlm_time        : float
    llm_time        : float
    total_time      : float
    analyzed_at     : str


class ReportResponse(BaseModel):
    image_id        : str
    filename        : str
    status          : str
    disease_label   : Optional[str] = ""
    top_diseases    : Optional[list] = []
    vlm_caption     : Optional[str] = ""
    llm_report      : Optional[str] = ""
    llm_backend     : Optional[str] = ""
    vlm_time        : Optional[float] = 0.0
    llm_time        : Optional[float] = 0.0
    total_time      : Optional[float] = 0.0
    uploaded_at     : Optional[str] = ""
    analyzed_at     : Optional[str] = ""
    is_dicom        : Optional[bool] = False
    dicom_metadata  : Optional[dict] = {}


class HealthResponse(BaseModel):
    status          : str
    message         : str
    mongodb         : str
    total_reports   : int