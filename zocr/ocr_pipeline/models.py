"""Data models for the modular OCR pipeline.

These models keep inputs and outputs explicit across each component of the
pipeline so implementations can be swapped without changing data exchange
formats. Pydantic is used for validation and convenience when constructing
objects in tests or future integrations.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box in pixel coordinates."""

    x: int = Field(..., ge=0)
    y: int = Field(..., ge=0)
    width: int = Field(..., ge=1)
    height: int = Field(..., ge=1)


class PageInput(BaseModel):
    """Single page or image to process."""

    document_id: str
    page_number: int = Field(..., ge=1)
    image: object
    dpi: Optional[int] = Field(None, ge=1)


class RegionType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


class SegmentedRegion(BaseModel):
    """Region produced by the segmenter before classification."""

    region_id: str
    bounding_box: BoundingBox
    image_crop: object
    confidence: float = Field(..., ge=0.0, le=1.0)
    reading_order: Optional[int] = Field(None, ge=0)


class ClassifiedRegion(BaseModel):
    """Region after classification."""

    region_id: str
    bounding_box: BoundingBox
    classification: RegionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    reading_order: Optional[int] = Field(None, ge=0)


class WordInfo(BaseModel):
    word: str
    bounding_box: Optional[BoundingBox]
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)


class TextOcrResult(BaseModel):
    region_id: str
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    language: Optional[str] = None
    words: Optional[List[WordInfo]] = None


class ImageCaptionResult(BaseModel):
    region_id: str
    caption: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    detailed_description: Optional[str] = None
    detected_objects: Optional[List[str]] = None


class TableData(BaseModel):
    headers: List[str]
    rows: List[dict]
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None


class TableExtractionResult(BaseModel):
    region_id: str
    table_data: TableData
    confidence: float = Field(..., ge=0.0, le=1.0)
    format: str = Field("simple")


class RegionOutput(BaseModel):
    region_id: str
    type: RegionType
    bounding_box: BoundingBox
    reading_order: Optional[int]
    content: dict


class DocumentMetadata(BaseModel):
    total_regions: int
    text_regions: int
    image_regions: int
    table_regions: int


class DocumentOutput(BaseModel):
    document_id: str
    page_number: int
    regions: List[RegionOutput]
    metadata: DocumentMetadata

