"""
Pydantic schemas for prediction endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """
    Schema representing incoming raw transaction telemetry.
    
    Excludes the target is_fraud. Maps exactly to database schema fields.
    """
    trans_date_trans_time: str = Field(..., description="Timestamp of transaction (YYYY-MM-DD HH:MM:SS).", example="2020-06-21 12:14:30")
    cc_num: int = Field(..., description="Credit card identification number.")
    merchant: str = Field(..., description="Merchant store name.")
    category: str = Field(..., description="Merchant business category.")
    amt: float = Field(..., description="Transaction purchase amount.")
    first: str = Field(..., description="Cardholder first name.")
    last: str = Field(..., description="Cardholder last name.")
    gender: str = Field(..., description="Cardholder gender (M/F).")
    street: str = Field(..., description="Cardholder street address.")
    city: str = Field(..., description="Cardholder city.")
    state: str = Field(..., description="Cardholder state abbreviation.")
    zip: int = Field(..., description="Cardholder billing ZIP code.")
    lat: float = Field(..., description="Cardholder latitude coordinate.")
    long: float = Field(..., description="Cardholder longitude coordinate.")
    city_pop: int = Field(..., description="Cardholder city population size.")
    job: str = Field(..., description="Cardholder occupation profession.")
    dob: str = Field(..., description="Cardholder date of birth (YYYY-MM-DD).", example="1988-04-12")
    trans_num: str = Field(..., description="Unique alphanumeric identifier of transaction.")
    unix_time: int = Field(..., description="Unix epoch timestamp value.")
    merch_lat: float = Field(..., description="Merchant latitude coordinate.")
    merch_long: float = Field(..., description="Merchant longitude coordinate.")


class PredictionResponse(BaseModel):
    """Response containing decision results and metrics for a single prediction request."""
    is_fraud: int = Field(..., description="Decision classification classification (1 = Fraud, 0 = Legitimate).")
    probability: float = Field(..., description="Model confidence probability prediction value.")
    threshold_used: float = Field(..., description="Classification threshold border value used.")
    model_version: str = Field(..., description="Model version run ID.")
    latency_ms: float = Field(..., description="Compute latency execution duration in milliseconds.")


class BatchPredictionResponse(BaseModel):
    """Response containing list of decision results for a batch prediction request."""
    predictions: list[PredictionResponse] = Field(..., description="List of single prediction outcomes.")
    batch_size: int = Field(..., description="Number of items processed.")
    latency_ms: float = Field(..., description="Total batch latency execution duration in milliseconds.")
