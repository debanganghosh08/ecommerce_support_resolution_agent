"""
Structured Output Schemas for the Agentic RAG System.

This module defines Pydantic models to strictly enforce the output format 
for both the E-commerce Support Agent and the Academic Course Planner.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class EcommerceSupportResponse(BaseModel):
    """
    Schema for E-commerce Support Resolution Writer output.
    """
    classification: str = Field(
        ..., 
        description="The type of issue (e.g., Refund, Shipping) along with a confidence score."
    )
    clarifying_questions: List[str] = Field(
        ...,
        max_length=3,
        description="Up to 3 optional questions to ask the customer if more context is needed."
    )
    decision: str = Field(
        ..., 
        description="The final decision: 'Approve', 'Deny', 'Partial', or 'Escalate'."
    )
    rationale: str = Field(
        ..., 
        description="A policy-based explanation justifying the decision."
    )
    citations: List[str] = Field(
        ..., 
        description="A list of strict citations formatted as [Source URL + Section/Chunk ID] backing the rationale."
    )
    customer_response_draft: str = Field(
        ..., 
        description="A polite draft email/message responding to the customer."
    )
    next_steps_internal_notes: str = Field(
        ..., 
        description="Any internal actions required, such as forwarding to finance or updating inventory."
    )


class CoursePlanningResponse(BaseModel):
    """
    Schema for Academic Course Planner output.
    """
    answer_plan: str = Field(
        ..., 
        description="The proposed sequence of courses or the direct answer to the student's query."
    )
    why: str = Field(
        ..., 
        description="Explanation of how the plan satisfies prerequisites and degree requirements."
    )
    citations: List[str] = Field(
        ..., 
        description="A list of strict citations formatted as [Source URL + Section/Chunk ID] containing the rules used."
    )
    clarifying_questions: List[str] = Field(
        default_factory=list,
        description="Questions for the student if details like past credits are missing."
    )
    assumptions_not_in_catalog: str = Field(
        default="",
        description="Any constraints or assumptions not explicitly found in the catalog."
    )
