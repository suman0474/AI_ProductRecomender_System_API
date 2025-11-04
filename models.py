
# models.py
# Contains Pydantic models for structured output
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ProductMatch(BaseModel):
    product_name: str = Field(description="Only the specific submodel/series name (e.g., 'STD800', 'SMV800') - just the short model identifier, not full descriptions")
    product_type: str = Field(description="type of the matching product")

    vendor: str = Field(description="Vendor/manufacturer name")
    match_score: int = Field(description="Match score from 0-100")
    requirements_match: bool = Field(description="Whether the product meets ALL critical/mandatory requirements (True) or has fundamental gaps (False)")
    reasoning: str = Field(description="Detailed parameter-by-parameter analysis of why this product matches the requirements, including user requirement vs product specification for each parameter")
    limitations: str = Field(description="Detailed analysis of any gaps, limitations, or areas needing verification for each parameter")
    

class VendorAnalysis(BaseModel):
    vendor_matches: List[ProductMatch] = Field(description="Best products match from each vendor")

class RankedProduct(BaseModel):
    rank: int = Field(description="Overall ranking position")
    product_name: str = Field(description="Only the specific submodel/series name (e.g., 'STD800', 'SMV800') - just the short model identifier, not full descriptions")
    vendor: str = Field(description="Vendor name")
    
    overall_score: int = Field(description="Overall score 0-100")
    requirements_match: bool = Field(description="Whether the product meets ALL critical/mandatory requirements (True) or has fundamental gaps (False)")
    key_strengths: str = Field(description="Detailed parameter-by-parameter analysis of key strengths showing user requirement vs product specification and basis for match")
    concerns: str = Field(description="Detailed parameter-by-parameter analysis of potential concerns, limitations, or verification needs")

class OverallRanking(BaseModel):
    ranked_products: List[RankedProduct] = Field(description="All products ranked by suitability")

class RequirementValidation(BaseModel):
    
    provided_requirements: Dict[str, Any] = Field(description="Requirements that were provided")
    product_type: str = Field(description="Detected product type from user input")
