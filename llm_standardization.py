# llm_standardization.py
# Simple LLM-based standardization using existing Gemini models

import json
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class StandardizedResult(BaseModel):
    """Result of LLM standardization"""
    vendor: str = Field(description="Standardized vendor name")
    product_type: str = Field(description="Standardized product type")
    model_family: Optional[str] = Field(description="Standardized model family", default=None)
    specifications: Dict[str, str] = Field(description="Standardized specification names", default_factory=dict)
    confidence: float = Field(description="Confidence score 0-1", default=1.0)

class LLMStandardizer:
    """Dynamic LLM-based standardization using Gemini - fully configurable via prompts"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.1, 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=StandardizedResult)
        
        # Import the dynamic standardization prompt
        from prompts import standardization_prompt
        self.prompt = standardization_prompt
        
        self.chain = self.prompt | self.llm | self.parser
        
    def standardize_data(self, raw_data: Dict[str, Any], context: str = "") -> StandardizedResult:
        """
        Standardize raw vendor/product data using LLM
        
        Args:
            raw_data: Dictionary with vendor, product_type, model_family, specifications
            context: Additional context for standardization
            
        Returns:
            StandardizedResult with standardized names
        """
        try:
            result = self.chain.invoke({
                "vendor": raw_data.get("vendor", ""),
                "product_type": raw_data.get("product_type", ""),
                "model_family": raw_data.get("model_family", ""),
                "specifications": json.dumps(raw_data.get("specifications", {}), indent=2),
                "context": context,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Handle both dict and StandardizedResult objects
            if isinstance(result, dict):
                # Convert dict to StandardizedResult
                return StandardizedResult(
                    vendor=result.get("vendor", ""),
                    product_type=result.get("product_type", ""),
                    model_family=result.get("model_family"),
                    specifications=result.get("specifications", {}),
                    confidence=result.get("confidence", 1.0)
                )
            else:
                # Already a StandardizedResult object
                return result
        except Exception as e:
            logging.error(f"LLM standardization failed: {e}")
            # Fallback to basic standardization
            return self._fallback_standardize(raw_data)
    
    def standardize(self, raw_data: Dict[str, Any], context: str = "") -> StandardizedResult:
        """Backward compatibility method - calls standardize_data"""
        return self.standardize_data(raw_data, context)
    
    def _fallback_standardize(self, raw_data: Dict[str, Any]) -> StandardizedResult:
        """Simple fallback standardization"""
        vendor = raw_data.get("vendor", "").strip()
        product_type = raw_data.get("product_type", "").strip().lower()
        
        # Use dynamic vendor standardization instead of hardcoded mapping
        from standardization_utils import standardize_vendor_name
        standardized_vendor = standardize_vendor_name(vendor)
        
        # Use the dynamic standardization function instead of hardcoded logic
        from standardization_utils import suggest_standard_product_type
        standardized_product = suggest_standard_product_type(product_type)
            
        return StandardizedResult(
            vendor=standardized_vendor,
            product_type=standardized_product,
            model_family=raw_data.get("model_family"),
            specifications=raw_data.get("specifications", {}),
            confidence=0.7  # Lower confidence for fallback
        )

# Global standardizer instance
_llm_standardizer = None

def get_llm_standardizer() -> LLMStandardizer:
    """Get global LLM standardizer instance"""
    global _llm_standardizer
    if _llm_standardizer is None:
        _llm_standardizer = LLMStandardizer()
    return _llm_standardizer

def standardize_with_llm(data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """
    Standardize data using LLM
    
    Args:
        data: Raw data to standardize
        context: Additional context
        
    Returns:
        Standardized data dictionary
    """
    standardizer = get_llm_standardizer()
    result = standardizer.standardize(data, context)
    
    # Handle both Pydantic model and dictionary results
    if hasattr(result, 'vendor'):
        # Pydantic model result
        return {
            "vendor": result.vendor,
            "product_type": result.product_type, 
            "model_family": result.model_family,
            "specifications": result.specifications,
            "confidence": result.confidence,
            "method": "llm"
        }
    else:
        # Dictionary result (from fallback)
        return {
            "vendor": result.get("vendor", ""),
            "product_type": result.get("product_type", ""), 
            "model_family": result.get("model_family", ""),
            "specifications": result.get("specifications", {}),
            "confidence": result.get("confidence", 0.7),
            "method": "fallback"
        }

# Integration functions for existing code
def standardize_vendor_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize vendor analysis result using LLM"""
    if not analysis_result:
        return analysis_result
        
    # Extract data for standardization
    raw_data = {
        "vendor": analysis_result.get("vendor", ""),
        "product_type": analysis_result.get("product_type", ""),
        "model_family": analysis_result.get("model_family", ""),
        "specifications": analysis_result.get("specifications", {})
    }
    
    # Get context from analysis
    context = f"Product analysis with {len(analysis_result.get('matched_models', []))} models"
    
    # Standardize
    standardized = standardize_with_llm(raw_data, context)
    
    # Update analysis result
    result = analysis_result.copy()
    result.update({
        "vendor": standardized["vendor"],
        "product_type": standardized["product_type"],
        "model_family": standardized["model_family"],
        "standardization_confidence": standardized["confidence"],
        "standardization_method": "llm"
    })
    
    return result

def standardize_ranking_result(ranking_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize ranking result using LLM"""
    if not ranking_result:
        return ranking_result
        
    result = ranking_result.copy()
    
    # Standardize each ranked vendor
    if "ranked_vendors" in result:
        for vendor_data in result["ranked_vendors"]:
            raw_data = {
                "vendor": vendor_data.get("vendor", ""),
                "product_type": vendor_data.get("product_type", ""),
                "model_family": vendor_data.get("model_family", ""),
                "specifications": vendor_data.get("specifications", {})
            }
            
            standardized = standardize_with_llm(raw_data)
            vendor_data.update({
                "vendor": standardized["vendor"],
                "product_type": standardized["product_type"], 
                "model_family": standardized["model_family"],
                "standardization_confidence": standardized["confidence"]
            })
    
    return result

if __name__ == "__main__":
    # Test the LLM standardizer
    print("🤖 Testing LLM Standardization...")
    
    test_data = {
        "vendor": "abb ltd",
        "product_type": "pressure measurement device",
        "model_family": "266GST",
        "specifications": {
            "operating temp": "-40 to 85°C",
            "pressure range": "0-100 bar",
            "signal output": "4-20mA"
        }
    }
    
    standardizer = get_llm_standardizer()
    result = standardizer.standardize(test_data, "Industrial pressure measurement")
    
    print(f"Original: {test_data}")
    print(f"Standardized: {result}")
    print(f"Confidence: {result.confidence}")