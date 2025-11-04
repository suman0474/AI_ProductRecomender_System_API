# advanced_parameters.py
# Advanced parameter discovery from vendor websites using a fully LLM-based approach.

import json
import logging
import requests
from typing import Dict, Any, List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time

# Load environment variables
load_dotenv()

# --- Pydantic Models for Data Structuring ---
class VendorInfo(BaseModel):
    """Vendor information with website"""
    vendor: str = Field(description="Vendor/manufacturer name")
    website: str = Field(description="Official website URL")
    confidence: float = Field(description="Confidence score 0-1", default=1.0)

class AdvancedParameters(BaseModel):
    """Advanced parameters discovered from vendors"""
    vendor: str = Field(description="Vendor name")
    parameters: List[str] = Field(description="List of parameter names found")
    source_url: str = Field(description="Source URL where parameters were found")

# --- Main Discovery Class ---
class AdvancedParametersDiscovery:
    """Discovers advanced parameters from vendor websites"""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro", # Note: As of this writing, gemini-2.5-pro might be a placeholder; use a valid available model
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY1")
        )
        self.parser = JsonOutputParser(pydantic_object=VendorInfo)

    def get_top_vendors(self, product_type: str, count: int = 10) -> List[VendorInfo]:
        """Get top vendors for a product type using LLM"""
        prompt = ChatPromptTemplate.from_template("""
You are an expert in industrial equipment manufacturers. 
Given the product type "{product_type}", identify the top {count} vendors/manufacturers and their official websites.
Return a JSON array with vendor information:
[
    {{"vendor": "ABB", "website": "https://new.abb.com", "confidence": 0.95}},
    {{"vendor": "Emerson", "website": "https://www.emerson.com", "confidence": 0.95}}
]
Focus on major, well-known manufacturers. Include confidence scores based on their market presence.
Return ONLY the JSON array, no additional text.
""")
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "product_type": product_type,
                "count": count
            })
            cleaned_response = response.strip().replace('```json', '').replace('```', '')
            vendors_data = json.loads(cleaned_response)
            vendors = [VendorInfo(**vendor_data) for vendor_data in vendors_data[:count]]
            logging.info(f"Found {len(vendors)} vendors for {product_type}")
            return vendors
        except Exception as e:
            logging.error(f"Failed to get top vendors: {e}")
            return self._get_fallback_vendors()

    def _get_fallback_vendors(self) -> List[VendorInfo]:
        """Fallback vendor list if LLM fails"""
        common_vendors_data = [
            {"vendor": "ABB", "website": "https://new.abb.com", "confidence": 0.9},
            {"vendor": "Emerson", "website": "https://www.emerson.com", "confidence": 0.9},
            {"vendor": "Siemens", "website": "https://www.siemens.com", "confidence": 0.85},
            {"vendor": "Endress+Hauser", "website": "https://www.endress.com", "confidence": 0.85},
            {"vendor": "Honeywell", "website": "https://www.honeywell.com", "confidence": 0.8},
        ]
        return [VendorInfo(**data) for data in common_vendors_data]

    def scrape_vendor_parameters(self, vendor: VendorInfo, product_type: str) -> AdvancedParameters:
        """Scrape parameters from vendor website"""
        try:
            search_urls = self._build_search_urls(vendor.website, product_type)
            all_parameters = set()
            source_url = vendor.website
            
            for url in search_urls[:3]:  # Limit to top 3 URLs
                try:
                    parameters = self._extract_parameters_from_url(url, product_type)
                    if parameters:
                        all_parameters.update(parameters)
                        source_url = url
                        if len(all_parameters) >= 10:
                            break
                except Exception as e:
                    logging.warning(f"Failed to scrape {url}: {e}")
                    continue
                time.sleep(1)

            # If scraping finds nothing, use the new LLM-based fallback
            if not all_parameters:
                logging.warning(f"Scraping failed for {vendor.vendor}. Using LLM knowledge as fallback.")
                all_parameters = self._get_llm_fallback_parameters(vendor.vendor, product_type)

            return AdvancedParameters(
                vendor=vendor.vendor,
                parameters=list(all_parameters)[:15],
                source_url=source_url
            )
        except Exception as e:
            logging.error(f"Failed to scrape parameters for {vendor.vendor}: {e}")
            # Fallback to LLM-generated typical parameters if scraping fails entirely
            return AdvancedParameters(
                vendor=vendor.vendor,
                parameters=self._get_llm_fallback_parameters(vendor.vendor, product_type),
                source_url=vendor.website
            )

    def _build_search_urls(self, base_website: str, product_type: str) -> List[str]:
        """Build search URLs for product-specific pages"""
        search_paths = [
            f"/products/{product_type.replace(' ', '-')}",
            f"/{product_type.replace(' ', '-')}",
            f"/industrial-instrumentation/{product_type.replace(' ', '-')}",
            "/products",
            "/solutions"
        ]
        return [urljoin(base_website, path) for path in search_paths]

    def _extract_parameters_from_url(self, url: str, product_type: str) -> List[str]:
        """Extract parameter names from a URL's content using an LLM"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.get_text()
            return self._extract_parameters_with_llm(text_content, product_type)
        except Exception as e:
            logging.warning(f"Failed to extract from {url}: {e}")
            return []

    def _extract_parameters_with_llm(self, text_content: str, product_type: str) -> List[str]:
        """Use LLM to extract parameter names from text content"""
        prompt = ChatPromptTemplate.from_template("""
You are an expert in industrial equipment specifications.
Extract advanced parameter names from the following text content about {product_type}.
Text content (first 3000 chars):
{text_content}

Return ONLY a JSON array of parameter names specific to {product_type}. 
Focus on technical specs, features, and config options.
Avoid generic terms like "price", "description", "manual".
Return ONLY the JSON array: ["param1", "param2"]
""")
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "text_content": text_content[:3000],
                "product_type": product_type
            })
            cleaned_response = response.strip().replace('```json', '').replace('```', '')
            parameters = json.loads(cleaned_response)
            
            cleaned_parameters = []
            if isinstance(parameters, list):
                for param in parameters:
                    if isinstance(param, str) and len(param) > 2:
                        clean_param = re.sub(r'[^a-zA-Z0-9\s]', '', param).lower().replace(' ', '_')
                        if clean_param and clean_param not in cleaned_parameters:
                            cleaned_parameters.append(clean_param)
            return cleaned_parameters[:10]
        except Exception as e:
            logging.warning(f"LLM parameter extraction from text failed: {e}")
            return []

    def _get_llm_fallback_parameters(self, vendor: str, product_type: str) -> List[str]:
        """If scraping fails, ask the LLM directly for typical parameters."""
        prompt = ChatPromptTemplate.from_template("""
You are an expert in industrial equipment. 
List the 8 most common advanced technical specifications for a "{product_type}" from a major manufacturer like "{vendor}".
Focus on technical, non-obvious parameters. 
Return ONLY a JSON array of the parameter names, formatted in snake_case: ["param_one", "param_two"]
""")
        try:
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({
                "product_type": product_type,
                "vendor": vendor
            })
            cleaned_response = response.strip().replace('```json', '').replace('```', '')
            parameters = json.loads(cleaned_response)
            if isinstance(parameters, list):
                return [str(p) for p in parameters][:8]
            return []
        except Exception as e:
            logging.error(f"LLM fallback parameter generation failed: {e}")
            return []

def get_existing_parameters_from_schema(product_type: str) -> set:
    """Helper to load and parse existing schema parameters."""
    try:
        from loading import load_requirements_schema
        existing_schema = load_requirements_schema(product_type)
        existing_parameters = set()
        
        if not existing_schema:
            return existing_parameters

        # Extract from mandatory requirements
        for fields in existing_schema.get("mandatory_requirements", {}).values():
            if isinstance(fields, dict):
                existing_parameters.update(fields.keys())
                
        # Extract from optional requirements
        for fields in existing_schema.get("optional_requirements", {}).values():
            if isinstance(fields, dict):
                existing_parameters.update(fields.keys())
                
        logging.info(f"Found {len(existing_parameters)} existing parameters in schema.")
        return existing_parameters
    except ImportError:
        logging.warning("`loading` module not found. Cannot filter existing parameters.")
        return set()
    except Exception as e:
        logging.error(f"Failed to load or parse existing schema: {e}")
        return set()

# --- Main Orchestration Function ---
def discover_advanced_parameters(product_type: str) -> Dict[str, Any]:
    """Main function to discover advanced parameters for a product type."""
    try:
        discovery = AdvancedParametersDiscovery()
        existing_parameters = get_existing_parameters_from_schema(product_type)
        vendors = discovery.get_top_vendors(product_type, count=10)
        
        all_vendor_parameters = []
        # This map will hold the normalized version as key and the "clean" param as value
        # This ensures we only keep one version (preferably the most specific) of each concept
        normalized_to_original_map = {} 

        for vendor in vendors[:5]:  # Limit to top 5 vendors for scraping
            try:
                vendor_params = discovery.scrape_vendor_parameters(vendor, product_type)
                all_vendor_parameters.append(vendor_params)
                
                for param in vendor_params.parameters:
                    normalized_param = param.lower().strip().replace('_', '')
                    
                    if not normalized_param: # Skip empty strings
                        continue

                    # --- Start New Logic ---
                    
                    # Check 1: Is this new param a substring of an existing, more specific param?
                    # (e.g., new='responsetime', existing='totalresponsetime')
                    found_more_specific = False
                    for existing_norm in list(normalized_to_original_map.keys()):
                        if normalized_param in existing_norm and normalized_param != existing_norm:
                            # The existing param is more specific, so we do nothing with the new, general param.
                            found_more_specific = True
                            break
                    
                    if found_more_specific:
                        continue # Skip this new, less-specific parameter

                    # Check 2: Does this new param contain an existing, less specific param?
                    # (e.g., new='totalresponsetime', existing='responsetime')
                    found_less_specific_to_remove = []
                    for existing_norm in list(normalized_to_original_map.keys()):
                        if existing_norm in normalized_param and existing_norm != normalized_param:
                            # This new param is more specific. Mark the old, general param for removal.
                            found_less_specific_to_remove.append(existing_norm)
                    
                    # Remove all the less-specific params this new one replaces
                    for norm_to_remove in found_less_specific_to_remove:
                        if norm_to_remove in normalized_to_original_map:
                            del normalized_to_original_map[norm_to_remove]
                    
                    # Check 3: Add the new param if its normalized form isn't already a key.
                    # This handles both new additions and the 'turn_down_ratio' vs 'turndown_ratio' case.
                    if normalized_param not in normalized_to_original_map:
                         normalized_to_original_map[normalized_param] = param
                
                    # --- End New Logic ---

                logging.info(f"Found parameters for {vendor.vendor}")
            except Exception as e:
                logging.warning(f"Failed to get parameters for {vendor.vendor}: {e}")

        # Now, the set of *truly* unique parameters is the values of this map
        all_discovered_params = set(normalized_to_original_map.values())

        # Filter out parameters that already exist in the schema, with semantic/substring logic
        existing_parameters_normalized = {
            ep.lower().strip().replace('_', '') for ep in existing_parameters
        }
        
        filtered_parameters = set()

        for param in all_discovered_params:
            normalized_param = param.lower().strip().replace('_', '')
            
            # 1. Check for exact normalized match
            if normalized_param in existing_parameters_normalized:
                continue # Already exists, skip it

            # 2. Check for semantic duplication (substring)
            is_semantic_duplicate = False
            for existing_norm in existing_parameters_normalized:
                # If the new param is a substring of an existing param (e.g., new='responsetime', existing='totalresponsetime')
                # OR the existing param is a substring of the new param (e.g., new='totalresponsetime', existing='responsetime')
                if normalized_param in existing_norm or existing_norm in normalized_param:
                    is_semantic_duplicate = True
                    break # It's a semantic duplicate, skip it
            
            if not is_semantic_duplicate:
                filtered_parameters.add(param)

        result = {
            "product_type": product_type,
            "vendor_parameters": [vp.dict() for vp in all_vendor_parameters],
            "unique_parameters": list(filtered_parameters)[:20],
            "total_vendors_searched": len(all_vendor_parameters),
            "total_unique_parameters": len(filtered_parameters),
            "existing_parameters_filtered": len(all_discovered_params) - len(filtered_parameters)
        }
        
        logging.info(f"Discovery complete: {len(filtered_parameters)} new unique parameters found.")
        return result
        
    except Exception as e:
        logging.critical(f"Advanced parameters discovery failed entirely: {e}")
        # Final fallback if the whole process crashes
        discovery = AdvancedParametersDiscovery()
        fallback_params = discovery._get_llm_fallback_parameters("", product_type)
        return {
            "product_type": product_type,
            "vendor_parameters": [],
            "unique_parameters": fallback_params,
            "total_vendors_searched": 0,
            "total_unique_parameters": len(fallback_params),
            "fallback": True,
            "error": str(e)
        }

