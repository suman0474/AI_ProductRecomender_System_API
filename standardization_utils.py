# standardization_utils.py
# Utility functions for data standardization across vendors and model families

import json
import logging
import os
from typing import Dict, Any, List, Optional
from llm_standardization import standardize_with_llm

def standardize_vendor_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize vendor analysis result using LLM
    
    Args:
        analysis_result: Raw analysis result from vendor analysis
        
    Returns:
        Standardized analysis result with consistent naming
    """
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
    matched_models = analysis_result.get("matched_models", [])
    context = f"Product analysis with {len(matched_models)} matched models"
    if matched_models:
        context += f". Top model: {matched_models[0].get('model_series', 'Unknown')}"
    
    # Standardize using LLM
    try:
        standardized = standardize_with_llm(raw_data, context)
        
        # Update analysis result with standardized data
        result = analysis_result.copy()
        result.update({
            "vendor": standardized["vendor"],
            "product_type": standardized["product_type"],
            "model_family": standardized["model_family"],
            "specifications": standardized.get("specifications", raw_data["specifications"]),
            "standardization_confidence": standardized["confidence"],
            "standardization_method": "llm"
        })
        
        # Standardize matched models if they exist
        if "matched_models" in result:
            result["matched_models"] = standardize_matched_models(result["matched_models"])
            
        return result
        
    except Exception as e:
        logging.error(f"Failed to standardize vendor analysis result: {e}")
        return analysis_result

def standardize_ranking_result(ranking_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize ranking result using LLM
    
    Args:
        ranking_result: Raw ranking result from vendor ranking
        
    Returns:
        Standardized ranking result with consistent naming
    """
    if not ranking_result:
        return ranking_result
        
    result = ranking_result.copy()
    
    # Standardize each ranked vendor
    if "ranked_vendors" in result:
        standardized_vendors = []
        
        for vendor_data in result["ranked_vendors"]:
            raw_data = {
                "vendor": vendor_data.get("vendor", ""),
                "product_type": vendor_data.get("product_type", ""),
                "model_family": vendor_data.get("model_family", ""),
                "specifications": vendor_data.get("specifications", {})
            }
            
            context = f"Ranked vendor with score: {vendor_data.get('score', 'Unknown')}"
            
            try:
                standardized = standardize_with_llm(raw_data, context)
                
                # Update vendor data
                standardized_vendor = vendor_data.copy()
                standardized_vendor.update({
                    "vendor": standardized["vendor"],
                    "product_type": standardized["product_type"], 
                    "model_family": standardized["model_family"],
                    "specifications": standardized.get("specifications", raw_data["specifications"]),
                    "standardization_confidence": standardized["confidence"]
                })
                
                # Standardize models if they exist
                if "models" in standardized_vendor:
                    standardized_vendor["models"] = standardize_models_list(standardized_vendor["models"])
                    
                standardized_vendors.append(standardized_vendor)
                
            except Exception as e:
                logging.error(f"Failed to standardize vendor {vendor_data.get('vendor', 'Unknown')}: {e}")
                standardized_vendors.append(vendor_data)
        
        result["ranked_vendors"] = standardized_vendors
    
    return result

def enhance_submodel_mapping(submodel_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enhanced submodel to model series mapping using standardization
    
    Args:
        submodel_data: Raw submodel data
        
    Returns:
        Enhanced submodel data with standardized naming
    """
    if not submodel_data:
        return submodel_data
        
    raw_data = {
        "vendor": submodel_data.get("vendor", ""),
        "product_type": submodel_data.get("product_type", ""),
        "model_family": submodel_data.get("model_series", ""),
        "specifications": {}
    }
    
    # Extract specifications from sub_models
    if "sub_models" in submodel_data:
        for sub_model in submodel_data["sub_models"]:
            if "specifications" in sub_model:
                raw_data["specifications"].update(sub_model["specifications"])
    
    context = f"Submodel mapping for {submodel_data.get('model_series', 'Unknown')} series"
    
    try:
        standardized = standardize_with_llm(raw_data, context)
        
        result = submodel_data.copy()
        result.update({
            "vendor": standardized["vendor"],
            "product_type": standardized["product_type"],
            "model_series": standardized.get("model_family", submodel_data.get("model_series", "")),
            "standardization_confidence": standardized["confidence"]
        })
        
        # Standardize sub_models
        if "sub_models" in result:
            result["sub_models"] = standardize_submodels_list(result["sub_models"])
            
        return result
        
    except Exception as e:
        logging.error(f"Failed to enhance submodel mapping: {e}")
        return submodel_data

def standardize_product_image_mapping(image_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Standardize product image mapping data
    
    Args:
        image_data: Raw image mapping data
        
    Returns:
        Standardized image mapping data
    """
    if not image_data:
        return image_data
        
    raw_data = {
        "vendor": image_data.get("vendor", ""),
        "product_type": image_data.get("product_type", ""),
        "model_family": image_data.get("model_name", ""),
        "specifications": {}
    }
    
    context = f"Product image mapping for {image_data.get('model_name', 'Unknown')}"
    
    try:
        standardized = standardize_with_llm(raw_data, context)
        
        result = image_data.copy()
        result.update({
            "vendor": standardized["vendor"],
            "product_type": standardized["product_type"],
            "model_name": standardized.get("model_family", image_data.get("model_name", "")),
            "standardization_confidence": standardized["confidence"]
        })
        
        return result
        
    except Exception as e:
        logging.error(f"Failed to standardize product image mapping: {e}")
        return image_data

def create_standardization_report() -> Dict[str, Any]:
    """
    Create comprehensive standardization report
    
    Returns:
        Detailed report of standardization status across the system
    """
    report = {
        "timestamp": "",
        "summary": {
            "total_vendors": 0,
            "standardized_vendors": 0,
            "total_products": 0,
            "standardized_products": 0,
            "inconsistencies_found": 0
        },
        "vendor_analysis": {},
        "product_type_analysis": {},
        "model_family_analysis": {},
        "inconsistencies": [],
        "recommendations": []
    }
    
    try:
        from datetime import datetime
        report["timestamp"] = datetime.now().isoformat()
        
        # Analyze vendor directories
        vendors_dir = "vendors"
        documents_dir = "documents"
        
        if os.path.exists(vendors_dir):
            vendor_analysis = analyze_vendor_consistency(vendors_dir, documents_dir)
            report["vendor_analysis"] = vendor_analysis
            report["summary"]["total_vendors"] = vendor_analysis.get("total_vendors", 0)
            report["summary"]["standardized_vendors"] = vendor_analysis.get("standardized_vendors", 0)
        
        # Analyze product types
        if os.path.exists(vendors_dir):
            product_analysis = analyze_product_type_consistency(vendors_dir)
            report["product_type_analysis"] = product_analysis
            report["summary"]["total_products"] = product_analysis.get("total_products", 0)
            report["summary"]["standardized_products"] = product_analysis.get("standardized_products", 0)
        
        # Analyze model families
        if os.path.exists(vendors_dir):
            model_analysis = analyze_model_family_consistency(vendors_dir)
            report["model_family_analysis"] = model_analysis
        
        # Collect inconsistencies
        inconsistencies = collect_inconsistencies(vendors_dir, documents_dir)
        report["inconsistencies"] = inconsistencies
        report["summary"]["inconsistencies_found"] = len(inconsistencies)
        
        # Generate recommendations
        recommendations = generate_standardization_recommendations(report)
        report["recommendations"] = recommendations
        
        return report
        
    except Exception as e:
        logging.error(f"Failed to create standardization report: {e}")
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat() if 'datetime' in locals() else "",
            "summary": report["summary"]
        }

def analyze_vendor_consistency(vendors_dir: str, documents_dir: str) -> Dict[str, Any]:
    """Analyze vendor naming consistency between directories"""
    analysis = {
        "vendors_directory": [],
        "documents_directory": [],
        "consistent_vendors": [],
        "inconsistent_vendors": [],
        "total_vendors": 0,
        "standardized_vendors": 0
    }
    
    try:
        # Get vendor lists
        if os.path.exists(vendors_dir):
            analysis["vendors_directory"] = [d for d in os.listdir(vendors_dir) 
                                           if os.path.isdir(os.path.join(vendors_dir, d))]
        
        if os.path.exists(documents_dir):
            analysis["documents_directory"] = [d for d in os.listdir(documents_dir) 
                                             if os.path.isdir(os.path.join(documents_dir, d))]
        
        # Find consistent and inconsistent vendors
        vendors_set = set(analysis["vendors_directory"])
        documents_set = set(analysis["documents_directory"])
        
        # Check for exact matches (case-sensitive)
        for vendor in vendors_set:
            if vendor in documents_set:
                analysis["consistent_vendors"].append(vendor)
            else:
                # Check for case-insensitive matches
                matches = [d for d in documents_set if d.lower() == vendor.lower()]
                if matches:
                    analysis["inconsistent_vendors"].append({
                        "vendors_name": vendor,
                        "documents_name": matches[0],
                        "issue": "case_mismatch"
                    })
                else:
                    analysis["inconsistent_vendors"].append({
                        "vendors_name": vendor,
                        "documents_name": None,
                        "issue": "missing_in_documents"
                    })
        
        # Check for documents not in vendors
        for doc_vendor in documents_set:
            if doc_vendor not in vendors_set:
                case_matches = [v for v in vendors_set if v.lower() == doc_vendor.lower()]
                if not case_matches:
                    analysis["inconsistent_vendors"].append({
                        "vendors_name": None,
                        "documents_name": doc_vendor,
                        "issue": "missing_in_vendors"
                    })
        
        analysis["total_vendors"] = len(vendors_set | documents_set)
        analysis["standardized_vendors"] = len(analysis["consistent_vendors"])
        
    except Exception as e:
        logging.error(f"Failed to analyze vendor consistency: {e}")
        analysis["error"] = str(e)
    
    return analysis

def analyze_product_type_consistency(vendors_dir: str) -> Dict[str, Any]:
    """Analyze product type naming consistency using LLM-driven standardization"""
    analysis = {
        "product_types_found": {},
        "inconsistent_files": [],
        "total_products": 0,
        "standardized_products": 0,
        "llm_suggestions": {}
    }
    
    try:
        # First pass: collect all product types and get LLM suggestions
        all_product_types = set()
        
        for vendor_dir in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                    
                for json_file in os.listdir(product_path):
                    if not json_file.endswith('.json'):
                        continue
                        
                    file_path = os.path.join(product_path, json_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        product_type = data.get('product_type', '').strip()
                        if product_type:
                            all_product_types.add(product_type)
                            
                    except Exception as e:
                        logging.error(f"Failed to read {file_path}: {e}")
        
        # Get LLM suggestions for all unique product types
        for product_type in all_product_types:
            try:
                suggested = suggest_standard_product_type(product_type)
                analysis["llm_suggestions"][product_type] = suggested
            except Exception as e:
                logging.warning(f"LLM standardization failed for '{product_type}': {e}")
                analysis["llm_suggestions"][product_type] = product_type.lower()
        
        # Second pass: analyze consistency using LLM suggestions
        for vendor_dir in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                    
                for json_file in os.listdir(product_path):
                    if not json_file.endswith('.json'):
                        continue
                        
                    file_path = os.path.join(product_path, json_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        product_type = data.get('product_type', '').strip()
                        analysis["total_products"] += 1
                        
                        # Track all product types found
                        if product_type in analysis["product_types_found"]:
                            analysis["product_types_found"][product_type] += 1
                        else:
                            analysis["product_types_found"][product_type] = 1
                        
                        # Check consistency using LLM suggestions
                        suggested_type = analysis["llm_suggestions"].get(product_type, product_type)
                        
                        # Consider it standardized if the LLM suggests the same or very similar
                        if product_type.lower() == suggested_type.lower() or suggested_type.lower() in product_type.lower():
                            analysis["standardized_products"] += 1
                        else:
                            analysis["inconsistent_files"].append({
                                "file": file_path,
                                "current_type": product_type,
                                "suggested_type": suggested_type,
                                "directory_category": product_dir
                            })
                            
                    except Exception as e:
                        logging.error(f"Failed to read {file_path}: {e}")
                        
    except Exception as e:
        logging.error(f"Failed to analyze product types: {e}")
        analysis["error"] = str(e)
    
    return analysis

def analyze_model_family_consistency(vendors_dir: str) -> Dict[str, Any]:
    """Analyze model family naming consistency"""
    analysis = {
        "model_families": {},
        "duplicate_families": [],
        "total_families": 0
    }
    
    try:
        family_tracker = {}
        
        for vendor_dir in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                    
                for json_file in os.listdir(product_path):
                    if not json_file.endswith('.json'):
                        continue
                        
                    file_path = os.path.join(product_path, json_file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        models = data.get('models', [])
                        for model in models:
                            family = model.get('model_series', '').strip()
                            if family:
                                key = f"{vendor_dir}_{family}"
                                if key in family_tracker:
                                    family_tracker[key]["count"] += 1
                                else:
                                    family_tracker[key] = {
                                        "vendor": vendor_dir,
                                        "family": family,
                                        "product_type": product_dir,
                                        "count": 1,
                                        "files": []
                                    }
                                family_tracker[key]["files"].append(file_path)
                                
                    except Exception as e:
                        logging.error(f"Failed to read {file_path}: {e}")
        
        analysis["model_families"] = family_tracker
        analysis["total_families"] = len(family_tracker)
        
        # Find duplicates
        for key, data in family_tracker.items():
            if data["count"] > 1:
                analysis["duplicate_families"].append(data)
                
    except Exception as e:
        logging.error(f"Failed to analyze model families: {e}")
        analysis["error"] = str(e)
    
    return analysis

def collect_inconsistencies(vendors_dir: str, documents_dir: str) -> List[Dict[str, Any]]:
    """Collect all standardization inconsistencies"""
    inconsistencies = []
    
    # Vendor directory naming inconsistencies
    try:
        vendors_list = os.listdir(vendors_dir) if os.path.exists(vendors_dir) else []
        documents_list = os.listdir(documents_dir) if os.path.exists(documents_dir) else []
        
        # Check for case mismatches
        for vendor in vendors_list:
            for doc in documents_list:
                if vendor.lower() == doc.lower() and vendor != doc:
                    inconsistencies.append({
                        "type": "directory_case_mismatch",
                        "issue": f"Vendor directory case mismatch: '{vendor}' vs '{doc}'",
                        "severity": "medium",
                        "suggestion": f"Standardize to: {standardize_vendor_name(vendor)}"
                    })
                    
    except Exception as e:
        logging.error(f"Failed to collect directory inconsistencies: {e}")
    
    return inconsistencies

def generate_standardization_recommendations(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate actionable standardization recommendations"""
    recommendations = []
    
    # Vendor naming recommendations
    vendor_analysis = report.get("vendor_analysis", {})
    inconsistent_vendors = vendor_analysis.get("inconsistent_vendors", [])
    
    if inconsistent_vendors:
        recommendations.append({
            "priority": "high",
            "category": "vendor_naming",
            "title": "Standardize Vendor Directory Names",
            "description": f"Found {len(inconsistent_vendors)} vendor naming inconsistencies",
            "actions": [
                "Rename vendor directories to use proper capitalization",
                "Ensure consistency between vendors/ and documents/ directories",
                "Update file paths in code references"
            ]
        })
    
    # Product type recommendations
    product_analysis = report.get("product_type_analysis", {})
    inconsistent_files = product_analysis.get("inconsistent_files", [])
    
    if inconsistent_files:
        recommendations.append({
            "priority": "medium",
            "category": "product_types", 
            "title": "Standardize Product Type Names",
            "description": f"Found {len(inconsistent_files)} files with non-standard product types",
            "actions": [
                "Update JSON files to use lowercase product types",
                "Use standard terminology: 'pressure transmitter', 'temperature transmitter', etc.",
                "Run batch update script to fix all files"
            ]
        })
    
    # Model family recommendations
    model_analysis = report.get("model_family_analysis", {})
    duplicate_families = model_analysis.get("duplicate_families", [])
    
    if duplicate_families:
        recommendations.append({
            "priority": "low",
            "category": "model_families",
            "title": "Review Duplicate Model Families", 
            "description": f"Found {len(duplicate_families)} potentially duplicate model families",
            "actions": [
                "Review model families for actual duplicates vs legitimate variations",
                "Merge duplicate entries where appropriate",
                "Ensure consistent naming within each vendor"
            ]
        })
    
    return recommendations

# Helper functions

def standardize_matched_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize a list of matched models"""
    standardized = []
    for model in models:
        try:
            raw_data = {
                "vendor": model.get("vendor", ""),
                "product_type": model.get("product_type", ""),
                "model_family": model.get("model_series", ""),
                "specifications": model.get("specifications", {})
            }
            
            standardized_data = standardize_with_llm(raw_data, "Matched model standardization")
            
            standardized_model = model.copy()
            standardized_model.update({
                "vendor": standardized_data["vendor"],
                "product_type": standardized_data["product_type"],
                "model_series": standardized_data.get("model_family", model.get("model_series", ""))
            })
            
            standardized.append(standardized_model)
            
        except Exception as e:
            logging.error(f"Failed to standardize model: {e}")
            standardized.append(model)
    
    return standardized

def standardize_models_list(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize a list of models"""
    return standardize_matched_models(models)

def standardize_submodels_list(submodels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Standardize a list of submodels"""
    standardized = []
    for submodel in submodels:
        try:
            # Standardize specifications within submodel
            if "specifications" in submodel:
                standardized_specs = standardize_specifications(submodel["specifications"])
                submodel_copy = submodel.copy()
                submodel_copy["specifications"] = standardized_specs
                standardized.append(submodel_copy)
            else:
                standardized.append(submodel)
                
        except Exception as e:
            logging.error(f"Failed to standardize submodel: {e}")
            standardized.append(submodel)
    
    return standardized

def standardize_specifications(specs: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize specification keys and values"""
    if not specs:
        return specs
        
    try:
        raw_data = {
            "vendor": "",
            "product_type": "",
            "model_family": "",
            "specifications": specs
        }
        
        result = standardize_with_llm(raw_data, "Specification standardization")
        return result.get("specifications", specs)
        
    except Exception as e:
        logging.error(f"Failed to standardize specifications: {e}")
        return specs

def suggest_standard_product_type(product_type: str) -> str:
    """
    Smart product type standardization with LLM fallback to rule-based approach.
    Tries LLM first, falls back to intelligent rule-based standardization if LLM fails.
    """
    if not product_type:
        return product_type
    
    # First try LLM standardization (with timeout protection using threading)
    try:
        from llm_standardization import LLMStandardizer
        import threading
        import time
        
        # Try LLM with a simple timeout approach
        result_container = [None]
        error_container = [None]
        
        def llm_call():
            try:
                standardizer = LLMStandardizer()
                raw_data = {
                    "vendor": "",
                    "product_type": product_type,
                    "model_family": "",
                    "specifications": {}
                }
                result = standardizer.standardize_data(raw_data, "Quick product type standardization")
                result_container[0] = result
            except Exception as e:
                error_container[0] = e
        
        # Start LLM call in thread
        thread = threading.Thread(target=llm_call)
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout
        thread.join(timeout=3.0)  # 3 second timeout
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError("LLM call timed out")
        
        if error_container[0]:
            raise error_container[0]
        
        if result_container[0]:
            return result_container[0].product_type if result_container[0].product_type else product_type.lower().strip()
        
        raise Exception("No result from LLM")
        
    except Exception as e:
        # Fallback to intelligent rule-based standardization
        import logging
        logging.info(f"Using rule-based fallback for '{product_type}': LLM unavailable")
        
        return _rule_based_product_standardization(product_type)

def _rule_based_product_standardization(product_type: str) -> str:
    """
    Intelligent rule-based product type standardization as fallback.
    This provides good standardization without requiring external LLM calls.
    """
    if not product_type:
        return product_type
    
    # Clean the input
    cleaned = product_type.lower().strip()
    cleaned = " ".join(cleaned.split())  # Remove extra spaces
    
    # Smart pattern matching for common industrial instrument types
    
    # Pressure instruments
    if any(term in cleaned for term in ["pressure"]):
        if any(term in cleaned for term in ["relief", "safety", "regulator"]):
            return "pressure sensor"  # Valves are sensors in this context
        elif any(term in cleaned for term in ["sensor", "switch", "gauge"]):
            return "pressure sensor"
        else:
            return "pressure transmitter"  # Default for pressure measurement
    
    # Temperature instruments  
    elif any(term in cleaned for term in ["temperature", "temp", "thermocouple", "rtd", "thermistor", "thermal"]):
        return "temperature sensor"
    
    # Flow instruments
    elif any(term in cleaned for term in ["flow", "rotameter"]):
        return "flow meter"
    
    # Level instruments
    elif any(term in cleaned for term in ["level"]):
        if any(term in cleaned for term in ["switch", "sensor"]):
            return "level sensor"
        else:
            return "level transmitter"
    
    # Chemical/analytical instruments
    elif any(term in cleaned for term in ["ph", "orp", "redox"]):
        return "ph sensor"
    elif any(term in cleaned for term in ["conductivity", "conductance"]):
        return "conductivity sensor"
    elif any(term in cleaned for term in ["oxygen", "o2", "dissolved"]):
        return "oxygen sensor"
    elif any(term in cleaned for term in ["turbidity"]):
        return "turbidity sensor"
    elif any(term in cleaned for term in ["gas", "voc", "ammonia", "ozone", "h2s", "hydrogen"]):
        return "gas sensor"
    
    # Motion/position instruments
    elif any(term in cleaned for term in ["proximity"]):
        return "proximity sensor"
    elif any(term in cleaned for term in ["displacement", "lvdt", "position"]):
        return "displacement sensor"
    elif any(term in cleaned for term in ["encoder"]):
        return "encoder"
    elif any(term in cleaned for term in ["tachometer", "speed", "rpm"]):
        return "speed sensor"
    elif any(term in cleaned for term in ["strain", "stress"]):
        return "strain sensor"
    elif any(term in cleaned for term in ["load", "force", "weight"]):
        return "load sensor"
    
    # Control devices
    elif any(term in cleaned for term in ["valve"]):
        return "valve"
    elif any(term in cleaned for term in ["controller", "pid"]):
        return "controller"
    elif any(term in cleaned for term in ["relay"]):
        return "relay"
    elif any(term in cleaned for term in ["switch", "button"]):
        return "switch"
    
    # Optical/other sensors
    elif any(term in cleaned for term in ["photoelectric", "optical", "laser", "infrared"]):
        return "optical sensor"
    elif any(term in cleaned for term in ["vibration"]):
        return "vibration sensor"
    elif any(term in cleaned for term in ["humidity", "moisture"]):
        return "humidity sensor"
    
    # Generic fallback - try to extract key terms
    # Look for measurement type + device type pattern
    words = cleaned.split()
    
    # If we can identify a measurement, combine with appropriate device type
    for word in words:
        if word in ["pressure"]:
            return "pressure transmitter"
        elif word in ["temperature", "thermal"]:
            return "temperature sensor" 
        elif word in ["flow"]:
            return "flow meter"
        elif word in ["level"]:
            return "level transmitter"
    
    # Final fallback - return cleaned version
    return cleaned

def get_analysis_search_categories(detected_product_type: str) -> list:
    """
    Get the categories to search during analysis based on detected product type.
    
    Rules:
    - If sensor detected → search BOTH sensor AND transmitter folders (comprehensive search)
    - If transmitter detected → search ONLY transmitter folders (specific search)
    
    Args:
        detected_product_type: The product type detected during validation
        
    Returns:
        List of product categories to search during analysis
    """
    if not detected_product_type:
        return ["pressure transmitter"]  # fallback
    
    product_lower = detected_product_type.lower().strip()
    
    # Extract the measurement type and device type
    if "pressure" in product_lower:
        if "sensor" in product_lower:
            # Pressure sensor → search BOTH sensor and transmitter
            return ["pressure sensor", "pressure transmitter"]
        else:
            # Pressure transmitter → search ONLY transmitter
            return ["pressure transmitter"]
            
    elif any(word in product_lower for word in ["temperature", "temp"]):
        if "sensor" in product_lower:
            # Temperature sensor → search BOTH sensor and transmitter
            return ["temperature sensor", "temperature transmitter"]
        else:
            # Temperature transmitter → search ONLY transmitter
            return ["temperature transmitter"]
            
    elif "level" in product_lower:
        if "sensor" in product_lower:
            # Level sensor → search BOTH sensor and transmitter
            return ["level sensor", "level transmitter"]
        else:
            # Level transmitter → search ONLY transmitter
            return ["level transmitter"]
            
    elif "flow" in product_lower:
        # Flow is always meter regardless
        return ["flow meter"]
        
    elif any(word in product_lower for word in ["ph", "orp", "redox"]):
        # pH is typically always sensor
        return ["ph sensor"]
        
    elif "humidity" in product_lower:
        if "sensor" in product_lower:
            # Humidity sensor → search BOTH sensor and transmitter
            return ["humidity sensor", "humidity transmitter"]
        else:
            # Humidity transmitter → search ONLY transmitter
            return ["humidity transmitter"]
    
    # Default fallback
    return [detected_product_type]

def suggest_analysis_product_type(product_type: str) -> str:
    """
    DEPRECATED: Use get_analysis_search_categories() instead for smarter analysis.
    
    This function is kept for backward compatibility but the new approach 
    uses get_analysis_search_categories() which provides more intelligent
    search logic based on sensor vs transmitter detection.
    """
    # For backward compatibility, return the first category from the smart function
    search_categories = get_analysis_search_categories(product_type)
    return search_categories[0] if search_categories else product_type

def standardize_vendor_name(vendor: str) -> str:
    """Use LLM to standardize vendor name dynamically - no hardcoded logic"""
    if not vendor:
        return vendor
    
    try:
        # Use LLM for vendor standardization
        from llm_standardization import LLMStandardizer
        
        standardizer = LLMStandardizer()
        
        # Get actual vendor directories for context
        vendors_dir = "vendors"
        documents_dir = "documents"
        actual_vendors = set()
        
        # Collect existing vendor names as context for LLM
        if os.path.exists(vendors_dir):
            try:
                actual_vendors.update([d for d in os.listdir(vendors_dir) 
                                     if os.path.isdir(os.path.join(vendors_dir, d))])
            except Exception:
                pass
        
        if os.path.exists(documents_dir):
            try:
                actual_vendors.update([d for d in os.listdir(documents_dir) 
                                     if os.path.isdir(os.path.join(documents_dir, d))])
            except Exception:
                pass
        
        # Create context for LLM with existing vendors
        context = f"Vendor standardization. Existing vendors in system: {', '.join(sorted(actual_vendors)) if actual_vendors else 'None'}. Standardize vendor name to match existing or create proper format."
        
        # Create data structure for LLM
        raw_data = {
            "vendor": vendor,
            "product_type": "",
            "model_family": "",
            "specifications": {}
        }
        
        # Let LLM decide the standardized vendor name
        result = standardizer.standardize_data(raw_data, context)
        
        return result.vendor if result.vendor else vendor.title()
        
    except Exception as e:
        # Fallback to directory matching if LLM fails
        import logging
        logging.warning(f"LLM vendor standardization failed for '{vendor}': {e}")
        
        # Simple fallback - check existing directories first
        vendor_lower = vendor.lower().strip()
        
        # Get actual vendor directories
        vendors_dir = "vendors"
        actual_vendors = set()
        
        if os.path.exists(vendors_dir):
            try:
                actual_vendors.update([d for d in os.listdir(vendors_dir) 
                                     if os.path.isdir(os.path.join(vendors_dir, d))])
            except Exception:
                pass
        
        # Find exact match (case-insensitive)
        for actual_vendor in actual_vendors:
            if actual_vendor.lower() == vendor_lower:
                return actual_vendor
        
        # Find partial match
        for actual_vendor in actual_vendors:
            actual_lower = actual_vendor.lower()
            if vendor_lower in actual_lower or actual_lower in vendor_lower:
                return actual_vendor
        
        # Default to title case
        return vendor.title()

def update_existing_vendor_files_with_standardization():
    """Update existing vendor files with standardized naming"""
    try:
        vendors_dir = "vendors"
        if not os.path.exists(vendors_dir):
            logging.warning("Vendors directory not found")
            return
            
        updated_files = []
        
        for vendor_dir in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                    
                for json_file in os.listdir(product_path):
                    if not json_file.endswith('.json'):
                        continue
                        
                    file_path = os.path.join(product_path, json_file)
                    try:
                        # Read existing file
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Standardize the data
                        standardized_data = standardize_vendor_analysis_result(data)
                        
                        # Write back if changed
                        if standardized_data != data:
                            with open(file_path, 'w', encoding='utf-8') as f:
                                json.dump(standardized_data, f, indent=2, ensure_ascii=False)
                            updated_files.append(file_path)
                            
                    except Exception as e:
                        logging.error(f"Failed to update {file_path}: {e}")
        
        logging.info(f"Updated {len(updated_files)} vendor files with standardization")
        return updated_files
        
    except Exception as e:
        logging.error(f"Failed to update vendor files: {e}")
        return []

if __name__ == "__main__":
    # Test the standardization utilities
    print("🔧 Testing Standardization Utilities...")
    
    # Test vendor analysis result standardization
    test_analysis = {
        "vendor": "abb ltd",
        "product_type": "Pressure Measurement Device",
        "model_family": "266GST",
        "specifications": {
            "operating temp": "-40 to 85°C",
            "pressure range": "0-100 bar"
        },
        "matched_models": [
            {
                "vendor": "ABB",
                "model_series": "2600T Series",
                "specifications": {"accuracy": "0.2%"}
            }
        ]
    }
    
    standardized = standardize_vendor_analysis_result(test_analysis)
    print(f"Standardized analysis result: {standardized.get('vendor')} - {standardized.get('product_type')}")
    
    # Test standardization report
    print("\n📊 Generating standardization report...")
    report = create_standardization_report()
    print(f"Report summary: {report.get('summary', {})}")