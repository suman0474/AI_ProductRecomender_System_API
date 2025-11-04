import concurrent.futures
import logging
import pandas as pd
import os
import glob
import io
import json
import threading
import time
from typing import List, Dict, Any

# Import all necessary functions from loading.py and test.py
from loading import (
    discover_top_vendors, 
    create_schema_from_vendor_data,
    _save_schema_to_specs,
    _search_vendor_pdfs,
    ProgressTracker
)
from test import (
    extract_data_from_pdf,
    identify_and_save_product_image,
    send_to_language_model,
    aggregate_results,
    split_product_types,
    save_json
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global semaphore to control concurrent LLM calls (max 2 simultaneous)
llm_semaphore = threading.Semaphore(2)

def build_schema_parallel(product_types):
    """
    Build requirements schema for multiple product types in parallel using ThreadPoolExecutor.
    
    Args:
        product_types (list): List of product types to process.
        
    Returns:
        dict: Dictionary mapping product types to their schemas.
    """
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create a dictionary mapping futures to their product types
        future_to_product = {
            executor.submit(build_complete_catalog, product_type): product_type 
            for product_type in product_types
        }
        
        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_product):
            product_type = future_to_product[future]
            try:
                result = future.result()
                results[product_type] = result
                logger.info(f"Successfully completed catalog building for {product_type}")
            except Exception as e:
                logger.error(f"Error building catalog for {product_type}: {str(e)}")
                results[product_type] = None
    
    return results

def process_vendor_parallel(vendor_info, product_type):
    """
    Process a single vendor in parallel: search PDFs for the vendor.
    
    Args:
        vendor_info (dict): Vendor information with name and model families
        product_type (str): Product type being processed
        
    Returns:
        dict: Vendor data with models/PDFs found
    """
    vendor_name = vendor_info.get("vendor")
    model_families = vendor_info.get("model_families", [])
    
    try:
        logger.info(f"[PARALLEL] Processing vendor: {vendor_name} for {product_type}")
        models = _search_vendor_pdfs(vendor_name, product_type, model_families)
        return {
            "vendor": vendor_name,
            "models": models,
            "success": True
        }
    except Exception as e:
        logger.error(f"[PARALLEL] Failed to process vendor {vendor_name}: {e}")
        return {
            "vendor": vendor_name,
            "models": [],
            "success": False,
            "error": str(e)
        }

def process_vendor_complete_pipeline(vendor_data, product_type):
    """
    Complete vendor processing pipeline in parallel:
    1. Take vendor data with found PDFs
    2. Download and process all PDFs for this vendor
    3. Extract structured data using LLM
    4. Save JSONs and images
    
    Args:
        vendor_data (dict): Vendor data with models and PDFs
        product_type (str): Product type being processed
        
    Returns:
        dict: Processing results for the vendor
    """
    vendor_name = vendor_data.get("vendor", "unknown")
    models = vendor_data.get("models", [])
    
    if not vendor_data.get("success", False) or not models:
        logger.warning(f"[PARALLEL] Skipping vendor {vendor_name} - no successful PDF search results")
        return {
            "vendor": vendor_name,
            "status": "skipped",
            "reason": "no_pdfs_found",
            "results_generated": 0
        }
    
    logger.info(f"[PARALLEL] Starting complete pipeline for vendor: {vendor_name}")
    vendor_start_time = time.time()
    
    try:
        # Process PDFs for this vendor using existing function but wrapped for parallel execution
        base_dir = "documents"
        os.makedirs(base_dir, exist_ok=True)
        
        vendor_results = []
        processing_stats = {
            "total_pdfs_attempted": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "processing_errors": 0
        }
        
        # Process each model family for this vendor
        for model in models:
            if not isinstance(model, dict):
                continue
                
            model_family = (model.get("model_family") or "").strip().replace(" ", " ")
            pdfs = model.get("pdfs", [])
            logger.info(f"[PARALLEL] Vendor {vendor_name}: Processing model '{model_family}' with {len(pdfs)} PDFs")
            
            # Process top 2 PDFs per model to avoid overwhelming
            for pdf_info in pdfs[:2]:
                pdf_url = pdf_info.get("pdf_url")
                if not pdf_url:
                    continue
                    
                processing_stats["total_pdfs_attempted"] += 1
                
                try:
                    # Create vendor directory structure
                    vendor_dir = os.path.join(base_dir, vendor_name.replace(" ", " "))
                    product_dir = os.path.join(vendor_dir, product_type.replace(" ", "_"))
                    os.makedirs(product_dir, exist_ok=True)
                    
                    # Download PDF with unique filename to avoid conflicts
                    import re
                    filename = os.path.basename(pdf_url.split("?")[0]) or f"{vendor_name}_{model_family}_{int(time.time())}.pdf"
                    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
                    file_path = os.path.join(product_dir, filename)
                    
                    # Download PDF (reuse existing download logic)
                    from loading import _download_pdf_with_retry
                    download_success, download_message = _download_pdf_with_retry(pdf_url, file_path)
                    
                    if not download_success:
                        logger.warning(f"[PARALLEL] {vendor_name}: Failed to download {pdf_url}: {download_message}")
                        processing_stats["failed_downloads"] += 1
                        continue
                    
                    processing_stats["successful_downloads"] += 1
                    logger.info(f"[PARALLEL] {vendor_name}: Downloaded PDF to {file_path}")
                    
                    # Extract and process PDF content with LLM semaphore protection
                    try:
                        with open(file_path, "rb") as pdf_file:
                            pdf_bytes = io.BytesIO(pdf_file.read())
                            
                            # Extract text
                            text_chunks = extract_data_from_pdf(pdf_bytes)
                            if not text_chunks or all(len(chunk.strip()) < 50 for chunk in text_chunks):
                                logger.warning(f"[PARALLEL] {vendor_name}: PDF has minimal text content: {file_path}")
                                continue
                            
                            # Use LLM with semaphore to prevent rate limiting
                            with llm_semaphore:
                                logger.info(f"[PARALLEL] {vendor_name}: Acquiring LLM semaphore for text processing")
                                pdf_results = send_to_language_model(text_chunks)
                                pdf_results = [item for r in pdf_results for item in (r if isinstance(r, list) else [r])]
                                vendor_results.extend(pdf_results)
                                logger.info(f"[PARALLEL] {vendor_name}: Released LLM semaphore, extracted {len(pdf_results)} results")
                            
                            # Extract product image
                            pdf_bytes.seek(0)
                            if vendor_name and model_family:
                                identify_and_save_product_image(pdf_bytes, vendor_name, model_family)
                                
                    except Exception as e:
                        logger.error(f"[PARALLEL] {vendor_name}: Failed to process PDF content {file_path}: {e}")
                        processing_stats["processing_errors"] += 1
                        continue
                        
                except Exception as e:
                    logger.error(f"[PARALLEL] {vendor_name}: Failed to process PDF {pdf_url}: {e}")
                    processing_stats["failed_downloads"] += 1
        
        # Aggregate and save results for this vendor
        final_results = []
        if vendor_results:
            logger.info(f"[PARALLEL] {vendor_name}: Aggregating {len(vendor_results)} results")
            final_result = aggregate_results(vendor_results, product_type)
            split_results = split_product_types([final_result])
            
            # Save per-family JSONs for this vendor
            for result in split_results:
                vendor_from_result = (result.get("vendor") or "").strip()
                for model in result.get("models", []):
                    if not model.get("model_series"):
                        continue
                    
                    single_family_payload = {
                        "product_type": result.get("product_type", product_type),
                        "vendor": vendor_from_result,
                        "models": [model],
                    }
                    
                    safe_vendor = vendor_from_result.replace(" ", " ") or "UnknownVendor"
                    safe_ptype = (result.get("product_type") or product_type).replace(" ", " ") or "UnknownProduct"
                    safe_series = (model.get("model_series") or "unknown_model").replace("/", " ").replace("\\", " ")
                    
                    vendor_dir = os.path.join("vendors", safe_vendor, safe_ptype)
                    os.makedirs(vendor_dir, exist_ok=True)
                    file_path = os.path.join(vendor_dir, f"{safe_series}.json")
                    save_json(single_family_payload, file_path)
                    logger.info(f"[PARALLEL] {vendor_name}: Saved JSON: {file_path}")
                    
                    final_results.append(single_family_payload)
        
        processing_time = time.time() - vendor_start_time
        logger.info(f"[PARALLEL] ✅ {vendor_name}: Complete pipeline finished in {processing_time:.1f}s")
        
        return {
            "vendor": vendor_name,
            "status": "success",
            "results_generated": len(final_results),
            "processing_time": processing_time,
            "processing_stats": processing_stats
        }
        
    except Exception as e:
        processing_time = time.time() - vendor_start_time
        logger.error(f"[PARALLEL] ❌ {vendor_name}: Complete pipeline failed after {processing_time:.1f}s: {e}")
        return {
            "vendor": vendor_name,
            "status": "failed",
            "reason": str(e),
            "processing_time": processing_time,
            "results_generated": 0
        }

def build_complete_catalog(product_type: str) -> Dict[str, Any]:
    """
    Complete catalog building process for a single product type:
    1. Discover top 5 vendors and model families
    2. Generate schema
    3. Search for PDFs
    4. Download and process documents
    5. Save JSONs and images
    """
    logger.info(f"[CATALOG] Starting complete catalog building for '{product_type}'")
    
    progress = ProgressTracker(6, f"Catalog Building for {product_type}")
    
    try:
        # Step 1: Discover vendors
        progress.update("Discovering vendors", f"Finding top 5 vendors for {product_type}")
        vendors_with_families = discover_top_vendors(product_type)
        if not vendors_with_families:
            logger.warning(f"No vendors discovered for {product_type}")
            return {"product_type": product_type, "status": "failed", "reason": "no_vendors_found"}
        
        # Step 2: Generate schema
        progress.update("Generating schema", f"Creating schema from {len(vendors_with_families)} vendors")
        try:
            schema = create_schema_from_vendor_data(product_type, vendors_with_families)
            schema_path = _save_schema_to_specs(product_type, schema)
            logger.info(f"Schema saved to: {schema_path}")
        except Exception as e:
            logger.warning(f"Schema generation failed for {product_type}: {e}")
        
        # Step 3: Search for PDFs in parallel (5 vendors at a time)
        progress.update("Searching PDFs", "Finding PDF documents via search engines (parallel vendor processing)")
        vendors_data = []
        
        # Process vendors in parallel using ThreadPoolExecutor
        max_vendor_workers = min(len(vendors_with_families), 5)  # Maximum 5 vendors at a time
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_vendor_workers) as vendor_executor:
            # Submit all vendor search tasks
            future_to_vendor = {
                vendor_executor.submit(process_vendor_parallel, vendor_info, product_type): vendor_info.get("vendor", "unknown")
                for vendor_info in vendors_with_families
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_vendor):
                vendor_name = future_to_vendor[future]
                try:
                    result = future.result(timeout=300)  # 5 minutes max per vendor
                    vendors_data.append(result)
                    
                    if result["success"]:
                        logger.info(f"✅ Successfully processed vendor: {vendor_name} ({len(result['models'])} models found)")
                    else:
                        logger.warning(f"⚠️ Failed to process vendor: {vendor_name}")
                        
                except Exception as e:
                    logger.error(f"❌ Vendor {vendor_name} processing failed with exception: {e}")
                    vendors_data.append({
                        "vendor": vendor_name,
                        "models": [],
                        "success": False,
                        "error": str(e)
                    })
        
        # Filter successful vendors for PDF processing
        successful_vendors = [v for v in vendors_data if v["success"]]
        failed_vendors = [v for v in vendors_data if not v["success"]]
        
        if failed_vendors:
            logger.warning(f"Failed to process {len(failed_vendors)} vendors: {[v['vendor'] for v in failed_vendors]}")
        
        logger.info(f"Successfully processed {len(successful_vendors)}/{len(vendors_with_families)} vendors in parallel")
        
        # Step 4: Process complete vendor pipelines in parallel (NEW: Full parallel extraction)
        progress.update("Processing vendor pipelines", "Running complete extraction pipeline for all vendors in parallel")
        
        if not successful_vendors:
            logger.warning(f"No successful vendors to process for {product_type}")
            return {"product_type": product_type, "status": "failed", "reason": "no_successful_vendors"}
        
        # Run complete vendor processing in parallel (5 vendors simultaneously)
        vendor_extraction_results = []
        max_extraction_workers = min(len(successful_vendors), 5)  # Keep at 5 as requested
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_extraction_workers) as extraction_executor:
            # Submit complete pipeline tasks for each vendor
            future_to_vendor = {
                extraction_executor.submit(process_vendor_complete_pipeline, vendor_data, product_type): vendor_data.get("vendor", "unknown")
                for vendor_data in successful_vendors
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_vendor):
                vendor_name = future_to_vendor[future]
                try:
                    result = future.result(timeout=600)  # 10 minutes max per vendor extraction
                    vendor_extraction_results.append(result)
                    
                    if result.get("status") == "success":
                        logger.info(f"✅ {vendor_name}: Complete pipeline successful ({result.get('results_generated', 0)} results in {result.get('processing_time', 0):.1f}s)")
                    else:
                        logger.warning(f"⚠️ {vendor_name}: Complete pipeline failed - {result.get('reason', 'unknown')}")
                        
                except Exception as e:
                    logger.error(f"❌ {vendor_name}: Complete pipeline exception: {e}")
                    vendor_extraction_results.append({
                        "vendor": vendor_name,
                        "status": "failed",
                        "reason": str(e),
                        "results_generated": 0
                    })
        
        # Collect final statistics
        successful_extractions = [r for r in vendor_extraction_results if r.get("status") == "success"]
        failed_extractions = [r for r in vendor_extraction_results if r.get("status") != "success"]
        total_results_generated = sum(r.get("results_generated", 0) for r in successful_extractions)
        
        logger.info(f"Vendor extraction summary: {len(successful_extractions)} successful, {len(failed_extractions)} failed")
        
        # Step 5: Final completion
        progress.update("Completion", f"Successfully processed {len(successful_extractions)} vendors with {total_results_generated} total results")
        
        return {
            "product_type": product_type,
            "status": "success",
            "vendors_discovered": len(vendors_with_families),
            "vendors_processed_successfully": len(successful_vendors),
            "vendors_extraction_successful": len(successful_extractions),
            "vendors_failed": len(failed_vendors),
            "vendors_extraction_failed": len(failed_extractions),
            "results_generated": total_results_generated,
            "processing_time": progress.get_progress()["elapsed_time"],
            "parallel_vendor_processing": True,
            "parallel_vendor_extraction": True
        }
        
    except Exception as e:
        logger.error(f"Complete catalog building failed for {product_type}: {e}")
        return {"product_type": product_type, "status": "failed", "reason": str(e)}

def main():
    # Read product types from Sheet 3
    products_df = pd.read_excel("./documents/Industrial_Instruments_Specs.xlsx", sheet_name="Sheet3")
    
    # Get product types from the first column (which appears to be 'Pressure Transmitters')
    product_types = products_df.iloc[:, 0].dropna().unique().tolist()
    
    logger.info(f"Found {len(product_types)} product types in Sheet 3: {product_types}")

    # Check which product types already have JSON files in specs folder
    folder_path = "./specs/"
    file_pattern = "*.json"

    # Get a list of existing product_type files
    existing_files = glob.glob(os.path.join(folder_path, file_pattern))
    existing_product_types = [os.path.splitext(os.path.basename(file))[0] for file in existing_files]

    # Normalize product types for comparison (handle spaces and case differences)
    normalized_existing = [ptype.lower().replace(" ", " ").replace("_", " ") for ptype in existing_product_types]
    
    # Filter out product types that already have corresponding files
    missing_product_types = []
    for ptype in product_types:
        normalized_ptype = ptype.lower().replace(" ", " ").replace("_", " ")
        if normalized_ptype not in normalized_existing:
            missing_product_types.append(ptype)
    
    logger.info(f"Found {len(existing_product_types)} existing schemas in specs folder")
    logger.info(f"Missing schemas for {len(missing_product_types)} product types: {missing_product_types}")
    
    if not missing_product_types:
        logger.info("All product types already have schemas. No processing needed.")
        return
    
    logger.info("Starting parallel catalog building process...")
    results = build_schema_parallel(missing_product_types)
    
    # Process and display results
    successful_builds = 0
    failed_builds = 0
    
    for product_type, result in results.items():
        if result and result.get("status") == "success":
            successful_builds += 1
            logger.info(f"✅ Successfully completed catalog for {product_type}:")
            logger.info(f"   - Vendors discovered: {result.get('vendors_discovered', 0)}")
            logger.info(f"   - Vendors processed successfully: {result.get('vendors_processed_successfully', 0)}")
            logger.info(f"   - Vendors extraction successful: {result.get('vendors_extraction_successful', 0)}")
            logger.info(f"   - Vendors failed: {result.get('vendors_failed', 0)}")
            logger.info(f"   - Results generated: {result.get('results_generated', 0)}")
            logger.info(f"   - Processing time: {result.get('processing_time', 0):.1f}s")
            logger.info(f"   - Parallel vendor processing: {result.get('parallel_vendor_processing', False)}")
            logger.info(f"   - Parallel vendor extraction: {result.get('parallel_vendor_extraction', False)}")
        else:
            failed_builds += 1
            reason = result.get("reason", "unknown error") if result else "unknown error"
            logger.error(f"❌ Failed to build catalog for {product_type}: {reason}")
    
    logger.info(f"\n📊 FINAL SUMMARY:")
    logger.info(f"   - Total product types processed: {len(missing_product_types)}")
    logger.info(f"   - Successful builds: {successful_builds}")
    logger.info(f"   - Failed builds: {failed_builds}")
    logger.info(f"   - Success rate: {(successful_builds/len(missing_product_types)*100):.1f}%" if missing_product_types else "N/A")

if __name__ == "__main__":
    main()