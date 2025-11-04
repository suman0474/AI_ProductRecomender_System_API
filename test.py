import base64
from typing import IO, List, Dict, Any
from dotenv import load_dotenv
import os
import json
import re
import fitz  # PyMuPDF

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate

# Use LLM standardization instead of complex rule system
from llm_standardization import standardize_with_llm

print("1. Loading .env file...")
load_dotenv()


### NEW: Extract and LLM-identify product image ###
def identify_and_save_product_image(pdf_stream: IO[bytes], vendor_name: str, model_series: str) -> str:
    """
    Extracts all images from the entire PDF document, asks LLM which one is the product image,
    and saves only that identified image locally.
    """
    print("Attempting to identify product image using LLM...")
    try:
        vendor_folder = re.sub(r'[<>:\"/\\|?*]', '', vendor_name).strip()
        image_name = re.sub(r'[<>:\"/\\|?*]', '', model_series).strip() + ".png"

        save_dir = os.path.join("static", "images", vendor_folder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, image_name)

        pdf_stream.seek(0)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        if not doc.page_count:
            print("PDF has no pages, cannot extract image.")
            return None

        # Collect all candidate images from ALL pages
        image_candidates = []
        all_page_text = []
        
        print(f"Scanning {doc.page_count} pages for images...")
        for page_num in range(doc.page_count):
            page = doc[page_num]
            images = page.get_images(full=True)
            
            # Extract text from each page for context
            page_text = page.get_text("text") or ""
            all_page_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
            
            # Process images from this page
            for img in images:
                xref = img[0]
                # Check if we've already processed this image (images can be referenced multiple times)
                if any(candidate["xref"] == xref for candidate in image_candidates):
                    continue
                    
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                    image_candidates.append({
                        "xref": xref,
                        "width": img[2],
                        "height": img[3],
                        "page": page_num + 1,
                        "b64": image_b64
                    })
                except Exception as e:
                    print(f"Error extracting image xref {xref} from page {page_num + 1}: {e}")
                    continue

        if not image_candidates:
            print("No images found in the entire PDF document.")
            return None

        print(f"Found {len(image_candidates)} unique images across all pages.")

        # Extract text context from all pages
        full_document_text = "\n\n".join(all_page_text)

        # Send to LLM
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
        prompt = f"""
You are given a complete product datasheet PDF document.
There are multiple images across all pages and text content from the entire document.

Task: Identify which one is the MAIN PRODUCT IMAGE of the device/equipment described in the text.

Rules:
- Ignore logos, certification icons, graphs, diagrams, or technical drawings.
- Select the actual PRODUCT photo/rendering that shows the physical device.
- Consider images from all pages, not just the first page.
- The product image is usually a clear photo or 3D rendering of the actual device.
- Output ONLY JSON in the format: {{ "selected_index": <index_number> }}

Full Document Text Content:
{full_document_text}
        """

        # Attach image candidates to prompt
        messages = [HumanMessage(content=prompt)]
        for idx, img in enumerate(image_candidates):
            messages.append(
                HumanMessage(content=[
                    {"type": "text", "text": f"Candidate image {idx} (from page {img['page']}, size: {img['width']}x{img['height']})"},
                    {"type": "image_url", "image_url": f"data:image/png;base64,{img['b64']}"}
                ])
            )

        response = llm.invoke(messages)
        content = response.content.strip()

        # Clean JSON
        if content.startswith("```json"):
            content = content[7:].strip()
        elif content.startswith("```"):
            content = content[3:].strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        try:
            selection = json.loads(content)
            selected_index = selection.get("selected_index")
            if selected_index is None or not (0 <= selected_index < len(image_candidates)):
                print(f"LLM did not return a valid index. Got: {selected_index}, valid range: 0-{len(image_candidates)-1}")
                return None

            chosen_img = image_candidates[selected_index]
            img_bytes = base64.b64decode(chosen_img["b64"])
            with open(save_path, "wb") as img_file:
                img_file.write(img_bytes)

            print(f"✅ Product image identified and saved to {save_path}")
            print(f"   Selected image from page {chosen_img['page']} (size: {chosen_img['width']}x{chosen_img['height']})")
            return save_path
        except json.JSONDecodeError:
            print("LLM failed to return valid JSON. Output:", content)
            return None

    except Exception as e:
        print(f"Error identifying product image: {e}")
        return None

### Extract text and tables from PDF using PyMuPDF ###
def extract_data_from_pdf(pdf_stream: IO[bytes]) -> List[str]:
    """
    Extracts text from PDF pages and converts tables heuristically into key-value lines.
    Each chunk includes the page number for context.
    """
    print("2. Extracting text from PDF using PyMuPDF...")
    page_chunks = []
    try:
        pdf_stream.seek(0)
        doc = fitz.open(stream=pdf_stream, filetype="pdf")

        for page_number, page in enumerate(doc, start=1):
            # Extract raw text
            page_text = page.get_text("text") or ""

            # Extract table-like blocks heuristically
            table_lines = []
            blocks = page.get_text("blocks") or []
            for block in blocks:
                lines = block[4].splitlines()
                for line in lines:
                    if ":" in line:
                        table_lines.append(line.strip())

            # Combine text and table lines, and include page number
            combined_text = f"--- Page {page_number} ---\n{page_text}\n" + "\n".join(table_lines)
            combined_text = preprocess_specifications_text(combined_text)

            page_chunks.append(combined_text)

        print("2.1 PDF extraction into chunks successful.")
        return page_chunks

    except Exception as e:
        print(f"Error during PDF extraction: {e}")
        raise



def split_text(text: str, chunk_size: int = 3000) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def preprocess_specifications_text(text: str) -> str:
    """
    Convert lines like 'Key: Value' into 'Spec: key = value'
    """
    lines = text.splitlines()
    processed_lines = []
    for line in lines:
        match = re.match(r'^([\w \-/\(\)]+):\s*(.+)$', line.strip())
        if match:
            key = match.group(1).strip()
            val = match.group(2).strip()
            processed_lines.append(f"Spec: {key} = {val}")
        else:
            processed_lines.append(line)
    return "\n".join(processed_lines)


### Send chunks to the LLM for structured JSON extraction ###
def send_to_language_model(chunks: List[str]) -> List[Dict[str, Any]]:
    print("3. Sending concatenated text to the language model...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set your GOOGLE_API_KEY environment variable.")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

    prompt_template = PromptTemplate(
        input_variables=["full_text"],
        template="""
Extract structured technical data from the following text and return ONLY valid JSON according to this schema:
{{
  "product_type": "",
  "vendor": "",
  "models": [
    {{
      "model_series": "",
      "sub_models": [
        {{
          "name": "",
          "specifications": {{}}
        }}
      ]
    }}
  ]
}}

### Rules:
1. Output ONLY JSON. No explanations, comments, or extra text.
2. If any field is missing, use an empty string "" (do not omit fields).
3. Always include the keys exactly as shown in the schema.
4. Normalize specification keys:
   - lowercase
   - use underscores instead of spaces
5. Merge duplicate "model_series" entries into a single object with combined "sub_models".
6. Each sub-model includes:
   - "name": exact model name
   - "specifications": key-value pairs of all available specs
7. Flatten grouped specs into a single object.
8. If multiple model series exist, create separate JSON outputs for each model series.
9. For the product_type field: - If the product is a sub-category, return it under its parent category.
10. If a vendor is a sub-company or brand, return the vendor as the parent company.
11. Always include the original sub-model details even when changing the top-level product_type.
12. Do not include duplicate keys at different levels.
13. If no data is found for a field, leave it as "".
14. Return only valid JSON, with no extra characters or formatting.
15. Any "key: value" pair found in the text or tables MUST go into the "specifications" object of the corresponding sub_model.
16. If unsure which sub_model a specification belongs to, still include it under that sub_model's "specifications".


Text:
{full_text}
"""
    )

    full_text = "\n\n".join(chunks)
    prompt = prompt_template.format(full_text=full_text)

    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        data = json.loads(content)
        return data if isinstance(data, list) else [data]
    except json.JSONDecodeError:
        print("Warning: Could not decode JSON from LLM response.")
        print("LLM Output:", content[:500])
        return []



### Helpers for aggregation, normalization, and JSON saving ###
def normalize_series_name(series: str) -> str:
    if not series:
        return ""
    match = re.search(r'([a-z0-9]+)', series.lower())
    return match.group(1) if match else series.lower().replace(" ", "_")


def split_product_types(results: List[Dict]) -> List[Dict]:
    split_results = []
    for item in results:
        product_type = item.get("product_type", "")
        vendor = item.get("vendor", "")
        models = item.get("models", [])
        types = [t.strip() for t in re.split(r'/|,', product_type) if t.strip()] if "/" in product_type or "," in product_type else [product_type]
        for pt in types:
            split_results.append({"product_type": pt, "vendor": vendor, "models": models})
    return split_results


def aggregate_results(results: List[Dict], file_name: str = "") -> Dict:
    """
    Aggregate and normalize extracted LLM results into a structured JSON format.

    Args:
        results (List[Dict]): List of LLM-extracted results.
        file_name (str): Optional, used for reference.

    Returns:
        Dict: Normalized result containing product_type, vendor, and models.
    """
    print("5. Aggregating and cleaning results...")

    def is_meaningful_spec(specs: Dict[str, Any]) -> bool:
        """Check if the specification dictionary has any meaningful values."""
        return bool(specs and any(v and str(v).strip() != "" for v in specs.values()))

    normalized_models = {}
    vendor = ""
    product_type = ""

    for item in results:
        if isinstance(item, list):
            continue  # skip nested lists

        vendor = vendor or item.get("vendor", "").strip()
        product_type = product_type or item.get("product_type", "").strip()

        for model in item.get("models", []):
            series = model.get("model_series", "").strip()
            if not series:
                continue

            key = normalize_series_name(series)
            if key not in normalized_models:
                normalized_models[key] = {"model_series": series, "sub_models": []}

            for sub_model in model.get("sub_models", []):
                name = sub_model.get("name", "").strip()
                specs = sub_model.get("specifications") or {}
                # normalize keys
                specs = {k.lower().replace(" ", "_"): v for k, v in specs.items() if v}

                if not name and not is_meaningful_spec(specs):
                    continue

                # Merge with existing sub_model if present
                existing = next((sm for sm in normalized_models[key]["sub_models"] if sm.get("name") == name), None)
                if existing:
                    for k, v in specs.items():
                        if k not in existing["specifications"]:
                            existing["specifications"][k] = v
                else:
                    if not is_meaningful_spec(specs):
                        specs = {"_raw_text": "No structured specs extracted"}
                    normalized_models[key]["sub_models"].append({"name": name, "specifications": specs})

    # Filter out models with no meaningful sub-models
    filtered_models = []
    for model in normalized_models.values():
        meaningful_subs = [
            sm for sm in model["sub_models"]
            if sm.get("name") or is_meaningful_spec(sm.get("specifications", {}))
        ]
        if meaningful_subs:
            model["sub_models"] = meaningful_subs
            filtered_models.append(model)

    return {
        "product_type": product_type or "",
        "vendor": vendor or "",
        "models": filtered_models
    }

def generate_dynamic_path(final_result: Dict[str, Any]) -> str:
    print("6. Generating output filepath...")
    vendor_name = re.sub(r'[<>:"/\\|?*]', '', final_result.get("vendor") or "UnknownVendor")
    product_type = re.sub(r'[<>:"/\\|?*]', '', (final_result.get("product_type") or "UnknownProductType").lower())
    base_folder = "vendors"
    product_folder = os.path.join(base_folder, vendor_name, product_type)
    os.makedirs(product_folder, exist_ok=True)
    model_series_names = [m.get("model_series") for m in final_result.get("models", []) if m.get("model_series")]
    file_base_name = " ".join(re.sub(r'[<>:"/\\|?*]', '', name).strip() for name in model_series_names) or "extracted_data"
    return os.path.join(product_folder, f"{file_base_name}.json")


def save_json(final_result: Dict[str, Any], file_path: str):
    print(f"7. Saving JSON output to {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    print("7.1 JSON saved successfully.")


### Main function ###
def main(pdf_path: str):
    """
    Process a local PDF file: extract text and tables, generate structured JSON,
    and identify & save the product image using LLM.

    Args:
        pdf_path (str): Path to the local PDF file

    Returns:
        List[Dict]: List of processed product results
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Open PDF as binary stream
    with open(pdf_path, "rb") as file_stream:
        file_name = os.path.basename(pdf_path)

        # 1. Extract text chunks
        text_chunks = extract_data_from_pdf(file_stream)

        # 2. Generate structured JSON
        all_results = send_to_language_model(text_chunks)
        all_results = [item for r in all_results for item in (r if isinstance(r, list) else [r])]
        final_result = aggregate_results(all_results, file_name)
        split_results = split_product_types([final_result])

        # 3. Save JSON and identify product images
        for result in split_results:
            output_path = generate_dynamic_path(result)
            save_json(result, output_path)

            vendor = result.get("vendor")
            models = result.get("models", [])
            if vendor and models:
                model_series = models[0].get("model_series", "product")

                # Reset file pointer and extract product image
                file_stream.seek(0)
                identify_and_save_product_image(file_stream, vendor, model_series)

    return split_results
