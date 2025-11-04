

# pdf_utils.py
import fitz  # PyMuPDF
import logging
import os

# --- NEW: LLM Imports ---
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# --- NEW: Load environment variables and set up LLM ---
load_dotenv()
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GOOGLE_API_KEY")  # Use GOOGLE_API_KEY1 for analysis
    )
except ImportError:
    llm = None
    logging.warning("Could not import Google Generative AI. LLM functionality in pdf_utils is disabled.")


# --- NEW: Prompt and Chain for identifying the selection guide ---
guide_identification_prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst specializing in technical datasheets.
Your task is to determine if the following text from a PDF page contains a "Model Selection Guide",
"Ordering Information" table, or any table used to construct a specific product model number by 
selecting different options (like pressure range, output, materials, etc.).

The answer must be a single word: YES or NO. Do not provide any explanation or other text.

Page Text:
---
{page_text}
---
""")

if llm:
    guide_identification_chain = guide_identification_prompt | llm | StrOutputParser()
else:
    guide_identification_chain = None


def extract_model_selection_guide(pdf_path_or_bytes):
    """
    Extracts:
    1. All Model Selection Guide tables (mandatory parameters)
    2. Remaining PDF text (optional specifications)
    """
    if not guide_identification_chain:
        logging.error("LLM chain for guide identification is not available. Cannot perform extraction.")
        return {"model_selection_guides": [], "optional_specs": ""}

    model_selection_guides = []
    optional_texts = []

    doc = None
    try:
        if isinstance(pdf_path_or_bytes, str):
            doc = fitz.open(pdf_path_or_bytes)
        else:
            doc = fitz.open(stream=pdf_path_or_bytes, filetype="pdf")

        for page_num, page in enumerate(doc):
            logging.info(f"Analyzing page {page_num + 1} with LLM...")
            page_text = page.get_text("text")

            if len(page_text.strip()) < 100:
                continue

            # Check if this page has a Model Selection Guide
            llm_response = guide_identification_chain.invoke({"page_text": page_text})

            if "YES" in llm_response.upper():
                logging.info(f"LLM identified a Model Selection Guide on page {page_num + 1}.")
                # Extract tables on the identified page
                tables = page.find_tables()
                extracted_any = False
                if tables:
                    for table in tables:
                        table_data = table.extract()
                        if table_data and len(table_data) > 1:
                            headers = [str(h).replace('\n', ' ') for h in table_data[0]]
                            models = []
                            for row in table_data[1:]:
                                if len(row) == len(headers):
                                    model_info = {headers[i]: row[i] for i in range(len(headers))}
                                    models.append({
                                        "model_code": row[0],  # first column assumed as model identifier
                                        "parameters": model_info
                                    })
                            if models:
                                model_selection_guides.append({
                                    "page": page_num + 1,
                                    "headers": headers,
                                    "models": models
                                })
                                extracted_any = True

                # Heuristic: scan subsequent pages for continuation of the guide (tables or 'continued' markers)
                # Continue scanning forward until a page doesn't look like a continuation or the document ends.
                offset = 1
                while True:
                    next_idx = page_num + offset
                    if next_idx >= len(doc):
                        break
                    next_page = doc[next_idx]
                    next_text = next_page.get_text("text")
                    if not next_text or len(next_text.strip()) < 30:
                        # Likely a blank or irrelevant page
                        break

                    next_tables = next_page.find_tables()
                    added_next = False

                    # If tables exist on the following page, extract similarly and include
                    if next_tables:
                        for table in next_tables:
                            table_data = table.extract()
                            if table_data and len(table_data) > 1:
                                headers = [str(h).replace('\n', ' ') for h in table_data[0]]
                                models = []
                                for row in table_data[1:]:
                                    if len(row) == len(headers):
                                        model_info = {headers[i]: row[i] for i in range(len(headers))}
                                        models.append({
                                            "model_code": row[0],
                                            "parameters": model_info
                                        })
                                if models:
                                    model_selection_guides.append({
                                        "page": next_idx + 1,
                                        "headers": headers,
                                        "models": models
                                    })
                                    added_next = True

                    # If no tables, look for 'continued' or 'table continued' clues in text
                    if not added_next:
                        if re.search(r'\bcontinued\b|\bcont\.|table continued|continued from previous', next_text, flags=re.I):
                            # Treat this page as part of the guide (may contain continued parameters in text)
                            model_selection_guides.append({
                                "page": next_idx + 1,
                                "headers": [],
                                "models": []
                            })
                            added_next = True

                    # If the page didn't look like a continuation, stop scanning
                    if not added_next:
                        break
                    offset += 1
            else:
                # Collect text for optional parameters
                if page_text.strip():
                    optional_texts.append(page_text)

    except Exception as e:
        logging.error(f"Failed to extract sections from PDF: {e}")
    finally:
        if doc:
            doc.close()

    return {
        "model_selection_guides": model_selection_guides,
        "optional_specs": "\n\n".join(optional_texts)
    }
