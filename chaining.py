# chaining.py
# Contains LangChain components setup and analysis chain creation
import json
import logging
import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough
from models import  VendorAnalysis, OverallRanking, RequirementValidation
from prompts import validation_prompt, requirements_prompt, vendor_prompt, ranking_prompt, additional_requirements_prompt
from loading import load_requirements_schema, load_products_runnable
from dotenv import load_dotenv

# Import the OutputFixingParser
from langchain.output_parsers import OutputFixingParser
from langchain_google_genai import ChatGoogleGenerativeAI

# Import standardization utilities
from standardization_utils import standardize_ranking_result


# Load environment variables
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_langchain_components():
    # Use the old import since it's still functional (just shows deprecation warning)
    from langchain.callbacks import OpenAICallbackHandler
    callback_handler = OpenAICallbackHandler()
    
    # Create different models for different purposes
    # Gemini 2.5 Flash for simple conversations and text generation
    llm_flash = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Gemini 2.5 Pro for complex analysis tasks - uses second API key
    llm_pro = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.1, google_api_key=os.getenv("GOOGLE_API_KEY1"))
    
    memory = ConversationBufferWindowMemory(k=1, memory_key="chat_history", return_messages=True)
    
    # 1. Instantiate the base parsers
    validation_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    vendor_parser = JsonOutputParser(pydantic_object=VendorAnalysis)
    ranking_parser = JsonOutputParser(pydantic_object=OverallRanking)
    str_parser = StrOutputParser()
    
    # 2. Wrap the JSON parsers with OutputFixingParser for robustness
    # Use Flash model for validation and additional requirements (faster)
    validation_fixing_parser = OutputFixingParser.from_llm(parser=validation_parser, llm=llm_flash)
    # Use Pro model only for complex vendor analysis and ranking
    vendor_fixing_parser = OutputFixingParser.from_llm(parser=vendor_parser, llm=llm_pro)
    ranking_fixing_parser = OutputFixingParser.from_llm(parser=ranking_parser, llm=llm_pro)
    
    # 3. Use the appropriate models for different tasks
    # Fast tasks use Flash model
    validation_chain = validation_prompt | llm_flash | validation_fixing_parser
    requirements_chain = requirements_prompt | llm_flash | str_parser
    
    # Final analysis tasks use Pro model (vendor analysis and ranking)
    vendor_chain = vendor_prompt | llm_pro | vendor_fixing_parser
    ranking_chain = ranking_prompt | llm_pro | ranking_fixing_parser
    
    # --- NEW CHAIN FOR ADDITIONAL REQUIREMENTS ---
    additional_requirements_parser = JsonOutputParser(pydantic_object=RequirementValidation)
    additional_requirements_fixing_parser = OutputFixingParser.from_llm(parser=additional_requirements_parser, llm=llm_flash)
    additional_requirements_chain = additional_requirements_prompt | llm_flash | additional_requirements_fixing_parser

    # Get format instructions from the original parsers (the fixing parser wraps them)
    validation_format_instructions = validation_parser.get_format_instructions()
    vendor_format_instructions = vendor_parser.get_format_instructions()
    ranking_format_instructions = ranking_parser.get_format_instructions()
    additional_requirements_format_instructions = additional_requirements_parser.get_format_instructions()
    
    return {
        'llm': llm_flash,  # Default LLM for conversations (backward compatibility)
        'llm_flash': llm_flash,  # For conversations and simple text generation
        'llm_pro': llm_pro,  # For complex analysis tasks
        'memory': memory,
        'validation_chain': validation_chain,
        'requirements_chain': requirements_chain,
        'vendor_chain': vendor_chain,
        'ranking_chain': ranking_chain,
        # Add the new chain and its format instructions
        'additional_requirements_chain': additional_requirements_chain,
        'additional_requirements_format_instructions': additional_requirements_format_instructions,
        'validation_format_instructions': validation_format_instructions,
        'vendor_format_instructions': vendor_format_instructions,
        'ranking_format_instructions': ranking_format_instructions,
        'callback_handler': callback_handler
    }

def get_final_ranking(vendor_analysis_dict):
    """
    Processes a dictionary of vendor analysis to create a final ranked list.
    Takes a dictionary, not a Pydantic object.
    """
    products = []
    # Use .get() for safe access in case vendor_matches is missing
    if not vendor_analysis_dict or not vendor_analysis_dict.get('vendor_matches'):
        return {'ranked_products': []}
        
    for product in vendor_analysis_dict['vendor_matches']:
        product_score = product.get('match_score', 0)
        # Ensure requirements_match is a boolean. Default to False if missing.
        products.append({
            'product_name': product.get('product_name', ''),
            'vendor': product.get('vendor', ''),
            'match_score': product_score,
            'requirements_match': product.get('requirements_match', False),
            'reasoning': product.get('reasoning', ''),
            'limitations': product.get('limitations', '')
        })
    
    products_sorted = sorted(products, key=lambda x: x['match_score'], reverse=True)
    
    final_ranking = []
    rank = 1
    for product in products_sorted:
        final_ranking.append({
            'rank': rank,
            'product_name': product['product_name'],
            'vendor': product['vendor'],
            'overall_score': product['match_score'],
            'requirements_match': product['requirements_match'],
            'key_strengths': product['reasoning'],  # Use the same reasoning from vendor match
            'concerns': product['limitations']
        })
        rank += 1
        
    # Apply standardization to the ranking result
    ranking_result = {'ranked_products': final_ranking}
    standardized_ranking = standardize_ranking_result(ranking_result)
    return standardized_ranking

def to_dict_if_pydantic(obj):
    """Helper function to safely convert Pydantic object to dict."""
    if hasattr(obj, 'dict'):
        return obj.dict()
    return obj

# chaining.py
# ... (add this to your imports)
from loading import load_pdf_content_runnable

# ... (inside the file)

def create_analysis_chain(components, vendors_base_path):
    product_loader = load_products_runnable(vendors_base_path)
    # --- NEW: Instantiate the PDF content loader ---
    pdf_loader = load_pdf_content_runnable("documents")
    
    analysis_chain = (
        RunnablePassthrough.assign(
            structured_requirements=lambda x: components['requirements_chain'].invoke({"user_input": x["user_input"]})
        ).with_config(run_name="StructuredRequirementsGeneration")
        | RunnablePassthrough.assign(
            detected_product_type=lambda x: components['validation_chain'].invoke({
                "user_input": x["user_input"],
                "schema": json.dumps(load_requirements_schema(), indent=2),
                "format_instructions": components['validation_format_instructions']
            }).get('product_type', None)
        ).with_config(run_name="ProductTypeDetection")
        | product_loader.with_config(run_name="ProductDataLoading")
        # --- NEW: Add the PDF loading step to the chain ---
        | pdf_loader.with_config(run_name="LocalPDFLoading")
        | RunnablePassthrough.assign(
            # MODIFIED: Pass the new pdf_content_json to the vendor chain
            vendor_analysis=lambda x: components['vendor_chain'].invoke({
                "structured_requirements": x["structured_requirements"],
                "products_json": x["products_json"],
                "pdf_content_json": x["pdf_content_json"], # Pass the new PDF content
                "format_instructions": components['vendor_format_instructions']
            })
        ).with_config(run_name="VendorAnalysis")
        | RunnablePassthrough.assign(
            overall_ranking=lambda x: get_final_ranking(to_dict_if_pydantic(x["vendor_analysis"]))
        ).with_config(run_name="FinalRanking")
    )
    
    return analysis_chain
