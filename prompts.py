

# prompts.py
# Contains all prompt templates for LangChain
from langchain_core.prompts import ChatPromptTemplate

validation_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for industrial requisitioners and buyers. Your job is to validate technical product requirements in a way that helps procurement professionals make informed decisions.

User Input:
{user_input}

Requirements Schema:
{schema}

Tasks:
1. Intelligently identify the CORE PRODUCT CATEGORY from user input
2. Extract the requirements that were provided, focusing on what matters to buyers

CRITICAL: Dynamic Product Type Intelligence:
Your job is to determine the most appropriate and standardized product category based on the user's input. Use your knowledge of industrial instruments and measurement devices to:

1. **Identify the core measurement function** - What is being measured? (pressure, temperature, flow, level, pH, etc.)
2. **Determine the appropriate device type** - What type of instrument is needed? (sensor, transmitter, meter, gauge, controller, valve, etc.)
3. **Remove technology-specific modifiers** - Focus on function over implementation (remove terms like "differential", "vortex", "radar", "smart", etc.)
4. **Standardize terminology** - Use consistent, industry-standard naming conventions

EXAMPLES (learn the pattern, don't memorize):
- "differential pressure transmitter" → analyze: measures pressure + transmits signal → "pressure transmitter"
- "vortex flow meter" → analyze: measures flow + meter device → "flow meter"
- "RTD temperature sensor" → analyze: measures temperature + sensing function → "temperature sensor"
- "smart level indicator" → analyze: measures level + indicates/transmits → "level transmitter"
- "pH electrode" → analyze: measures pH + sensing function → "ph sensor"

YOUR APPROACH:
1. Analyze what physical parameter is being measured
2. Determine what type of industrial device is most appropriate
3. Use standard industrial terminology
4. Focus on procurement-relevant categories that buyers understand
5. Be consistent - similar requests should get similar categorizations

Remember: The goal is to create logical, searchable categories that help procurement teams find the right products efficiently. Use your expertise to make intelligent decisions about standardization.

{format_instructions}
""")

requirements_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for industrial requisitioners and buyers. Extract and structure the key requirements from this user input so a procurement professional can quickly understand what is needed and why.


User Input:
{user_input}


Focus on:
- Technical specifications (pressure ranges, accuracy, etc.)
- Connection types and standards
- Application context and environment
- Performance requirements
- Compliance or certification needs
- Any business or operational considerations relevant to buyers

Return a clear, structured summary of requirements, using language that is actionable and easy for buyers to use in procurement.Only include sections and details for which information is explicitly present in the user's input. Do not add any inferred requirements or placeholders for missing information
""")





vendor_prompt = ChatPromptTemplate.from_template("""
You are a meticulous procurement and technical matching expert.  
Your task is to analyze user requirements against vendor product documentation (PDF datasheets and/or JSON product summaries) and identify the single best-fitting model for each product series.  

Follow these instructions carefully:

## 1. Matching System

**Tier 1: Mandatory & Optional Specifications (From PDF)**
This is the primary tier and relies on the presence of a PDF document. The classification of a spec as CRITICAL or OPTIONAL depends on whether a "Model Selection Guide" is found within that PDF.

*Scenario A: Model Selection Guide EXISTS*
- Source: The main PDF document
- Logic:
  - Any specification listed within the Model Selection Guide is considered CRITICAL
  - All other specifications found in the rest of the PDF are considered OPTIONAL/SECONDARY
- Analysis Output: Parameter-by-parameter breakdown including parameter name, user requirement, product specification, status (CRITICAL or OPTIONAL), and holistic explanation for the match

*Scenario B: Model Selection Guide is MISSING*
- Source: The main PDF document
- Logic: The system must intelligently identify which specifications are mandatory/critical versus optional based on document context and content analysis
  - Analyze the document structure, headings, and content to determine which specifications are fundamental/core to the product
  - Specifications that are essential for basic product operation, safety, or core functionality are considered CRITICAL
  - Specifications that enhance performance, provide additional features, or offer customization options are considered OPTIONAL/SECONDARY
  - Look for contextual clues such as: required vs available options, standard vs optional features, basic vs advanced specifications
- Analysis Output: Parameter-by-parameter breakdown with reasoning for why each specification is classified as CRITICAL or OPTIONAL based on document analysis

**Tier 2: Optional Specifications (Other Document Specs)**
This tier is only applicable under Tier 1's "Scenario A" (when a Model Selection Guide exists). It handles specs that are not important enough to be in the guide.
- Source: Specs in the PDF that are not listed in the Model Selection Guide
- Logic: All requirements matching specifications in this tier are automatically classified as OPTIONAL/SECONDARY
- Analysis Output: List of matches including parameter name, user requirement, product specification, and justification for optional status

**Tier 3: Fallback to JSON Product Summary**
This is the fallback tier, used only when no PDF document can be found.
- Source: The JSON product summary file
- Logic: User requirements are matched against all available data in the JSON summary
  - A match is considered CRITICAL if the parameter is fundamental to the product's core function
  - Otherwise, the match is considered OPTIONAL
- Analysis Output: Perform the same parameter-by-parameter analysis as with PDFs:
  - Show parameter name
  - Show user requirement  
  - Show product specification from JSON (with field reference if available)
  - Provide a holistic explanation exactly as with PDFs

## 2. Parameter-Level Analysis Requirements
For each parameter (mandatory or optional):
1. Show **Parameter Name**(User Requirement)
2. Show **Product Specification with source reference (PDF field or JSON field)**
3. Provide a **single holistic explanation paragraph** that:
  - Explains why the user requirement **matches** the product specification.
  - Justifies the match using **datasheet or JSON evidence**.
  - Describes how this requirement contributes to the **overall suitability**.
  - Mentions interactions with **other parameters** if relevant.
  - Avoid breaking into subpoints; keep it integrated.

## 3. Output Structure
- Output must be in **Markdown**
- Separate **Mandatory** and **Optional** requirements
- **Do not include separate holistic analysis sections**; each parameter’s explanation should already be holistic
- Include **Comprehensive Analysis & Assessment**:
  - Final reasoning for selection

## 4. Sources
- Use PDF datasheets first if available
- Fallback to JSON product summaries if PDF is missing
- In fallback mode, the JSON fields must be treated as the **primary authoritative source** for all parameter-by-parameter matching

### **User Requirements**
{structured_requirements}

### **Primary Source: PDF Datasheet Content**
{pdf_content_json}

### **Fallback Source: JSON Product Summaries**
{products_json}

---


**I. Mandatory Parameters Analysis**

*[For each mandatory parameter, create a section like the example below. If none, state "No mandatory parameters to analyze."]*

- **[e.g., Output Signal](e.g., 4-20 mA)** 
  - **Product Specification:** [e.g., "4-20mA" from Datasheet Page 1, 'Communications/Output Options']
  - **Explanation:** [A concise paragraph explaining the match. This requirement is met because the datasheet explicitly lists 4-20mA as an output. This ensures compatibility with the user's control system and is a critical factor for model suitability.]

- **[e.g., Pressure Range](e.g., 10 inH2O span)**
  - **Product Specification:** [e.g., "URL 10 inH2O, LRL -10 inH2O" from Datasheet Page 1, 'Span & Range Limits']
  - **Explanation:** [Explain the match. The specified span of 10 inH2O is within the transmitter's upper and lower range limits, making it a perfect fit for the application's measurement needs.]

**II. Optional Parameters Analysis**

*[For each optional parameter, create a section like the example below. If none, state "No optional parameters to analyze."]*

- **[e.g., Wetted Parts Material](e.g., Hastelloy C276)** 
  - **Product Specification:** [e.g., "Hastelloy® C-276" available in Datasheet Page 15, 'Model Selection Guide']
  - **Explanation:** [Explain the match. The required material is available as a selectable option, ensuring the transmitter will have the necessary corrosion resistance for the process fluid.]


**III. Comprehensive Analysis & Assessment**
- **Reasoning for Selection:** 
  Provide a concise paragraph (2–3 sentences) that clearly states:
  1. Which **mandatory requirements** are fully met by the model.
  2. Which **optional requirements** are also satisfied.
  3. How the combination of these matches **supports the overall suitability** for the user's application.
  4. Reference any critical parameters explicitly, using datasheet or JSON as justification.

{format_instructions}
""")





ranking_prompt = ChatPromptTemplate.from_template("""
You are a product ranking specialist for industrial requisitioners and buyers. Based on the vendor analysis and original requirements, create an **overall ranking of all products** with detailed parameter-by-parameter analysis.

**CRITICAL: You must extract and preserve ALL information from the vendor analysis, especially:**
1. **Mandatory Parameters Analysis** - Convert these to Key Strengths
2. **Optional Parameters Analysis** - Convert these to Key Strengths or Concerns based on match
3. **Comprehensive Analysis & Assessment** - Extract both Reasoning and Key Limitations
4. **Any unmatched requirements** - These become Concerns

Original Requirements:
{structured_requirements}

Vendor Analysis Results:
{vendor_analysis}

For each product, provide detailed keyStrengths and concerns that include:

**Key Strengths:**  
For each parameter that matches requirements:
- **[Friendly Parameter Name](User Requirement)** - Product provides "[Product Specification]" - [Holistic explanation paragraph: why it matches, justification from datasheet/JSON, impact on overall suitability, interactions with other parameters].

**Concerns:**  
For each parameter that does not match:
- Holistic explanation paragraph: why it does not meet requirement, limitation, potential impact, interactions with other parameters.


**Guidelines:**
- **MANDATORY**: Extract and include ALL limitations mentioned in the vendor analysis "Key Limitations" section
- Include EVERY parameter from the user requirements in either strengths or concerns
- For each parameter, show: Parameter name, User requirement, Product specification, Detailed holistic explanation
- Explain the technical and business impact of each match or mismatch
- Each explanation should be 1-2 sentences that clearly show why it's a strength or concern
- Base explanations on actual specifications from the vendor analysis
- If a parameter wasn't analyzed, note it as "Not specified in available documentation"
- **Always preserve limitations from vendor analysis** - these are critical for buyer decision-making

Use clear, business-relevant language that helps buyers understand exactly how each product meets or fails to meet their specific requirements.
{format_instructions}
""")

# --- NEW PROMPT FOR ADDITIONAL REQUIREMENTS ---
additional_requirements_prompt = ChatPromptTemplate.from_template("""
You are an expert assistant for industrial requisitioners and buyers. The user wants to add or modify a requirement for a {product_type}.

User's new input:
{user_input}

Current requirements schema for the product:
{schema}

Tasks:
1. Identify and extract any specific requirements from the user's new input.
2. Only include requirements that are explicitly mentioned in this latest input. Do not repeat old requirements.
3. If no new requirements are found, return an empty dictionary for the requirements.

{format_instructions}
""")

# --- DYNAMIC STANDARDIZATION PROMPT ---
standardization_prompt = ChatPromptTemplate.from_template("""
You are an expert in industrial instrumentation and procurement standardization. Your task is to standardize naming conventions for industrial products, vendors, and specifications to create consistency across procurement systems.

Context: {context}

Data to standardize:
Vendor: {vendor}
Product Type: {product_type}
Model Family: {model_family}
Specifications: {specifications}

Instructions:
1. **Vendor Standardization**: Use proper corporate naming conventions (e.g., "ABB", "Emerson", "Siemens"). Consider common variations and abbreviations.

2. **Product Type Standardization**: 
   - Focus on the core measurement function and device type
   - Remove technology-specific modifiers (differential, vortex, smart, etc.)
   - Use standard industrial terminology
   - Consider what procurement professionals would search for
   - Examples: "pressure transmitter", "flow meter", "temperature sensor", "level transmitter"

3. **Model Family Standardization**: Clean up model naming while preserving essential identifiers.

4. **Specification Standardization**: Normalize units, terminology, and parameter names.

Provide standardized versions that are:
- Consistent with industry standards
- Searchable and procurement-friendly
- Free of unnecessary technical modifiers
- Properly capitalized and formatted

Be intelligent about variations - use your knowledge of industrial instrumentation to make appropriate standardization decisions.

Response format:
{{
    "vendor": "[standardized vendor name]",
    "product_type": "[standardized product type]", 
    "model_family": "[standardized model family]",
    "specifications": {{[standardized specifications]}}
}}
""")