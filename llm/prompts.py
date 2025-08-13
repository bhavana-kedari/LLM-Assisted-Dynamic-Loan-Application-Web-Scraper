# llm/prompts.py

PROMPT_FIND_LOAN_NAV = """
You are an expert at finding loan application links in website navigation.
Given the page snippet, identify buttons/links that would lead to a loan application.
Focus on navigation bar items and main call-to-action buttons.

Page snippet: {snippet}

Return a JSON object:
{{
    "loan_application_buttons": [
        {{
            "text": "button text",
            "selector": "CSS selector",
            "confidence": 0.0-1.0,
            "reason": "why this is likely a loan application button"
        }}
    ],
    "nav_area_found": true/false
}}
ONLY output the JSON.
"""

PROMPT_CLASSIFY_FORM_OR_INTERMEDIATE = """
You are analyzing a page in a loan application flow. You must determine if this is:

1. An INTERMEDIATE selection page:
   - Contains multiple loan options/products to choose from
   - Has buttons/cards for different loan types (e.g., "Business Loan", "Equipment Financing")
   - Asks user to select a category or type BEFORE showing the actual application form
   - Does NOT contain form input fields like name, email, phone, etc.

2. A FORM page:
   - Contains actual input fields (text inputs, dropdowns, checkboxes)
   - Asks for personal/business information
   - Has fields like name, email, phone, amount requested
   - Shows a loan application form
   - Simple CTA buttons like "Apply Now" that lead directly to forms are NOT intermediate pages

Page snippet: {snippet}
URL: {url}

Return a JSON object:
{{
    "page_type": "intermediate" or "form",
    "confidence": 0.0-1.0,
    "reason": "detailed explanation of why this matches form or intermediate criteria",
    "form_fields_found": boolean,  # Are there ANY input fields for collecting user data?
    "choice_options_found": boolean,  # Are there MULTIPLE loan type options to select from?
    "input_fields": ["list", "of", "field", "names", "found"],  # Empty list if none found
    "loan_options": ["list", "of", "loan", "types", "found"]  # Empty list if none found
}}

IMPORTANT: A simple "Apply Now" or "Get Started" page with a single loan application button is NOT an intermediate page - it's just a landing page that leads to a form.

ONLY output valid JSON.
"""

PROMPT_ANALYZE_INTERMEDIATE_OPTIONS = """
You are analyzing an intermediate page in a loan application flow.
Extract all clickable options that a user might need to choose from.

Page snippet: {snippet}

Return a JSON object:
{{
    "options": [
        {{
            "text": "option text/label",
            "selector": "CSS selector",
            "description": "what this option means",
            "recommended": boolean
        }}
    ],
    "total_options": number,
    "requires_human_choice": boolean,
    "decision_context": "what decision the user needs to make here"
}}
ONLY output the JSON.
"""
# {snippet}

# Return a JSON object:
# {{
#   "suggested_actions": [
#     {{"selector":"button.apply-now", "type":"css", "label":"Apply Now"}},
#     {{"selector":"//a[contains(text(),'Apply')]", "type":"xpath", "label":"Apply link"}}
#   ],
#   "confidence": 0.0
# }}

# IMPORTANT:
# - Ensure any XPath selector uniquely identifies exactly one element on the page.
# - Prefer attributes like 'id', 'data-testid', or unique text content in the XPath.
# - Do NOT output anything other than JSON.
# ONLY output JSON.
# """

PROMPT_EXTRACT_INTERMEDIATE = """
This page appears to be an intermediate page (choose client/product).
Given the snippet:
{snippet}

Return JSON:
{{
  "options": ["Individual", "Business", "Mortgage"],
  "selector_hints": [{{"selector":"button.client-individual","type":"css"}}],
  "confidence": 0.0
}}

IMPORTANT:
- If providing XPath selectors, ensure each XPath matches exactly one element uniquely.
- Use distinctive attributes or text for XPath uniqueness.
ONLY output JSON.
"""

PROMPT_EXTRACT_FORM = """
You are a scraper that extracts structured form data for a loan application page.
Given the page snippet:
{snippet}

Return JSON:
{{
  "form_name":"Personal Loan Application",
  "multi_step": true,
  "steps_count": 3,
  "fields": [
    {{"name":"First name", "input_type":"text", "required":true, "selector_hint":"input[name='first_name']","notes":""}}
  ],
  "confidence": 0.0
}}
Only output JSON.
"""

PROMPT_SUGGEST_NEXT = """
You are a scraper assistant that should suggest the UI elements which go to the next step in a multi-step application.
Given the page snippet:
{snippet}

Return JSON:
{{
  "next_selectors": [
    {{"selector":"button.next-step", "type":"css", "label":"Next"}},
    {{"selector":"//button[contains(.,'Continue')]", "type":"xpath", "label":"Continue"}}
  ],
  "confidence": 0.0
}}

IMPORTANT:
- Any XPath selector must uniquely identify exactly one element.
- Use stable, unique attributes or exact text matches in XPath.
ONLY output JSON.
"""

NEXT_BUTTON_PROMPT = """
You are an expert at finding navigation buttons on web forms.
Given this page snippet:

{snippet}

Please provide CSS selectors for buttons or links that are most likely to be the "Next" button to advance the form step.
Return a JSON with a list of selectors:

{{
  "next_selectors": [
    {{"selector": "button.next"}},
    {{"selector": "input[type='submit']"}}
  ]
}}
"""

