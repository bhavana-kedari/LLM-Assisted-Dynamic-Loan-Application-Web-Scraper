"""
Async-capable Groq client wrapper.

Provides async wrappers around HTTP calls using asyncio.to_thread
because the Groq HTTP SDK here is synchronous (requests).
Replace with an async client if available.

Each method returns a dict; the LLM is instructed to output JSON
which we attempt to parse safely.
"""

import os
import logging
import asyncio
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
import time

# Load env vars from .env before we read them
load_dotenv()

from .prompts import (
    PROMPT_EXTRACT_FORM,
    PROMPT_FIND_LOAN_NAV,
    PROMPT_CLASSIFY_FORM_OR_INTERMEDIATE,
    PROMPT_ANALYZE_INTERMEDIATE_OPTIONS
)


logger = logging.getLogger(__name__)


class GroqClientAsync:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.base_url = base_url or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")
        self.model = model or os.getenv("GROQ_MODEL", "llama3-70b-8192")
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set. Groq calls will fail until provided.")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    async def find_next_or_submit_button(self, html: str, snippet: str):
        """
        Ask Groq to classify the main action button as either 'next' or 'submit'
        and return the type along with the CSS/XPath selectors.
        """
        prompt = f"""
        You are an expert at analyzing HTML pages. 
        Given this HTML and snippet, determine if the main action button is a 'next' button or a 'submit' button.

        Rules:
        - If it proceeds to another step, call it "next".
        - If it finalizes or submits the form, call it "submit".
        - Output format: JSON with keys "page_type" and "selectors" (list of CSS selectors).
        
        HTML:
        {html}

        Snippet:
        {snippet}
        """

        response = await self.client.chat.completions.create(
            model="llama3-8b-8192",  # or whichever model you're using
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )

        import json
        try:
            parsed = json.loads(response.choices[0].message.content)
            return parsed.get("page_type", "none"), parsed.get("selectors", [])
        except Exception as e:
            self.logger.error(f"Error parsing Groq response for next/submit: {e}")
            return "none", []


    async def _call(self, prompt: str, max_tokens: int = 800) -> Dict[str, Any]:
        """Send a prompt to Groq using the chat completions endpoint."""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a precise JSON-producing assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
        }

        endpoint = f"{self.base_url}/chat/completions"

        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                resp = await client.post(endpoint, headers=self.headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                result_text = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )

                # 🔍 Print the raw response for debugging
                print("\n================ RAW LLM RESPONSE ================\n")
                print(result_text)
                print("\n===================================================\n")

                return {"raw": data, "text": result_text}
            except Exception as e:
                logger.exception("Groq API call failed: %s", e)
                return {"error": str(e), "text": ""}
            


    async def extract_form_fields(self, snippet: str) -> Dict[str, Any]:
        """Extract form fields and their properties from a form page."""
        prompt = PROMPT_EXTRACT_FORM.format(snippet=snippet)
        resp = await self._call(prompt)
        return self._safe_parse_json(resp.get("text", ""))

    async def find_loan_application_nav(self, snippet: str) -> Dict[str, Any]:
        """
        Find navigation elements that lead to loan application pages.
        Returns a list of potential buttons/links with their selectors.
        """
        prompt = PROMPT_FIND_LOAN_NAV.format(snippet=snippet)
        resp = await self._call(prompt)
        return self._safe_parse_json(resp.get("text", ""))

    async def classify_form_or_intermediate(self, snippet: str, url: str = "") -> Dict[str, Any]:
        """
        Classify a page as either a form page or an intermediate selection page.
        """
        prompt = PROMPT_CLASSIFY_FORM_OR_INTERMEDIATE.format(snippet=snippet, url=url)
        resp = await self._call(prompt)
        return self._safe_parse_json(resp.get("text", ""))

    async def analyze_intermediate_options(self, snippet: str) -> Dict[str, Any]:
        """
        Analyze an intermediate page to extract all possible options a user might need to select from.
        Returns detailed information about each option to help with human decision making.
        """
        prompt = PROMPT_ANALYZE_INTERMEDIATE_OPTIONS.format(snippet=snippet)
        resp = await self._call(prompt)
        return self._safe_parse_json(resp.get("text", ""))

    def _safe_parse_json(self, text: str) -> Dict[str, Any]:
        """Attempt to extract and parse JSON from possibly messy LLM output."""
        import json
        import re

        # 📜 Always print what we received before parsing
        print("\n🔍 Parsing LLM output:\n", text, "\n")

        if not text:
            logger.debug("LLM returned empty text for JSON parsing.")
            return {"raw_text": "", "confidence": 0.0}

        # Strip common markdown JSON fences
        fenced = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```", text.strip(), re.IGNORECASE)
        if fenced:
            text = fenced.group(1).strip()
            logger.debug("🔍 Stripped markdown code fences. New text:\n%s", text)

        # First attempt: naive bracket slice
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            candidate = text[start:end]
            parsed = json.loads(candidate)
            logger.debug("✅ Parsed JSON: %s", parsed)
            return parsed
        except Exception:
            pass  # We'll try regex fallback next

        # Fallback: scan for any {...} blocks
        matches = re.findall(r"\{[\s\S]*\}", text)
        for match in matches:
            try:
                parsed = json.loads(match)
                logger.debug("✅ Parsed JSON via fallback: %s", parsed)
                return parsed
            except Exception:
                continue

        logger.debug("❌ No valid JSON found in text.")
        return {"raw_text": text, "confidence": 0.0}
    
  
