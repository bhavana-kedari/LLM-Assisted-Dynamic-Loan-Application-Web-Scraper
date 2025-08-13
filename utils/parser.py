"""
HTML parsing helpers using BeautifulSoup (sync functions).
Used by async nodes — that's fine (pure CPU-bound parsing).
"""

from bs4 import BeautifulSoup
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    text = soup.get_text(separator="\n", strip=True)
    text = "\n".join([line.strip() for line in text.splitlines() if line.strip()])
    return text


def extract_forms(html_or_text: str) -> List[Dict]:
    try:
        soup = BeautifulSoup(html_or_text, "html.parser")
        forms = []
        for f in soup.find_all("form"):
            inputs = []
            for inp in f.find_all(["input", "select", "textarea"]):
                attrs = dict(inp.attrs)
                input_type = attrs.get("type", "text") if inp.name == "input" else inp.name
                inputs.append({
                    "tag": inp.name,
                    "type": input_type,
                    "name": attrs.get("name"),
                    "id": attrs.get("id"),
                    "attrs": attrs
                })
            forms.append({
                "action": f.get("action"),
                "method": f.get("method"),
                "inputs": inputs
            })
        return forms
    except Exception as e:
        logger.exception("Failed to parse forms: %s", e)
        return []
