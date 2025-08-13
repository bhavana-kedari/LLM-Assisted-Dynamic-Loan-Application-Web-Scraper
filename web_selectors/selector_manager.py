"""
SelectorManager: lightweight selector heuristics for async flow.
"""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class SelectorManager:
    def __init__(self):
        self.registry = {}

    def rank_selectors(self, candidates: List[Dict]) -> List[Dict]:
        def score(cand):
            s = 0
            sel = cand.get("selector", "")
            text = cand.get("text", "")
            
            # Highest priority: Unique identifiers
            if sel.strip().startswith("#"):
                s += 100  # ID selectors are most reliable
                
            # Test-specific attributes
            if "data-testid" in sel or "data-test" in sel:
                s += 90
                
            # Form-specific attributes
            if "name=" in sel or "for=" in sel:
                s += 80
                
            # Accessibility attributes
            if "aria-label" in sel:
                s += 70
            if "[role=" in sel:
                s += 65
                
            # Data attributes
            if "data-" in sel:
                s += 60
            
            # Text matching with button/link context
            if text and ("button" in sel.lower() or "a[" in sel or "link" in sel.lower()):
                s += 55
                
            # Class-based selectors
            if sel.count(".") == 1:  # Single class
                s += 50
            elif sel.count(".") > 1:  # Multiple classes
                s += 40
                
            # Position-based XPath
            if "nth-child" in sel or "nth-of-type" in sel:
                s += 30
                
            # Text content exact match
            if text and f'text()="{text}"' in sel:
                s += 45
            elif text and "contains(text()" in sel:
                s += 35
                
            # Penalties
            if len(sel) > 150:  # Very long selectors
                s -= 20
            if sel.count("/") > 5:  # Deep XPath
                s -= 15
            if sel.count(" > ") > 3:  # Deep CSS path
                s -= 15
                
            return s

        ranked = sorted(candidates, key=lambda c: -score(c))
        logger.debug("Ranked selectors: %s", ranked)
        return ranked

    def choose_best(self, candidates: List[Dict]) -> Optional[Dict]:
        ranked = self.rank_selectors(candidates)
        return ranked[0] if ranked else None

    def generate_selector_from_attrs(self, tag: str, attrs: Dict[str, str]) -> str:
        if "id" in attrs:
            return f"{tag}#{attrs['id']}"
        for k in attrs:
            if k.startswith("data-"):
                return f"{tag}[{k}='{attrs[k]}']"
        if "aria-label" in attrs:
            return f"{tag}[aria-label='{attrs['aria-label']}']"
        if "name" in attrs:
            return f"{tag}[name='{attrs['name']}']"
        if "class" in attrs:
            cls = attrs["class"].split()[0]
            return f"{tag}.{cls}"
        if attrs:
            k, v = next(iter(attrs.items()))
            return f"{tag}[{k}='{v}']"
        return tag
