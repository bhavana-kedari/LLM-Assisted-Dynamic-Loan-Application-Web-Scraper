import pytest
from pathlib import Path
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock

# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.orchestrator import AsyncLangGraphOrchestrator
from playwright.async_api import async_playwright
from llm.groq_client import GroqClientAsync

# Create a mock GroqClient for testing
class MockGroqClient(GroqClientAsync):
    def __init__(self):
        pass
    
    async def classify_form_or_intermediate(self, snippet: str, url: str):
        return {"page_type": "form"}
        
    async def find_loan_application_nav(self, snippet: str):
        return {"loan_application_buttons": []}

@pytest.mark.asyncio
async def test_try_selectors_with_valid_button():
    """Test that _try_selectors works with a valid button selector"""
    mock_groq = MockGroqClient()
    orchestrator = AsyncLangGraphOrchestrator(groq=mock_groq)
    url = "https://brooklynfundinggroup.com/apply-now/"
    selectors = [
        {
            "selector": "#new-borrower-apply",
            "text": "Apply Now"
        },
        {
            "selector": '//*[contains(text(),"Apply for Your First Loan")]',
            "text": "Apply for Your First Loan"
        }
    ]
    
    result = await orchestrator._try_selectors(url, selectors, {})
    assert result is not None
    assert "url" in result
    assert "html" in result
    assert "selector" in result
    assert result["selector"] in [s["selector"] for s in selectors]

@pytest.mark.asyncio
async def test_try_selectors_with_invalid_selector():
    """Test that _try_selectors handles invalid selectors gracefully"""
    mock_groq = MockGroqClient()
    orchestrator = AsyncLangGraphOrchestrator(groq=mock_groq)
    url = "https://brooklynfundinggroup.com/apply-now/"
    selectors = [
        {
            "selector": "#non-existent-button",
            "text": "Does Not Exist"
        }
    ]
    
    result = await orchestrator._try_selectors(url, selectors, {})
    assert result is None

@pytest.mark.asyncio
async def test_try_selectors_with_multiple_selectors():
    """Test that _try_selectors tries multiple selectors in order"""
    mock_groq = MockGroqClient()
    orchestrator = AsyncLangGraphOrchestrator(groq=mock_groq)
    url = "https://brooklynfundinggroup.com/apply-now/"
    selectors = [
        {
            "selector": "#non-existent-button",
            "text": "Does Not Exist"
        },
        {
            "selector": '//*[contains(text(),"Apply for Your First Loan")]',
            "text": "Apply for Your First Loan"
        }
    ]
    
    result = await orchestrator._try_selectors(url, selectors, {})
    assert result is not None
    assert result["selector"] == selectors[1]["selector"]

@pytest.mark.asyncio
async def test_try_selectors_element_state():
    """Test that _try_selectors properly checks element state before clicking"""
    mock_groq = MockGroqClient()
    orchestrator = AsyncLangGraphOrchestrator(groq=mock_groq)
    url = "https://brooklynfundinggroup.com/apply-now/"
    selectors = [
        {
            "selector": '//*[contains(text(),"Apply for Your First Loan")]',
            "text": "Apply for Your First Loan"
        }
    ]
    
    result = await orchestrator._try_selectors(url, selectors, {})
    assert result is not None
    assert result is not None
    assert "visible" in result
    assert "enabled" in result
    assert result["visible"] is True
    assert result["enabled"] is True
    assert "bbox" in result  # bounding box information

@pytest.mark.asyncio
async def test_try_selectors_already_clicked():
    """Test that _try_selectors respects already_clicked list"""
    mock_groq = MockGroqClient()
    orchestrator = AsyncLangGraphOrchestrator(groq=mock_groq)
    url = "https://brooklynfundinggroup.com/apply-now/"
    selectors = [
        {
            "selector": '//*[contains(text(),"Apply for Your First Loan")]',
            "text": "Apply for Your First Loan"
        }
    ]
    
    # First click should work
    result1 = await orchestrator._try_selectors(url, selectors, {})
    assert result1 is not None
    
    # Second click with same selector in already_clicked should not click
    xpath_selector = f'//*[contains(text(),"{selectors[0]["text"]}")]'
    state = {"already_clicked": [xpath_selector]}
    result2 = await orchestrator._try_selectors(url, selectors, state)
    assert result2 is None

if __name__ == "__main__":
    # Change to the project root directory before running tests
    os.chdir(os.path.dirname(os.path.dirname(__file__)))
    pytest.main(["-v", __file__])
