import pytest
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync
from playwright.async_api import async_playwright
from utils.parser import html_to_text

@pytest.fixture
async def orchestrator():
    groq = GroqClientAsync()
    return AsyncLangGraphOrchestrator(groq, human_in_loop=False)

async def get_page_content(url: str) -> tuple[str, str]:
    """Helper to get HTML and snippet from a URL"""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded")
        html = await page.content()
        snippet = html_to_text(html)[:4000]
        await browser.close()
        return html, snippet

@pytest.mark.asyncio
async def test_capital_funding_form():
    """Test classification of Capital Funding's application form page"""
    # This should be the URL after clicking Apply Now
    url = "https://capitalfunding.com/apply"
    html, snippet = await get_page_content(url)
    orchestrator = await orchestrator()
    state = {
        "url": url,
        "html": html,
        "snippet": snippet
    }
    
    result = await orchestrator.classify_page_node(state)
    
    assert result is not None
    assert "classification" in result
    print(f"\nCapital Funding form classification: {result['classification']}")
    print(f"LLM details: {result['classification_llm']}")
    
    # Should be classified as a form page
    assert result["classification"] == "form"

@pytest.mark.asyncio
async def test_ondeck_product_selection():
    """Test classification of OnDeck's product selection page"""
    # This should be their loan products page
    url = "https://www.ondeck.com/small-business-loans"
    html, snippet = await get_page_content(url)
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/loan-options",
        "html": html,
        "snippet": "Page with multiple loan type options",
    }
    
    result = await orchestrator.classify_page_node(state)
    
    assert result is not None
    assert "classification" in result
    assert result["classification"] == "intermediate"
    assert "classification_llm" in result
    assert result["classification_llm"]["choice_options_found"] == True

@pytest.mark.asyncio
async def test_classify_ambiguous_page():
    """Test classification of a page that could be ambiguous"""
    # Page with both form elements and options
    html = """
    <div class="page-content">
        <h2>Quick Apply</h2>
        <a href="/full-application" class="button">Full Application</a>
        <form class="quick-form">
            <input type="email" placeholder="Email">
            <button>Get Started</button>
        </form>
    </div>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/quick-apply",
        "html": html,
        "snippet": "Page with both form and navigation elements",
    }
    
    result = await orchestrator.classify_page_node(state)
    
    # Should classify based on primary purpose
    assert result is not None
    assert "classification" in result
    assert "classification_llm" in result
    assert "confidence" in result["classification_llm"]
