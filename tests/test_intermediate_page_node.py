import pytest
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync

@pytest.fixture
def orchestrator():
    groq = GroqClientAsync()
    return AsyncLangGraphOrchestrator(groq, human_in_loop=False)

@pytest.mark.asyncio
async def test_intermediate_page_options_extraction():
    """Test if LLM correctly extracts and structures loan options"""
    html = """
    <div class="loan-options">
        <div class="option" id="business-loan">
            <h3>Business Loan</h3>
            <p>Up to $500,000 for established businesses</p>
            <a href="/apply/business">Select</a>
        </div>
        <div class="option" id="startup-loan">
            <h3>Startup Funding</h3>
            <p>Perfect for new businesses</p>
            <a href="/apply/startup">Select</a>
        </div>
        <div class="recommended option" id="equipment">
            <h3>Equipment Financing</h3>
            <p>Best rates for equipment purchases</p>
            <a href="/apply/equipment">Select</a>
        </div>
    </div>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/options",
        "html": html,
        "snippet": "Multiple loan options including recommended ones",
    }
    
    result = await orchestrator.intermediate_page_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    
    # Check if LLM found all options
    llm_resp = result.get("intermediate_llm", {})
    options = llm_resp.get("options", [])
    assert len(options) == 3
    
    # Verify recommended option is marked
    recommended = [opt for opt in options if opt.get("recommended", False)]
    assert len(recommended) == 1
    assert "equipment" in recommended[0].get("text", "").lower()

@pytest.mark.asyncio
async def test_intermediate_page_option_selection():
    """Test if the system can successfully click on a selected option"""
    html = """
    <div class="loan-options">
        <a href="/apply/business" id="business-loan" class="option">Business Loan</a>
    </div>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/options",
        "html": html,
        "snippet": "Single business loan option",
        "visited_urls": ["https://example.com/options"]
    }
    
    result = await orchestrator.intermediate_page_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    action = result["_new_actions"][0]
    
    # Should have clicked the option
    assert action["type"] == "intermediate_option_selected"
    assert "selector_used" in action

@pytest.mark.asyncio
async def test_intermediate_page_no_options():
    """Test behavior when no valid options are found"""
    html = """
    <div class="content">
        <p>No loan options available at this time.</p>
    </div>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/options",
        "html": html,
        "snippet": "Page with no loan options",
        "visited_urls": ["https://example.com/options"]
    }
    
    result = await orchestrator.intermediate_page_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    action = result["_new_actions"][0]
    assert action["type"] == "no_option_selected"
