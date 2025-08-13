import pytest
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync

@pytest.fixture
def orchestrator():
    groq = GroqClientAsync()
    return AsyncLangGraphOrchestrator(groq, human_in_loop=False)

@pytest.mark.asyncio
async def test_form_navigation():
    """Test navigation to next step in multi-step form"""
    html = """
    <form id="step1" class="loan-form">
        <div class="step">Step 1 of 3</div>
        <input type="text" name="business_name" value="Test Business">
        <button type="submit" class="next-button">Next Step</button>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply/step1",
        "html": html,
        "snippet": "First step of loan application",
        "visited_urls": ["https://example.com/apply/step1"]
    }
    
    result = await orchestrator.simulate_form_fill_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    action = result["_new_actions"][0]
    assert action["type"] == "form_next_click"
    assert "selector_used" in action

@pytest.mark.asyncio
async def test_form_fill_simulation():
    """Test if the system can simulate filling out form fields"""
    html = """
    <form id="application">
        <input type="text" name="business_name" required>
        <input type="email" name="email" required>
        <input type="tel" name="phone" required>
        <select name="loan_amount">
            <option value="10000">$10,000</option>
            <option value="25000">$25,000</option>
        </select>
        <button type="submit">Next</button>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply",
        "html": html,
        "snippet": "Application form with required fields",
        "visited_urls": ["https://example.com/apply"],
        "form_llm": {
            "fields": [
                {"name": "business_name", "value": "Test Corp"},
                {"name": "email", "value": "test@example.com"},
                {"name": "phone", "value": "1234567890"},
                {"name": "loan_amount", "value": "25000"}
            ]
        }
    }
    
    result = await orchestrator.simulate_form_fill_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    assert any(a["type"] == "form_next_click" for a in result["_new_actions"])

@pytest.mark.asyncio
async def test_form_fill_invalid_selectors():
    """Test behavior when form fields cannot be found"""
    html = """
    <form id="application">
        <input type="text" name="different_field">
        <button type="submit">Next</button>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply",
        "html": html,
        "snippet": "Form with different field names",
        "visited_urls": ["https://example.com/apply"],
        "form_llm": {
            "fields": [
                {"name": "nonexistent_field", "value": "test"}
            ]
        }
    }
    
    result = await orchestrator.simulate_form_fill_node(state)
    
    assert result is not None
    assert "_new_actions" in result
    assert any(a["type"] == "form_next_no_nav" for a in result["_new_actions"])
