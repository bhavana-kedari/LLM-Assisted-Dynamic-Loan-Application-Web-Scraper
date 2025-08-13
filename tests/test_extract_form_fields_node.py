import pytest
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync

@pytest.fixture
def orchestrator():
    groq = GroqClientAsync()
    return AsyncLangGraphOrchestrator(groq, human_in_loop=False)

@pytest.mark.asyncio
async def test_form_field_extraction():
    """Test extraction of various types of form fields"""
    html = """
    <form action="/submit" method="POST">
        <div class="form-group">
            <label for="business_name">Business Name</label>
            <input type="text" id="business_name" name="business_name" required>
        </div>
        <div class="form-group">
            <label for="loan_amount">Loan Amount</label>
            <select id="loan_amount" name="loan_amount">
                <option value="10000">$10,000</option>
                <option value="25000">$25,000</option>
                <option value="50000">$50,000</option>
            </select>
        </div>
        <div class="form-group">
            <label>Business Type</label>
            <input type="radio" name="business_type" value="startup"> Startup
            <input type="radio" name="business_type" value="established"> Established
        </div>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply",
        "html": html,
        "snippet": "Loan application form with various field types",
    }
    
    result = await orchestrator.extract_form_fields_node(state)
    
    assert result is not None
    assert "forms" in result
    assert len(result["forms"]) > 0
    
    form_data = result["forms"][0]
    assert "form_llm" in form_data
    assert "parsed_forms" in form_data
    
    # Check if all field types were detected
    fields = form_data["parsed_forms"]
    field_types = {f.get("type") for f in fields}
    assert "text" in field_types
    assert "select" in field_types
    assert "radio" in field_types

@pytest.mark.asyncio
async def test_multi_step_form_detection():
    """Test if the system correctly identifies multi-step forms"""
    html = """
    <form action="/submit-step-1" method="POST">
        <div class="step-indicator">Step 1 of 3</div>
        <div class="form-group">
            <label for="business_name">Business Name</label>
            <input type="text" id="business_name" name="business_name">
        </div>
        <button type="submit">Next Step</button>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply/step1",
        "html": html,
        "snippet": "First step of multi-step form",
    }
    
    result = await orchestrator.extract_form_fields_node(state)
    
    assert result is not None
    assert "form_llm" in result
    assert result["form_llm"].get("multi_step") == True
    assert "_new_actions" in result
    assert result["_new_actions"][0]["type"] == "form_extracted"
    assert result["_new_actions"][0]["multi_step"] == True

@pytest.mark.asyncio
async def test_form_field_validation():
    """Test detection of required fields and validation patterns"""
    html = """
    <form action="/submit" method="POST">
        <div class="form-group">
            <label for="email">Email</label>
            <input type="email" id="email" name="email" required>
        </div>
        <div class="form-group">
            <label for="phone">Phone</label>
            <input type="tel" id="phone" name="phone" pattern="[0-9]{10}" required>
        </div>
    </form>
    """
    orchestrator = await orchestrator()
    state = {
        "url": "https://example.com/apply",
        "html": html,
        "snippet": "Form with validation requirements",
    }
    
    result = await orchestrator.extract_form_fields_node(state)
    
    assert result is not None
    form_data = result["forms"][0]["parsed_forms"]
    
    # Check if required and validation patterns were detected
    email_field = next(f for f in form_data if f.get("name") == "email")
    phone_field = next(f for f in form_data if f.get("name") == "phone")
    
    assert email_field.get("required") == True
    assert phone_field.get("required") == True
    assert "pattern" in phone_field
