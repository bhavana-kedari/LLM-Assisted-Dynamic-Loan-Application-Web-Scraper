# tests/test_simulate_fill.py
import asyncio
import pytest
from types import SimpleNamespace
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClient

pytestmark = pytest.mark.asyncio

class FakePage:
    def __init__(self, html, url):
        self._html = html
        self.url = url
        self._filled = []
    async def goto(self, url, **kwargs):
        # no-op
        self.url = url
    async def content(self):
        return self._html
    async def query_selector(self, sel):
        # return a fake element if selector seems present; we return a SimpleNamespace with methods
        if "nonexistent" in sel:
            return None
        return SimpleNamespace(click=self._click, fill=self._fill, check=self._check)
    async def wait_for_selector(self, sel, timeout=0):
        if "notfound" in sel:
            raise Exception("not found")
        return True
    async def eval_on_selector_all(self, sel, script):
        # simulate select options
        return ["", "personal", "business"]
    async def screenshot(self, path=None, full_page=True):
        # write nothing — test won't assert file contents
        return None
    async def wait_for_load_state(self, _):
        return None
    async def _click(self):
        # simulate changing html/url on click for next navigation
        self.url = self.url + "/next"
        self._html = "<html><body><form><input name='step2_field'/></form></body></html>"
    async def _fill(self, value):
        self._filled.append(value)
    async def _check(self):
        pass

class FakeBrowser:
    def __init__(self, page):
        self.page = page
    async def new_context(self):
        return self
    async def new_page(self):
        return self.page
    async def close(self):
        return None

class FakeAsyncPlaywright:
    def __init__(self, page):
        self._page = page
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        return None
    async def chromium(self):
        return self
    async def launch(self, headless=True):
        return FakeBrowser(self._page)

@pytest.fixture
def fake_groq():
    class FakeGroq:
        async def classify_page(self, snippet, url=""):
            return {"page_type": "form", "confidence": 0.9}
        async def extract_form_fields(self, snippet):
            # returns that there is a form with two fields and multi_step true
            return {"form_name":"Fake","multi_step": True,"steps_count":2,"fields":[{"name":"first","input_type":"text","required":True,"selector_hint":"input[name='first']"}],"confidence":0.9}
        async def suggest_next_selectors(self, snippet):
            return {"next_selectors":[{"selector":"button.next","type":"css","label":"Next"}],"confidence":0.9}
    return FakeGroq()

@pytest.mark.asyncio
async def test_simulate_fill_multi_step(monkeypatch, tmp_path, fake_groq):
    html_step1 = "<html><body><form><input name='first'/></form><button class='next'>Next</button></body></html>"
    fake_page = FakePage(html_step1, "https://fake.test")
    # patch the orchestrator's _get_playwright to return our fake context manager
    orch = AsyncLangGraphOrchestrator(fake_groq, human_in_loop=False)

    async def fake_get_playwright():
        return FakeAsyncPlaywright(fake_page)

    monkeypatch.setattr(orch, "_get_playwright", fake_get_playwright)

    state = {"url":"https://fake.test","html":html_step1,"snippet":"apply now","actions":[],"forms":[],"runtime":{}}
    # call the simulate_fill_node directly
    updates = await orch.simulate_fill_node(state)
    # assert runtime simulate_fill exists and that multiple steps recorded
    runtime = updates.get("runtime", {})
    assert "simulate_fill" in runtime
    sim = runtime["simulate_fill"]
    assert "steps" in sim and isinstance(sim["steps"], list)
    # there should be at least one step recorded
    assert len(sim["steps"]) >= 1
    # each step has fills_summary
    assert "fills_summary" in sim["steps"][0]
