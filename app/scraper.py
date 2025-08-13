"""
Async AppScraper that fetches initial page with Playwright async API and invokes the async LangGraph orchestrator.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

from playwright.async_api import async_playwright

from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync
from utils.io import save_json_atomic
from utils.parser import html_to_text

logger = logging.getLogger(__name__)


@dataclass
class ScrapeResult:
    site: str
    metadata: Dict[str, Any]
    saved_path: str


class AppScraper:
    def __init__(self, out_dir: Path = Path("data"), human_in_loop: bool = False):
        self.out_dir = out_dir
        self.human_in_loop = human_in_loop
        self.groq = GroqClientAsync()
        self.orchestrator = AsyncLangGraphOrchestrator(self.groq, human_in_loop=self.human_in_loop)

    async def _open_page(self, url: str) -> Dict[str, Any]:
        """
        Open page with Playwright async API and return HTML & URL.
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            resp = await page.goto(url, wait_until="domcontentloaded", timeout=60000)
            status = resp.status if resp else None
            html = await page.content()
            current_url = page.url
            await browser.close()
            return {"html": html, "status": status, "url": current_url}

    async def run(self, url: str) -> ScrapeResult:
        # open initial page
        page_payload = await self._open_page(url)
        html = page_payload["html"]
        page_url = page_payload["url"]

        snippet = html_to_text(html)[:4000]

        # invoke async LangGraph orchestrator
        metadata = await self.orchestrator.invoke_start(url=page_url, html=html, snippet=snippet)

        filename = f"{self.out_dir}/{self._sanitize_filename(page_url)}.json"
        save_json_atomic(metadata, filename)
        return ScrapeResult(site=page_url, metadata=metadata, saved_path=filename)

    @staticmethod
    def _sanitize_filename(url: str) -> str:
        return (
            url.replace("://", "_")
            .replace("/", "_")
            .replace("?", "_")
            .replace("&", "_")
            .replace("=", "_")
        )
