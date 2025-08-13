"""
Async main entry point for the LangGraph async flow.
Usage:
    python -m app.main --url "https://example-loans.example/apply"
"""

import argparse
import asyncio
import logging
from pathlib import Path

from utils.logging_config import configure_logging
from app.scraper import AppScraper

from dotenv import load_dotenv

configure_logging()
logger = logging.getLogger(__name__)

load_dotenv()



def parse_args():
    parser = argparse.ArgumentParser(description="Async LangGraph-powered loan form scraper")
    parser.add_argument("--url", required=True, help="Starting URL of the site to scan")
    parser.add_argument("--out", default="data", help="Output directory for JSON metadata")
    parser.add_argument("--human", action="store_true", help="Force human-in-the-loop prompts")
    return parser.parse_args()


async def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting async scraper for %s", args.url)
    scraper = AppScraper(out_dir=out_dir, human_in_loop=args.human)
    try:
        result = await scraper.run(args.url)
        logger.info("Scrape completed. Metadata saved to %s", result.saved_path)
    except Exception as e:
        logger.exception("Fatal error during scraping: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
