"""
Async example to simulate run against a fake page.
"""

import asyncio
from pathlib import Path
from app.scraper import AppScraper

SAMPLE_FAKE_HTML = """
<html>
<head><title>Fake Loan - Apply</title></head>
<body>
<h1>Apply for a personal loan</h1>
<a class="apply" href="/apply">Apply Now</a>
<form action="/submit" method="post">
  <input name="first_name" />
  <input name="email" type="email" />
  <select name="loan_type"><option value="">Choose</option><option value="personal">Personal</option></select>
</form>
</body>
</html>
"""

async def run_fake():
    outdir = Path("data")
    scraper = AppScraper(out_dir=outdir, human_in_loop=True)
    metadata = await scraper.orchestrator.invoke_start(url="https://fake-loans.example/", html=SAMPLE_FAKE_HTML, snippet="Apply for a personal loan")
    import json
    print("Simulated metadata:")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    asyncio.run(run_fake())
