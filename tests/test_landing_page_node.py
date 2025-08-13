import os
import sys
import asyncio
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from agents.orchestrator import AsyncLangGraphOrchestrator
from llm.groq_client import GroqClientAsync
from playwright.async_api import async_playwright
from utils.parser import html_to_text

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
async def orchestrator():
    groq = GroqClientAsync()
    return AsyncLangGraphOrchestrator(groq, human_in_loop=False)

async def get_page_content(url: str) -> tuple[str, str]:
    """Helper to get HTML and snippet from a URL"""
    print(f"\n🌐 Opening URL: {url}")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        try:
            # Maximize the browser window
            await page.set_viewport_size({'width': 1920, 'height': 1080})
            await page.evaluate("document.documentElement.requestFullscreen()")
            
            print("⏳ Loading page...")
            # Wait for full network load and a bit extra for dynamic content
            # First try to load with networkidle, fallback to domcontentloaded if timeout
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)  # 60 second timeout
                print("✅ Page loaded successfully with networkidle")
            except Exception as e:
                print(f"⚠️ Network idle timeout, falling back to domcontentloaded: {str(e)}")
                await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # Wait additional time for any dynamic content
                await asyncio.sleep(5)
                print("✅ Page loaded successfully with fallback method")
            
            # Debug: Get all visible buttons/links with enhanced detection
            print("\n🔍 Scanning for clickable elements with enhanced detection...")
            buttons = await page.evaluate("""(() => {
                const isElementInViewport = (el) => {
                    const rect = el.getBoundingClientRect();
                    return (
                        rect.top >= 0 &&
                        rect.left >= 0 &&
                        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
                        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
                    );
                };
                
                const getElementText = (el) => {
                    // Try various ways to get text content
                    return (
                        el.innerText || 
                        el.textContent || 
                        el.getAttribute('aria-label') || 
                        el.getAttribute('title') || 
                        el.getAttribute('alt') || 
                        ''
                    ).toLowerCase().trim();
                };
                
                const isClickable = (el) => {
                    try {
                        const style = window.getComputedStyle(el);
                        const isInteractive = el.tagName === 'A' || el.tagName === 'BUTTON' || 
                                            el.getAttribute('role') === 'button' ||
                                            el.onclick != null;
                        const isVisible = style.display !== 'none' && 
                                        style.visibility !== 'hidden' && 
                                        parseFloat(style.opacity) > 0;
                        const hasSize = el.offsetWidth > 0 && el.offsetHeight > 0;
                        const isInViewport = isElementInViewport(el);
                        
                        // Also check if it has any text content
                        const hasContent = getElementText(el).length > 0 || el.getElementsByTagName('img').length > 0;
                        
                        return isInteractive && isVisible && hasSize && isInViewport && hasContent;
                    } catch (e) {
                        return false;
                    }
                };
                
                return Array.from(document.querySelectorAll('*'))
                    .filter(el => {
                        // Safe text extraction with null check
                        const text = (el.innerText || el.textContent || '').toLowerCase().trim();
                        
                        // Priority 1: Direct application buttons/links
                        const directApplyTerms = ['apply now', 'apply for a loan', 'get started', 'start application'];
                        const isDirectApply = directApplyTerms.some(term => text.includes(term));
                        if (isDirectApply && isClickable(el)) return true;
                        
                        // Priority 2: Loan program/product links
                        const loanProgramTerms = ['loan programs', 'our loans', 'loan products', 'explore loans', 
                                                'business loans', 'lending solutions', 'financing options'];
                        const isLoanProgram = loanProgramTerms.some(term => text.includes(term));
                        if (isLoanProgram && isClickable(el)) return true;
                        
                        // Priority 3: URL-based matching for links
                        const href = (el.href || '').toLowerCase();
                        const urlTerms = ['/apply', '/application', '/loans', '/get-started', '/products', '/loan-programs'];
                        const hasRelevantUrl = urlTerms.some(term => href.includes(term));
                        if (hasRelevantUrl && isClickable(el)) return true;
                        
                        // Priority 4: Generic loan-related terms as fallback
                        const genericTerms = ['apply', 'loan', 'funding', 'finance', 'lending'];
                        const isGenericLoan = genericTerms.some(term => text.includes(term));
                        return isGenericLoan && isClickable(el);
                    })
                    .map(el => {
                        // Generate multiple selector options
                        const selectors = [];
                        
                        // Try ID
                        if (el.id) selectors.push('#' + el.id);
                        
                        // Try classes
                        if (el.className) {
                            const classSelector = '.' + el.className.split(' ').join('.');
                            selectors.push(classSelector);
                        }
                        
                        // Try data attributes
                        Array.from(el.attributes).forEach(attr => {
                            if (attr.name.startsWith('data-')) {
                                selectors.push('[' + attr.name + '="' + attr.value + '"]');
                            }
                        });
                        
                        // Try text content
                        const text = el.innerText.trim();
                        if (text) {
                            selectors.push('//*[contains(text(),"' + text + '")]');
                        }
                        
                        // Default to tag name if nothing else works
                        selectors.push(el.tagName.toLowerCase());
                        
                        const rect = el.getBoundingClientRect();
                        return {
                            tag: el.tagName.toLowerCase(),
                            text: text,
                            href: el.href || '',
                            classes: el.className,
                            id: el.id,
                            selectors: selectors,
                            rect: {
                                x: rect.x,
                                y: rect.y,
                                width: rect.width,
                                height: rect.height
                            },
                            styles: {
                                display: window.getComputedStyle(el).display,
                                visibility: window.getComputedStyle(el).visibility,
                                opacity: window.getComputedStyle(el).opacity,
                                position: window.getComputedStyle(el).position,
                                zIndex: window.getComputedStyle(el).zIndex
                            }
                        };
                    });
            })()""")
            
            print("\n🔘 Found clickable elements that might be loan-related:")
            for btn in buttons:
                print("\nPotential loan-related button:")
                print(f"  Text: {btn['text']}")
                print(f"  Tag: {btn['tag']}")
                print(f"  Classes: {btn['classes']}")
                print(f"  ID: {btn['id']}")
                print(f"  Href: {btn.get('href', 'N/A')}")
                print(f"  Position: {btn.get('rect', {})}")
                print(f"  Styles: {btn.get('styles', {})}")
                print("\nTrying multiple selector strategies:")
                
                for selector in btn['selectors']:
                    try:
                        print(f"\nTesting selector: {selector}")
                        element = await page.wait_for_selector(selector, timeout=5000)
                        if element:
                            is_visible = await element.is_visible()
                            is_enabled = await element.is_enabled()
                            bbox = await element.bounding_box()
                            print(f"  Visible: {is_visible}")
                            print(f"  Enabled: {is_enabled}")
                            print(f"  Position: {bbox}")
                            
                            if is_visible and is_enabled and bbox:
                                print("  ✅ Found valid selector!")
                                
                                # Try clicking the element
                                try:
                                    print("  🖱️ Attempting to click...")
                                    await element.click(timeout=5000)
                                    await page.wait_for_load_state("networkidle")
                                    new_url = page.url
                                    print(f"  ✅ Click successful! New URL: {new_url}")
                                    
                                    # Return early with the successful click result
                                    html = await page.content()
                                    snippet = html_to_text(html)[:4000]
                                    print(f"\n📄 Got page content from new page ({len(html)} bytes)")
                                    await browser.close()
                                    return html, snippet
                                    
                                except Exception as e:
                                    print(f"  ❌ Click failed: {str(e)}")
                                    
                    except Exception as e:
                        print(f"  ❌ Selector failed: {str(e)}")
            
            # If we get here, no successful clicks were made
            html = await page.content()
            snippet = html_to_text(html)[:4000]
            print(f"\n📄 Got page content ({len(html)} bytes)")
            
            await browser.close()
            return html, snippet
        except Exception as e:
            print(f"❌ Error loading page: {e}")
            await browser.close()
            raise

async def test_single_site(url: str):
    """Test landing page detection for a single site"""
    print(f"\n{'='*50}")
    print(f"🏢 Testing website: {url}")
    
    # Set up our LangGraph for testing
    llm = GroqClientAsync(model="mixtral-8x7b-32768")
    orchestrator = AsyncLangGraphOrchestrator(llm)
    
    try:
        # Get initial page content and analysis (this will also attempt to find and click buttons)
        print("\n🔍 Analyzing and attempting to find loan-related buttons...")
        html, snippet = await get_page_content(url)
        print(f"\n📃 Initial content analysis complete")
        
        # Build the initial state with the new page content (after potential click)
        state = {
            "url": url,
            "html": html,
            "snippet": snippet,
            "actions": [],
            "forms": [],
            "runtime": {},
            "already_clicked": [],
            "visited_urls": [url],
            "intermediate_visits": 0,
        }
        
        # Run through landing page node to log what it found
        print("\n🤖 Running through landing page node...")
        result = await orchestrator.landing_page_node(state)
        
        print("\n📊 Results:")
        print("-" * 30)
        
        if result and "_new_actions" in result:
            action = result["_new_actions"][0]
            print(f"Action type: {action['type']}")
            
            if action["type"] == "found_loan_application":
                print(f"✅ Found loan application link!")
                print(f"🎯 Using selector: {action.get('selector_used', 'N/A')}")
                if "url" in result:
                    print(f"🔗 Navigation URL: {result['url']}")
                
                # No need to validate with new browser since we already clicked in get_page_content
                return result
            else:
                print(f"❌ No loan application button found")
                if "candidates" in action:
                    print("\nPotential candidates that were considered:")
                    for candidate in action["candidates"]:
                        print(f"- {candidate}")
                return result
        else:
            print("❌ No actions returned from landing page node")
            return None
                                    

        
    except Exception as e:
        print(f"❌ Error during test: {str(e)}")
        return None
    return result

async def main():
    """Run tests for all specified sites"""
    sites = [
        "https://capitalfunding.com/",
        "https://brooklynfundinggroup.com/",
        "https://www.limaone.com/"
    ]
    
    for url in sites:
        try:
            await test_single_site(url)
            # Pause between sites to avoid overwhelming
            await asyncio.sleep(2)
        except Exception as e:
            print(f"❌ Error testing {url}: {e}")
            continue

if __name__ == "__main__":
    # Run the async tests
    asyncio.run(main())
