import logging
import json
from pathlib import Path
import asyncio
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from playwright.async_api import async_playwright
from llm.groq_client import GroqClientAsync  # async variant
from web_selectors.selector_manager import SelectorManager
from utils.parser import html_to_text, extract_forms
from utils.io import save_json_atomic
from typing import Annotated
import operator
import csv

logger = logging.getLogger(__name__)


def save_form_fields_csv(forms_data, csv_path: Path):
    rows = []
    for form in forms_data:
        fields = form.get("parsed_forms", [])
        for field in fields:
            name = field.get("name") or field.get("label") or "unknown"
            ftype = field.get("type", "unknown")
            options = field.get("options", [])
            options_str = ", ".join(options) if options else "N/A"
            rows.append([name, ftype, options_str])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field_name", "type", "options"])
        writer.writerows(rows)


def overwrite(last, new):
    return new


class GraphState(TypedDict, total=False):
    url: Annotated[str, overwrite]
    snippet: Annotated[str, overwrite]
    classification: Annotated[str, overwrite]
    classification_llm: Annotated[Dict[str, Any], overwrite]  
    intermediate_llm: Annotated[Dict[str, Any], overwrite]    
    form_llm: Annotated[Dict[str, Any], overwrite]           
    actions: Annotated[List[Dict[str, Any]], operator.add]
    _new_actions: Annotated[List[Dict[str, Any]], operator.add]
    forms: Annotated[List[Dict[str, Any]], operator.add]    
    human_confirm: Annotated[Dict[str, Any], overwrite]      
    runtime: Annotated[Dict[str, Any], operator.ior]
    already_clicked: Annotated[List[str], operator.add]     
    visited_urls: Annotated[List[str], operator.add]         
    intermediate_visits: Annotated[int, operator.add]  


class AsyncLangGraphOrchestrator:
    def __init__(self, groq: GroqClientAsync, human_in_loop: bool = False):
        self.groq = groq
        self.human_in_loop = human_in_loop
        self.selector_manager = SelectorManager()
        
        # Cache for LLM-generated selectors to avoid repeated calls
        self.selector_cache = {}

        # Initialize the graph
        builder = StateGraph(GraphState)

        # Nodes
        builder.add_node(self.classify_page_node)
        builder.add_node(self.landing_page_node)
        builder.add_node(self.intermediate_page_node)
        builder.add_node(self.simulate_form_fill_node)
        builder.add_node(self.extract_form_fields_node)
        builder.add_node(self.find_submit_or_next_button_node)
        builder.add_node(self.merge_actions_node)
        builder.add_node(self.end_node)

        # Flow
        builder.add_edge(START, "landing_page_node")
        builder.add_edge("landing_page_node", "classify_page_node")

        builder.add_conditional_edges(
            "classify_page_node",
            lambda state: (
                "end_node" if state.get("intermediate_visits", 0) >= 2 
                else "intermediate_page_node" if state.get("classification") == "intermediate" 
                else "extract_form_fields_node"
            ),
            {
                "intermediate_page_node": "intermediate_page_node",
                "extract_form_fields_node": "extract_form_fields_node",
                "end_node": "end_node"
            }
        )
        
        # After intermediate page, go back to classify the next page
        builder.add_edge("intermediate_page_node", "classify_page_node")
        
        # Form page flow - conditional based on multi-step
        builder.add_conditional_edges(
            "extract_form_fields_node",
            lambda state: "simulate_form_fill_node" if state.get("form_llm", {}).get("multi_step") else "end_node",
            {
                "simulate_form_fill_node": "simulate_form_fill_node",
                "end_node": "end_node"
            }
        )
        
        # After simulating form fill, go back to extract fields from next form page
        builder.add_edge("simulate_form_fill_node", "extract_form_fields_node")
        builder.add_edge("end_node", END)

        self.graph = builder.compile()
        
    async def _get_llm_selectors(self, html: str, button_text: str) -> List[Dict[str, str]]:
        """Get selectors from LLM for a given button text and HTML context"""
        
        cache_key = f"{button_text}"
        if cache_key in self.selector_cache:
            return self.selector_cache[cache_key]
            
        # Prepare prompt for the LLM
        prompt = {
            "role": "user",
            "content": f"""Given this button text: "{button_text}"
            Generate effective CSS and XPath selectors to find the button element. The selectors should be robust and consider different potential HTML structures.
            Consider using multiple approaches:
            1. Text-based selectors (exact and partial matches)
            2. Button/link specific selectors
            3. Role-based selectors
            4. Class/ID based selectors if consistent patterns are found
            5. Parent-child relationships if helpful

            Format your response as a JSON array of objects with 'selector' and 'type' fields. Example:
            [
                {{"selector": "//button[contains(text(), 'Apply Now')]", "type": "xpath"}},
                {{"selector": "button:has-text('Apply Now')", "type": "css"}}
            ]

            Focus on these selector patterns:
            - XPath: exact text match: //button[text()='{button_text}']
            - XPath: contains text: //button[contains(text(), '{button_text}')]
            - XPath: normalize-space: //button[normalize-space()='{button_text}']
            - XPath: case-insensitive: //button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'{button_text.lower()}')]
            - CSS: Playwright specific: button:has-text('{button_text}')
            - CSS: with role: [role='button']:has-text('{button_text}')
            - Combined: Multiple attributes: //button[@type='submit'][contains(text(), '{button_text}')]"""}
        
        try:
            llm_result = await self.groq.raw_completion({"messages": [prompt]})
            selectors = json.loads(llm_result['choices'][0]['message']['content'])
            
            # Transform the selectors into our expected format
            formatted_selectors = []
            for sel in selectors:
                formatted_selectors.append({
                    "selector": sel["selector"],
                    "text": button_text,
                    "source": "llm",
                    "selector_type": sel["type"]
                })
            
            # Cache the results
            self.selector_cache[cache_key] = formatted_selectors
            return formatted_selectors
        except Exception as e:
            logger.error(f"Error getting LLM selectors: {str(e)}")
            return []


    async def invoke_start(self, url: str, html: str, snippet: str) -> Dict[str, Any]:
        state: GraphState = {
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

        while True:
            result = await self.graph.ainvoke(state)
            state.update(result)
            classification = state.get("classification", "unknown")
            logger.info("Classification: %s", classification)

            current_url = state.get("url")
            if current_url and current_url not in state["visited_urls"]:
                state["visited_urls"].append(current_url)

            if classification in ("intermediate", "simulate_fill") and state.get("html"):
                continue
            if classification == "form":
                save_json_atomic(state.get("runtime", {}), Path("data") / "runtime_snapshot.json")
                return state
            # If we have a found_loan_application action, stop processing
            if any(action.get("type") == "found_loan_application" for action in state.get("_new_actions", [])):
                return state
            return state

    async def classify_page_node(self, state: GraphState) -> GraphState:
        print("[TRACE] Entering classify_page_node")
        snippet = state.get("snippet", "")
        url = state.get("url", "")
        logger.info("Classifying page: %s", url)
        llm_result = await self.groq.classify_form_or_intermediate(snippet=snippet, url=url)
        page_type = llm_result.get("page_type")
        
        if page_type not in ["intermediate", "form"]:
            logger.warning(f"Unknown page type: {page_type}, defaulting to intermediate")
            page_type = "intermediate"
            
        return {
            "classification": page_type,
            "classification_llm": llm_result,
        }

    async def landing_page_node(self, state: GraphState) -> GraphState:
        print("[TRACE] Entering landing_page_node")
        snippet = state.get("snippet", "")
        # Look specifically for loan application navigation buttons
        llm_actions = await self.groq.find_loan_application_nav(snippet=snippet)
        nav_buttons = llm_actions.get("loan_application_buttons", [])
        nav_result = await self._try_selectors(state.get("url"), nav_buttons, state)

        if nav_result:
            # Get what we need before cleaning up
            url = nav_result["url"]
            html = nav_result["html"]
            selector = nav_result["selector"]
            
            # Clean up browser if it exists
            if "browser" in nav_result:
                try:
                    await nav_result["browser"].close()
                except Exception:
                    pass

            return {
                "url": url,
                "html": html,
                "snippet": html_to_text(html)[:4000],
                "_new_actions": [{"type": "found_loan_application", "selector_used": selector}],
            }
        else:
            logger.warning("Could not find loan application button in navigation")
            return {"_new_actions": [{"type": "loan_application_not_found", "candidates": nav_buttons}]}

    async def intermediate_page_node(self, state: GraphState) -> GraphState:
        print("[TRACE] Entering intermediate_page_node")

        # Increment the intermediate page visit counter
        intermediate_visits = state.get("intermediate_visits", 0) + 1
        snippet = state.get("snippet", "")

        # Get options
        llm_resp = await self.groq.analyze_intermediate_options(snippet=snippet)
        current_url = state.get("current_url") or state.get("url")

        if self.human_in_loop:
            # Present options to human
            options = llm_resp.get("options", [])
            print("\n=== Available Loan Options ===")
            print("Please choose one of the following options by entering its number:")

            recommended = [opt for opt in options if opt.get("recommended", False)]
            others = [opt for opt in options if not opt.get("recommended", False)]
            sorted_options = recommended + others

            for i, opt in enumerate(sorted_options, 1):
                print(f"\n{i}. {opt.get('text', '')}")
                print(f"   Description: {opt.get('description', 'No description available')}")
                if opt.get("recommended"):
                    print("   ✨ [Recommended Option]")

            choice = input(f"\nEnter a number (1-{len(sorted_options)}) or 'q' to quit: ").strip().lower()

            if choice == 'q':
                logger.info("User chose to quit")
                return {"_new_actions": [{"type": "human_cancelled"}]}

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(sorted_options):
                    selected = sorted_options[idx]
                    logger.info(f"User selected option: {selected.get('text')}")
                    nav_result = await self._try_selectors(current_url, [selected], state)
                else:
                    logger.error(f"Invalid choice number: {choice}")
                    return {"_new_actions": [{"type": "human_choice_invalid"}]}
            except (ValueError, IndexError) as e:
                logger.error(f"Invalid input: {choice} - {str(e)}")
                return {"_new_actions": [{"type": "human_choice_invalid"}]}

        else:
            # Auto-select path (LLM)
            nav_result = await self._try_selectors(current_url, llm_resp.get("options", []), state)

        if nav_result:
            url = nav_result["url"]
            html = nav_result["html"]
            selector = nav_result["selector"]

            if "browser" in nav_result:
                try:
                    await nav_result["browser"].close()
                except Exception:
                    pass

            return {
                "url": url,
                "html": html,
                "snippet": html_to_text(html)[:4000],
                "_new_actions": [{"type": "intermediate_option_selected", "selector_used": selector}],
                "intermediate_visits": intermediate_visits,
                "current_url": url,  # persist new location
            }
        else:
            return {
                "_new_actions": [{"type": "no_option_selected"}],
                "intermediate_visits": intermediate_visits,
            }


    async def simulate_form_fill_node(self, state: GraphState) -> GraphState:
        print(f"[TRACE] Entering simulate_form_fill_node")
        html = state.get("html", "")
        snippet = state.get("snippet", "")
        llm_resp = await self.groq.find_next_or_submit_button(html = html,snippet=snippet)
        next_selectors = llm_resp.get("next_selectors", [])
        nav_result = await self._try_selectors(state.get("url"), next_selectors, state)

        if nav_result:
            # Get what we need before cleaning up
            url = nav_result["url"]
            html = nav_result["html"]
            selector = nav_result["selector"]
            
            # Clean up browser if it exists
            if "browser" in nav_result:
                try:
                    await nav_result["browser"].close()
                except Exception:
                    pass

            return {
                "url": url,
                "html": html,
                "snippet": html_to_text(html)[:4000],
                "_new_actions": [{"type": "form_next_click", "selector_used": selector}],
            }
        else:
            return {
                "html": state.get("html", ""),  # Keep existing HTML if navigation fails
                "_new_actions": [{"type": "form_next_no_nav", "candidates": next_selectors}]
            }

    async def extract_form_fields_node(self, state: GraphState) -> GraphState:
        print(f"[TRACE] Entering Extract_form_fields_node")
        snippet = state.get("snippet", "")
        llm_resp = await self.groq.extract_form_fields(snippet=snippet)
        parsed_forms = extract_forms(state.get("html", ""))
        forms = state.get("forms", []) + [{"form_llm": llm_resp, "parsed_forms": parsed_forms}]
        save_form_fields_csv(forms, Path("data") / "form_fields.csv")
        return {
            "forms": forms,
            "form_llm": llm_resp,
            "_new_actions": [{"type": "form_extracted", "multi_step": llm_resp.get("multi_step")}],
        }

    async def find_submit_or_next_button_node(self, state: GraphState):
        # Add safety checks for required state
        print(f"[TRACE] Entering find_submit_or_next_button_node")
        snippet = state.get("snippet", "")
        llm_resp = await self.groq.find_next_or_submit_button(snippet=snippet)
        html = state.get("html")
        snippet = state.get("snippet")
        
        if not html or not snippet:
            logger.warning("Missing required state keys (html or snippet) in find_submit_or_next_button_node")
            return state
        
        # Now safely use the values we got
        page_type, selectors = await self.groq.find_next_or_submit_button(html, snippet)

        if page_type == "submit":
            logger.info("Submit button found. Ending process without clicking.")
            state.setdefault("_new_actions", []).append({
                "action": "submit_detected",
                "url": state.get("url")  # Changed from current_url to url to match state keys
            })
            state["clicked_submit"] = True
            state["classification"] = "end"
            return state

        elif page_type == "next":
            logger.info("Next button found. Attempting click.")
            nav_result = await self._try_selectors(state.get("url"), selectors, state)  # Changed from current_url to url
            if nav_result:
                state["url"] = nav_result["url"]
                state["html"] = nav_result["html"]
                state.setdefault("_new_actions", []).append({
                    "action": "next_clicked",
                    "selector": nav_result["selector"],
                    "url": nav_result["url"]
                })
            return state

        logger.warning("No next/submit button found.")
        return state


    async def merge_actions_node(self, state: GraphState) -> GraphState:
        # Return new actions directly to let LangGraph handle the merging
        print(f"[TRACE] Entering merge_actions_node")
        return {
            "_new_actions": state.get("_new_actions", []),
            "actions": state.get("_new_actions", [])  # This will be merged with existing actions via operator.add
        }

    async def end_node(self, state: GraphState) -> GraphState:
        print(f"[TRACE] Entering end_node")
        runtime = state.get("runtime", {})
        runtime.update({"final_url": state.get("url")})
        save_json_atomic(runtime, Path("data") / "runtime_end_snapshot.json")
        return {"runtime": runtime}

    async def _try_selectors(self, url: str, selectors: List[Dict[str, Any]], state: GraphState) -> Optional[Dict[str, Any]]:
        print(f"[TRACE] Entering _try_selectors")
        if not url or not selectors:
            return None

        already_clicked = set(state.get("already_clicked", []))
        visited_urls = set(state.get("visited_urls", []))
        
        # Filter out already clicked selectors
        selectors = [s for s in selectors if s.get("selector") not in already_clicked]
        if not selectors:
            logger.info("All suggested selectors already clicked, skipping navigation.")
            return None

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)  # Show browser window
            context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
            page = await context.new_page()
            
            try:
                # Ensure full screen mode
                await page.set_viewport_size({'width': 1920, 'height': 1080})
                await page.evaluate("document.documentElement.requestFullscreen()")
                
                logger.info(f"Loading URL: {url}")
                try:
                    # First attempt with networkidle
                    await page.goto(url, wait_until="networkidle", timeout=60000)  # 60 second timeout
                    logger.info("Page loaded successfully with networkidle")
                except Exception as e:
                    logger.warning(f"Network idle timeout, falling back to domcontentloaded: {str(e)}")
                    # Fallback to domcontentloaded
                    await page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    # Wait additional time for any dynamic content
                    await asyncio.sleep(5)
                    logger.info("Page loaded successfully with fallback method")
            except Exception as e:
                logger.error(f"Initial page load failed: {str(e)}")
                await browser.close()
                return None

            # Get HTML content for LLM analysis
            html = await page.content()
            
            # Rank selectors using SelectorManager
            ranked_selectors = self.selector_manager.rank_selectors(selectors)
            
            for s in ranked_selectors:
                # Generate multiple selector strategies for each element
                element_selectors = []
                text = s.get("text", "") if isinstance(s, dict) else ""
                
                # Get additional selectors from LLM if we have text
                if text:
                    llm_selectors = await self._get_llm_selectors(html, text)
                    element_selectors.extend(llm_selectors)
                
                # Original selector
                base_sel = s.get("selector") if isinstance(s, dict) else s
                if base_sel:
                    element_selectors.append({"selector": base_sel, "text": text})
                
                    # Text-based selectors with varying specificity
                if text:
                    # XPath text match with element type hints
                    element_selectors.extend([
                        {"selector": f'//button[contains(text(),"{text}")]', "text": text},
                        {"selector": f'//a[contains(text(),"{text}")]', "text": text},
                        {"selector": f'//input[@value="{text}"]', "text": text},
                        {"selector": f'//*[text()="{text}"]', "text": text},
                    ])
                    
                    # CSS selectors for common button/link patterns
                    element_selectors.extend([
                        {"selector": f'a:has-text("{text}")', "text": text},
                        {"selector": f'button:has-text("{text}")', "text": text},
                        {"selector": f'[role="button"]:has-text("{text}")', "text": text},
                    ])
                    
                    # Additional text-matching selectors
                    element_selectors.extend([
                        {"selector": f'//*[contains(text(),"{text}")]', "text": text},
                        {"selector": f'//*[normalize-space()="{text}"]', "text": text},
                        {"selector": f'[title="{text}"]', "text": text},
                    ])                # Add test-specific selectors
                if isinstance(s, dict):
                    for attr, value in s.items():
                        if attr in ["data-testid", "data-test", "data-qa"]:
                            element_selectors.append(
                                {"selector": f'[{attr}="{value}"]', "text": text}
                            )
                
                # Add accessibility selectors
                if isinstance(s, dict):
                    for attr in ["aria-label", "role", "title"]:
                        if attr in s:
                            element_selectors.append(
                                {"selector": f'[{attr}="{s[attr]}"]', "text": text}
                            )
                            
                # Add form-specific selectors
                if isinstance(s, dict):
                    for attr in ["name", "for", "placeholder"]:
                        if attr in s:
                            element_selectors.append(
                                {"selector": f'[{attr}="{s[attr]}"]', "text": text}
                            )
                
                # Add data attribute selectors
                if isinstance(s, dict):
                    for attr, value in s.items():
                        if attr.startswith("data-"):
                            element_selectors.append(
                                {"selector": f'[{attr}="{value}"]', "text": text}
                            )
                
                # Add compound selectors for better specificity
                if text and isinstance(s, dict) and "role" in s:
                    element_selectors.append({
                        "selector": f'[role="{s["role"]}"][text()="{text}"]',
                        "text": text
                    })
                
                # Try each selector strategy in order of their quality
                sorted_selectors = self.selector_manager.rank_selectors(element_selectors)
                for sel_dict in sorted_selectors:
                    sel = sel_dict["selector"]
                    if sel in already_clicked:
                        logger.info(f"Selector already clicked: {sel}")
                        continue
                        
                    try:
                        logger.info(f"Trying selector: {sel}")
                        element = await page.wait_for_selector(sel, timeout=5000)
                        
                        if element:
                            # Validate element state
                            is_visible = await element.is_visible()
                            is_enabled = await element.is_enabled()
                            bbox = await element.bounding_box()
                            
                            logger.info(f"Element state - Visible: {is_visible}, Enabled: {is_enabled}, Position: {bbox}")
                            
                            if is_visible and is_enabled and bbox:
                                # Check if element might lead to an already visited URL
                                href = await element.get_attribute("href")
                                if href:
                                    # Resolve relative URLs
                                    from urllib.parse import urljoin
                                    full_href = urljoin(url, href)
                                    if full_href in visited_urls:
                                        logger.info(f"Skipping selector '{sel}' - target URL already visited: {full_href}")
                                        continue
                                
                                # Try clicking
                                logger.info(f"Attempting to click element with selector: {sel}")
                                
                                # Store the current URL before clicking
                                pre_click_url = page.url
                                
                                # Click and wait for any navigation
                                await element.click(timeout=5000)
                                try:
                                    # Wait for either a navigation or network idle
                                    await page.wait_for_load_state("networkidle", timeout=5000)
                                except Exception:
                                    # If no navigation occurs, that's okay
                                    pass
                                
                                # Get the new URL after clicking
                                nav_url = page.url
                                
                                # If URL changed, we've had a successful navigation
                                if nav_url != pre_click_url:
                                    # Get the updated content
                                    html = await page.content()
                                    
                                    state.setdefault("already_clicked", []).append(sel)
                                    state.setdefault("visited_urls", []).append(nav_url)
                                    
                                    logger.info(f"Successfully clicked element. New URL: {nav_url}")
                                    
                            
                                    state["current_url"] = nav_url
                                    
                                    return {
                                        "url": nav_url,
                                        "html": html,
                                        "selector": sel,
                                        "browser": browser,
                                        "page": page
                                    }
                                else:
                                    logger.info(f"Click didn't result in navigation, trying next selector")
                                
                    except Exception as e:
                        logger.error(f"Error with selector '{sel}': {str(e)}")
                        continue

            await browser.close()
            logger.info("No successful clicks, closing browser")
        return None 
