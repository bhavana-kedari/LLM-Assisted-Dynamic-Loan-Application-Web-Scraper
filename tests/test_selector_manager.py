from web_selectors.selector_manager import SelectorManager

def test_generate_selector_from_attrs():
    sm = SelectorManager()
    sel = sm.generate_selector_from_attrs("button", {"data-action": "apply", "class": "cta"})
    assert "data-action" in sel or "cta" in sel

def test_rank_selectors():
    sm = SelectorManager()
    cands = [
        {"selector": "button[data-action='apply']"},
        {"selector": "#apply-now"},
        {"selector": "a.apply-now-btn.some-long-class-names"}
    ]
    ranked = sm.rank_selectors(cands)
    assert ranked
    top = ranked[0]
    assert "selector" in top
