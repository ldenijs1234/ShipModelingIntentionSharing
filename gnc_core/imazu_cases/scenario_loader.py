# Import whatever dictionary name is defined in each file
from gnc_core.imazu_cases.case01.case01_config import CASE01_CONFIG
from gnc_core.imazu_cases.case02.case02_config import CASE02_CONFIG
from gnc_core.imazu_cases.case03.case03_config import CASE03_CONFIG

def load_scenario(name: str):
    cases = {
        "case01": CASE01_CONFIG,
        "case02": CASE02_CONFIG,
        "case03": CASE03_CONFIG,
    }
    if name.lower() not in cases:
        raise ValueError(f"Unknown scenario '{name}'. Available: {list(cases.keys())}")
    return cases[name.lower()]