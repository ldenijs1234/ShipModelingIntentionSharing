"""
scenario_loader.py
Dynamic importer for benchmark encounter scenarios.
"""

import importlib
from typing import Dict, Any

def load_scenario(scenario_name: str) -> Dict[str, Any]:
    """Dynamically imports and returns the CASE_CONFIG for a given scenario name."""
    module_path = f"gnc_core.imazu_cases.{scenario_name}.{scenario_name}_config"
    try:
        scenario_module = importlib.import_module(module_path)
        return scenario_module.CASE_CONFIG
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Scenario '{scenario_name}' not found. "
            f"Ensure '{module_path}.py' exists."
        ) from e