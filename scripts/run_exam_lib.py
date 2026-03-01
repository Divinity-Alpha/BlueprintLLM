"""
Thin re-export shim for 12_run_exam.py (can't import a module starting with a digit).
"""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "12_run_exam.py"
_spec = importlib.util.spec_from_file_location("_run_exam_mod", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export the functions other scripts need
load_model = _mod.load_model
generate = _mod.generate
validate_dsl = _mod.validate_dsl
compare_outputs = _mod.compare_outputs
GENERATE_TIMEOUT = _mod.GENERATE_TIMEOUT
DSLStoppingCriteria = _mod.DSLStoppingCriteria
