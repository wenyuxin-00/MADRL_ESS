from pathlib import Path

import package
import scripts.export_project_code as export_project_code
import scripts.run_debug_training as run_debug_training
import scripts.train_smoke as train_smoke


def test_export_project_code_entrypoints_share_same_function():
    assert package.export_project_code is export_project_code.export_project_code
    assert export_project_code.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert export_project_code.OUTPUT_FILE == export_project_code.PROJECT_ROOT / "full_project_code.txt"


def test_train_smoke_wrapper_reuses_debug_training_main():
    assert train_smoke.main is run_debug_training.main
