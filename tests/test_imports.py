import ast
import sys
import unittest
from pathlib import Path


SCANDL2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCANDL2_ROOT.parent
EXCLUDED_DIRS = {"__pycache__", ".git", "tests"}
EXPECTED_IMPORT_FAILURES = {
    ("ScanDL2/app.py", "import gradio as gr"),
    ("ScanDL2/create_data.py", "from ScanDL2.scandl_module.scripts.sp_load_celer_zuco import load_emtec, process_emtec"),
    ("ScanDL2/create_data.py", "from ScanDL2.scandl_module.scripts.sp_load_celer_zuco import load_bsc, process_bsc"),
    ("ScanDL2/fix_dur_module/train_seq2seq.py", "from ScanDL2.CONSTANTS import COMPLETE_FIXDUR_MODULE_TRAIN_PATH_BSC"),
    ("ScanDL2/scandl_module/scripts/sp_run_train.py", "from ScanDL2.CONSTANTS import (\n    COMPLETE_SCANDL_MODULE_TRAIN_PATH_BSC"),
    ("ScanDL2/scandl_module/original_scandl/utils/logger.py", "import tensorflow as tf"),
    ("ScanDL2/scandl_module/original_scandl/utils/logger.py", "from tensorflow.python import pywrap_tensorflow"),
    ("ScanDL2/scandl_module/original_scandl/utils/logger.py", "from tensorflow.core.util import event_pb2"),
    ("ScanDL2/scandl_module/original_scandl/utils/logger.py", "from tensorflow.python.util import compat"),
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ScanDL2ImportTests(unittest.TestCase):
    def test_import_lines(self):
        failures = []

        for file_path in _python_files(SCANDL2_ROOT):
            source = file_path.read_text()
            tree = ast.parse(source, filename=str(file_path))

            for node in ast.walk(tree):
                if not isinstance(node, (ast.Import, ast.ImportFrom)):
                    continue
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue

                import_line = ast.get_source_segment(source, node)
                try:
                    exec(
                        compile(import_line, str(file_path), "exec"),
                        _import_globals(file_path),
                    )
                except Exception as exc:
                    relative_path = str(file_path.relative_to(PROJECT_ROOT))
                    if _is_expected_failure(relative_path, import_line):
                        continue
                    failures.append(
                        f"{relative_path}:{node.lineno}\n"
                        f"{import_line}\n"
                        f"{type(exc).__name__}: {exc}"
                    )

        if failures:
            self.fail("Failed import line(s):\n\n" + "\n\n".join(failures))


def _python_files(root):
    for file_path in root.rglob("*.py"):
        if any(part in EXCLUDED_DIRS for part in file_path.parts):
            continue
        yield file_path


def _import_globals(file_path):
    module_path = file_path.relative_to(PROJECT_ROOT).with_suffix("")
    module_parts = module_path.parts

    if module_parts[-1] == "__init__":
        module_name = ".".join(module_parts[:-1])
        package = module_name
    else:
        module_name = ".".join(module_parts)
        package = ".".join(module_parts[:-1])

    return {
        "__name__": module_name,
        "__package__": package,
    }


def _is_expected_failure(relative_path, import_line):
    return any(
        relative_path == expected_path and import_line.startswith(expected_import)
        for expected_path, expected_import in EXPECTED_IMPORT_FAILURES
    )


if __name__ == "__main__":
    unittest.main()
