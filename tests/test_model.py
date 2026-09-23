import sys
import unittest
from pprint import pprint
from pathlib import Path
from unittest.mock import patch


SCANDL2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SCANDL2_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ScanDL2SmokeTests(unittest.TestCase):
    def test_sentence_model_runs_real_inference(self):
        _skip_if_sentence_assets_are_missing()

        from ScanDL2.model import ScanDL2

        try:
            with patch.object(sys, "argv", [sys.argv[0]]):
                model = ScanDL2(text_type="sentence", bsz=1, save=None, filename=None)
                model.eval()
                output = model(["The quick brown fox jumps."])
        except PermissionError as exc:
            raise unittest.SkipTest(f"Real ScanDL2 smoke test needs socket access: {exc}") from exc

        print("\nScanDL2 output:")
        pprint(output)
        self.assertIsInstance(output, dict)
        self.assertIn("predicted_sp_words", output)
        self.assertIn("predicted_sp_ids", output)
        self.assertIn("original_sn", output)
        self.assertIn("predicted_fix_durs", output)
        self.assertIn("unique_idx", output)

    def test_scandl_and_fixdur_modules_run_real_inference(self):
        _skip_if_sentence_assets_are_missing()

        from ScanDL2.model import FixdurModule, ScanDLModule

        try:
            with patch.object(sys, "argv", [sys.argv[0]]):
                scandl_module = ScanDLModule(text_type="sentence", bsz=1)
                scandl_output = scandl_module(texts=["The quick brown fox jumps."])

                fixdur_module = FixdurModule(text_type="sentence", bsz=1)
                fixdur_output = fixdur_module(scandl_module_output=scandl_output)
        except PermissionError as exc:
            raise unittest.SkipTest(f"Real ScanDL2 smoke test needs socket access: {exc}") from exc

        print("\nScanDLModule output:")
        pprint(scandl_output)
        print("\nFixdurModule output:")
        pprint(fixdur_output)

        self.assertIsInstance(scandl_output, dict)
        self.assertIn("predicted_sp_words", scandl_output)
        self.assertIn("predicted_sp_ids", scandl_output)
        self.assertIn("original_sn", scandl_output)
        self.assertIn("unique_idx", scandl_output)

        self.assertIsInstance(fixdur_output, dict)
        self.assertIn("predicted_sp_words", fixdur_output)
        self.assertIn("predicted_sp_ids", fixdur_output)
        self.assertIn("original_sn", fixdur_output)
        self.assertIn("predicted_fix_durs", fixdur_output)
        self.assertIn("unique_idx", fixdur_output)


def _skip_if_sentence_assets_are_missing():
    required_dirs = [
        SCANDL2_ROOT / "models" / "sentence" / "scandl-module",
        SCANDL2_ROOT / "models" / "sentence" / "fixdur-module",
    ]
    missing = [path for path in required_dirs if not path.exists() or not any(path.iterdir())]
    if missing:
        raise unittest.SkipTest(
            "Missing ScanDL2 sentence model assets: "
            + ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        )


if __name__ == "__main__":
    unittest.main()
