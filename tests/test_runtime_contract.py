import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_colab_matches_verified_modal_runtime(self):
        notebook = json.loads(
            (ROOT / "colab" / "colab_l4_smoke.ipynb").read_text(encoding="utf-8")
        )
        source = "".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )
        self.assertIn('PINS = ["torch==2.13.0"', source)
        self.assertIn('"BATCH_SIZE": "524288"', source)
        self.assertIn('latest.get("batch_size") == 524288', source)
        self.assertIn('split("+", 1)[0] == "2.13.0"', source)
        self.assertNotIn('PINS = ["torch==2.10.0"', source)

        requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("torch==2.13.0", requirements)
        self.assertNotIn("torch==2.10.0", requirements)

        modal_source = (ROOT / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn('"torch==2.13.0"', modal_source)
        self.assertIn('"BATCH_SIZE": str(524288)', modal_source)

        modal_doc = (ROOT / "MODAL.md").read_text(encoding="utf-8")
        self.assertIn("BATCH_SIZE=524288", modal_doc)
        self.assertNotIn("BATCH_SIZE=65536", modal_doc)

    def test_tracked_registry_contains_strict_json_milestone_record(self):
        records = [
            json.loads(line, parse_constant=lambda value: self.fail(
                f"non-standard JSON constant: {value}"
            ))
            for line in (ROOT / "runs" / "index.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        milestone = next(
            record for record in records
            if record["run_id"] == "21c92807-418e-49ce-a78b-566b376f0914"
        )
        self.assertEqual(milestone["hardware"], "NVIDIA A100-SXM4-40GB")
        self.assertEqual(milestone["steps_completed"], 3250)
        self.assertTrue(math.isclose(milestone["final_val_loss"], 2.8055896759033203))

        smoke = next(
            record for record in records
            if record["run_id"] == "eadfda3f-8ae6-40af-a26a-e6eca74a4206"
        )
        self.assertEqual(smoke["torch_version"], "2.13.0+cu130")
        self.assertEqual(smoke["batch_size"], 524288)
        self.assertEqual(smoke["steps_completed"], 100)
        self.assertFalse(smoke["stopped_early"])
        self.assertFalse(smoke["final_val_loss_nonfinite"])
        self.assertEqual(smoke["final_loader_state"]["batches"], 100)

        corrected_stop = next(
            record for record in records
            if record["run_id"] == "18f3c915-f7a1-4573-af4b-847783cedfa8"
        )
        self.assertTrue(corrected_stop["historical_record_corrected"])
        self.assertEqual(corrected_stop["original_steps_completed"], 3250)
        self.assertEqual(corrected_stop["steps_completed"], 1125)
        self.assertEqual(corrected_stop["final_loader_state"]["batches"], 1125)


if __name__ == "__main__":
    unittest.main()
