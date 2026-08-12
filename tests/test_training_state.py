import os
import tempfile
import time
import unittest
from pathlib import Path

from config import config_hash
from training_state import (
    checkpoint_sidecar_path,
    completion_metadata,
    copy_sample_log_through_step,
    distributed_batch_granule,
    effective_training_overrides,
    latest_checkpoint,
    validate_loader_step,
    validate_resume_metadata,
)


FORMAT = 3
SIZING = {
    "batch_size": 8192,
    "mbs": 1,
    "seq_len": 1024,
    "train_steps": 6,
    "val_tokens": 8192,
}
IDENTITIES = {
    "config_hash": "config",
    "data_manifest_hash": "data",
    "shard_manifest_hash": "shards",
}


class DistributedSizingTests(unittest.TestCase):
    def test_global_granule_accounts_for_every_rank(self):
        self.assertEqual(distributed_batch_granule(1024, 8, 1), 8192)
        self.assertEqual(distributed_batch_granule(1024, 8, 8), 65536)

    def test_global_granule_rejects_nonpositive_dimensions(self):
        with self.assertRaisesRegex(ValueError, "must all be positive"):
            distributed_batch_granule(1024, 8, 0)


def metadata(**overrides):
    value = {
        "format": FORMAT,
        "step": 3,
        "world_size": 2,
        "loader_state": {"file_idx": 0, "pos": 24576, "batches": 3},
        "sizing": dict(SIZING),
        **IDENTITIES,
    }
    value.update(overrides)
    return value


def sample_block(step, suffix=""):
    return (
        f"\n{'=' * 72}\nStep {step}{suffix}\n{'=' * 72}\n"
        f"sample-{step}\n"
    )


class CheckpointDiscoveryTests(unittest.TestCase):
    def test_recursive_auto_discovery_ignores_rank_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            older = root / "old" / "ckpt_00009.pt"
            newer = root / "new" / "ckpt_00003.pt"
            sidecar = root / "new" / "ckpt_99999.rank00000.pt"
            for path in (older, newer, sidecar):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            now = time.time()
            os.utime(older, (now - 20, now - 20))
            os.utime(newer, (now - 10, now - 10))
            os.utime(sidecar, (now, now))

            self.assertEqual(latest_checkpoint(root, recursive=True), newer)
            self.assertEqual(
                checkpoint_sidecar_path(newer, 7),
                newer.with_name("ckpt_00003.rank00007.pt"),
            )


class ResumeMetadataTests(unittest.TestCase):
    def test_accepts_exact_ranked_checkpoint(self):
        self.assertEqual(
            validate_resume_metadata(
                metadata(), current_format=FORMAT, current_world_size=2,
                sizing=SIZING, identities=IDENTITIES,
            ),
            FORMAT,
        )

    def test_rejects_identity_drift(self):
        for key in IDENTITIES:
            with self.subTest(key=key), self.assertRaisesRegex(
                ValueError, "identity changed"
            ):
                validate_resume_metadata(
                    metadata(**{key: "different"}), current_format=FORMAT,
                    current_world_size=2, sizing=SIZING, identities=IDENTITIES,
                )

    def test_rejects_world_size_change(self):
        with self.assertRaisesRegex(ValueError, "world_size changed"):
            validate_resume_metadata(
                metadata(), current_format=FORMAT, current_world_size=1,
                sizing=SIZING, identities=IDENTITIES,
            )

    def test_legacy_format_two_is_single_rank_only(self):
        legacy = metadata(format=2)
        legacy.pop("world_size")
        self.assertEqual(
            validate_resume_metadata(
                legacy, current_format=FORMAT, current_world_size=1,
                sizing=SIZING, identities=IDENTITIES,
            ),
            2,
        )
        with self.assertRaisesRegex(ValueError, "world_size changed"):
            validate_resume_metadata(
                legacy, current_format=FORMAT, current_world_size=2,
                sizing=SIZING, identities=IDENTITIES,
            )

    def test_rejects_step_ahead_of_loader(self):
        with self.assertRaisesRegex(ValueError, "loader batches=3"):
            validate_loader_step(3, {"batches": 2})

    def test_effective_runtime_overrides_participate_in_config_hash(self):
        base = config_hash()
        effective = config_hash({"batch_size": 8192, "compile": False})
        self.assertNotEqual(base, effective)
        self.assertEqual(
            effective, config_hash({"batch_size": 8192, "compile": False})
        )
        self.assertNotEqual(
            config_hash({"torch_version": "2.10.0"}),
            config_hash({"torch_version": "2.13.0"}),
        )

    def test_runtime_identity_includes_launcher_controls(self):
        values = effective_training_overrides(
            batch_size=524288,
            mbs=8,
            seq_len=1024,
            train_steps=3250,
            val_tokens=4333568,
            sample_every=250,
            checkpoint_every=250,
            val_every=125,
            compile_model=True,
            adamw_fused=False,
            muon_compile=False,
            init_seed=7,
            world_size=1,
            torch_version="2.13.0+cu130",
            torch_cuda_version="13.0",
        )
        self.assertEqual(values["init_seed"], 7)
        self.assertIs(values["adamw_fused"], False)
        self.assertIs(values["muon_compile"], False)
        self.assertEqual(values["val_every"], 125)
        self.assertNotEqual(
            config_hash(values), config_hash({**values, "init_seed": 8})
        )


class CompletionMetadataTests(unittest.TestCase):
    def test_stop_after_records_actual_progress(self):
        result = completion_metadata(
            completed_step=1125,
            train_steps=3250,
            batch_size=524288,
            train_tokens_available=462267303,
            final_val_loss=3.1,
            final_val_step=1125,
        )
        self.assertEqual(result["steps_completed"], 1125)
        self.assertTrue(result["stopped_early"])
        self.assertEqual(result["final_val_step"], 1125)
        self.assertEqual(result["epochs_over_corpus"], 1.28)

    def test_nonfinite_loss_is_strict_json_safe(self):
        result = completion_metadata(
            completed_step=3,
            train_steps=3,
            batch_size=8192,
            train_tokens_available=100000,
            final_val_loss=float("nan"),
            final_val_step=3,
        )
        self.assertIsNone(result["final_val_loss"])
        self.assertTrue(result["final_val_loss_nonfinite"])
        self.assertFalse(result["stopped_early"])

    def test_rejects_impossible_completion_step(self):
        with self.assertRaisesRegex(ValueError, "completed_step"):
            completion_metadata(
                completed_step=7,
                train_steps=6,
                batch_size=8192,
                train_tokens_available=100000,
                final_val_loss=None,
                final_val_step=None,
            )


class SampleTrajectoryTests(unittest.TestCase):
    def test_copies_only_blocks_at_or_before_resume_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.log"
            destination = Path(tmp) / "destination.log"
            expected = sample_block(250) + sample_block(500)
            source.write_text(expected + sample_block(750, "  (+ held-out)"),
                              encoding="utf-8")

            copied = copy_sample_log_through_step(source, destination, 500)

            self.assertEqual(copied, 2)
            text = destination.read_text(encoding="utf-8")
            self.assertEqual(text, expected)
            self.assertIn("Step 250", text)
            self.assertIn("Step 500", text)
            self.assertNotIn("Step 750", text)

    def test_drops_a_future_only_log_without_leaving_a_blank_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.log"
            destination = Path(tmp) / "destination.log"
            source.write_text(sample_block(6, "  (+ held-out)"), encoding="utf-8")

            copied = copy_sample_log_through_step(source, destination, 3)

            self.assertEqual(copied, 0)
            self.assertEqual(destination.read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
