import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from baseline_model import GPT  # noqa: E402
from evals.exemplar import EOT_TOKEN  # noqa: E402
from evals.register_extract import (  # noqa: E402
    acts_contexts,
    capture_block_outputs,
    context_layer_means,
    layer_covariance,
    offtarget_windows,
    save_covariance,
)
from evals.exemplar import Passage  # noqa: E402


SHARD_MAGIC = 20240520
SHARD_VERSION = 1


def write_tiny_shard(path: Path, tokens: list[int]) -> None:
    header = [0] * 256
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    payload = struct.pack(f"<{len(tokens)}H", *tokens)
    path.write_bytes(struct.pack("<256i", *header) + payload)


class FakeEnc:
    """Deterministic char-pair 'tokenizer' for encode-path tests."""

    def encode(self, text, allowed_special=None):
        return [ord(c) % 1000 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def make_model() -> GPT:
    torch.manual_seed(0)
    # EOT 50256 must be in-vocab; head_dim*heads == dim (2*32 == 64)
    return GPT(vocab_size=50304, num_layers=3, model_dim=64,
               head_dim=32, num_heads=2).eval()


def manual_outputs(model: GPT, inputs: torch.Tensor) -> list[torch.Tensor]:
    """The reference forward: norm1(embed) then the block loop."""
    with torch.no_grad():
        x = model.norm1(model.embed(inputs))
        for block in model.blocks:
            x = block(x)
            yield x


class CaptureBlockOutputsTests(unittest.TestCase):
    def test_matches_manual_forward_and_leaves_no_hooks(self):
        model = make_model()
        inputs = torch.randint(0, 50304, (2, 10), dtype=torch.int64)
        outputs = capture_block_outputs(model, inputs)
        self.assertEqual(len(outputs), 3)
        for out, ref in zip(outputs, manual_outputs(model, inputs)):
            self.assertTrue(torch.equal(out, ref))
        # hooks are removed: a second capture appends exactly 3 fresh outputs
        outputs2 = capture_block_outputs(model, inputs[:1])
        self.assertEqual(len(outputs2), 3)
        for block in model.blocks:
            self.assertFalse(block._forward_hooks)


class ContextLayerMeansTests(unittest.TestCase):
    def test_tokens_and_means_match_manual_reference(self):
        model = make_model()
        enc = FakeEnc()
        text = "an eerie little passage"
        seq_len = 32
        n, means = context_layer_means(model, enc, Passage("p", text),
                                       seq_len, "cpu")
        self.assertEqual(n, len(enc.encode(text)))
        self.assertEqual(means.shape, (3, 64))
        self.assertEqual(means.dtype, torch.float32)
        ids = enc.encode(text)
        stream = torch.tensor([EOT_TOKEN] + ids, dtype=torch.int64).unsqueeze(0)
        for layer, ref in enumerate(manual_outputs(model, stream)):
            expected = ref[0, 1 : n + 1].to(torch.float32).mean(dim=0)
            self.assertTrue(torch.allclose(means[layer], expected, atol=1e-6))

    def test_overlong_passage_raises(self):
        model = make_model()
        enc = FakeEnc()
        text = "x" * 40  # 40 tokens + 1 EOT > seq_len 32
        with self.assertRaisesRegex(ValueError, "seq_len"):
            context_layer_means(model, enc, Passage("p", text), 32, "cpu")


class ActsContextsTests(unittest.TestCase):
    def _write_set(self, tmp: Path, extra_single: bool = False):
        ex = tmp / "ex.txt"
        single = "## solo\nsome single text\n---\n" if extra_single else ""
        ex.write_text(
            "## pair.contrast\nwrong member text here\n---\n"
            "## pair\ntarget member text here\n---\n" + single,
            encoding="utf-8",
        )
        cr = tmp / "cr.txt"
        cr.write_text("## craft\ncraft essay text here\n---\n", encoding="utf-8")
        return ex, cr

    def test_order_roles_keys_and_shapes(self):
        model = make_model()
        enc = FakeEnc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ex, cr = self._write_set(tmp)
            contexts = acts_contexts(model, enc, ex, cr, seq_len=64,
                                     device="cpu")
        self.assertEqual([c["id"] for c in contexts], ["pair", "pair.contrast",
                                                      "craft"])
        self.assertEqual([c["role"] for c in contexts],
                         ["target", "wrong", "craft"])
        for c in contexts:
            self.assertEqual(set(c.keys()), {"id", "role", "tokens",
                                             "layer_means"})
            self.assertEqual(set(c["layer_means"].keys()), {"0", "1", "2"})
            for layer in ("0", "1", "2"):
                self.assertEqual(len(c["layer_means"][layer]), 64)
                for v in c["layer_means"][layer]:
                    self.assertIsInstance(v, float)
            self.assertEqual(c["tokens"],
                             len(enc.encode(
                                 {"pair": "target member text here",
                                  "pair.contrast": "wrong member text here",
                                  "craft": "craft essay text here"}[c["id"]])))

    def test_single_in_exemplar_file_raises(self):
        model = make_model()
        enc = FakeEnc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ex, cr = self._write_set(tmp, extra_single=True)
            with self.assertRaisesRegex(ValueError, "unpaired"):
                acts_contexts(model, enc, ex, cr, seq_len=64, device="cpu")


class OfftargetWindowsTests(unittest.TestCase):
    def _write_shards(self, tmp: Path):
        shard_dir = tmp / "data" / "shards"
        shard_dir.mkdir(parents=True)
        first = list(range(500, 500 + 32))   # 2 windows of 16
        second = list(range(900, 900 + 64))
        write_tiny_shard(shard_dir / "gutenberg_train_000.bin", first)
        write_tiny_shard(shard_dir / "gutenberg_train_001.bin", second)
        return torch.tensor(first, dtype=torch.int64), "gutenberg_train_000.bin"

    def test_windows_are_first_shard_first_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            expected, first_name = self._write_shards(tmp)
            windows, source = offtarget_windows(
                "data/shards/gutenberg_train_*.bin",
                n_windows=2, window=16, root=tmp)
            self.assertEqual(windows.shape, (2, 16))
            self.assertEqual(windows.dtype, torch.int64)
            self.assertTrue(torch.equal(windows, expected.view(2, 16)))
            self.assertEqual(source, f"{first_name}:tokens[0:32]:2x16")

    def test_too_short_shard_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            self._write_shards(tmp)
            with self.assertRaisesRegex(ValueError, "32 tokens"):
                offtarget_windows("data/shards/gutenberg_train_*.bin",
                                  n_windows=4, window=16, root=tmp)

    def test_missing_glob_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "no shards match"):
                offtarget_windows("data/shards/gutenberg_train_*.bin",
                                  root=Path(tmp))


class LayerCovarianceTests(unittest.TestCase):
    def test_matches_numpy_cov_and_mean(self):
        model = make_model()
        torch.manual_seed(1)
        windows = torch.randint(0, 50304, (3, 16), dtype=torch.int64)
        result = layer_covariance(model, windows, "cpu", batch=2,
                                  exclude_position=0)
        self.assertEqual(result["count"], 3 * 15)
        self.assertEqual(result["covariance"].shape, (3, 64, 64))
        self.assertEqual(result["covariance"].dtype, np.float64)
        self.assertEqual(result["mean"].dtype, np.float64)
        for layer, ref in enumerate(manual_outputs(model, windows)):
            X = ref[:, 1:, :].reshape(-1, 64).to(torch.float64).numpy()
            expected_cov = np.cov(X, rowvar=False, ddof=1)
            expected_mean = X.mean(axis=0)
            np.testing.assert_allclose(result["covariance"][layer],
                                       expected_cov, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(result["mean"][layer], expected_mean,
                                       rtol=1e-10, atol=1e-12)


class SaveCovarianceTests(unittest.TestCase):
    def test_round_trip_exact_keys_dtypes_and_scalars(self):
        model = make_model()
        torch.manual_seed(2)
        windows = torch.randint(0, 50304, (3, 16), dtype=torch.int64)
        cov = layer_covariance(model, windows, "cpu", batch=8,
                               exclude_position=0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "register_covariance_01000.npz"
            save_covariance(path, cov, step=1000, window_tokens=16,
                            excluded_position=0,
                            window_source="gutenberg_train_000.bin:tokens[0:48]:3x16")
            data = np.load(path, allow_pickle=False)
            try:
                self.assertEqual(
                    set(data.keys()),
                    {"covariance", "mean", "count", "step", "window_tokens",
                     "excluded_position", "window_source"},
                )
                self.assertEqual(data["covariance"].shape, (3, 64, 64))
                self.assertEqual(data["covariance"].dtype, np.float64)
                self.assertEqual(data["mean"].shape, (3, 64))
                self.assertEqual(data["mean"].dtype, np.float64)
                for key, expected in (("count", 45), ("step", 1000),
                                      ("window_tokens", 16),
                                      ("excluded_position", 0)):
                    self.assertEqual(data[key].shape, ())
                    self.assertEqual(data[key].dtype, np.int64)
                    self.assertEqual(int(data[key]), expected)
                self.assertEqual(data["window_source"].shape, ())
                self.assertEqual(data["window_source"].dtype.kind, "U")
                self.assertEqual(str(data["window_source"]),
                                 "gutenberg_train_000.bin:tokens[0:48]:3x16")
                np.testing.assert_array_equal(data["covariance"],
                                              cov["covariance"])
                np.testing.assert_array_equal(data["mean"], cov["mean"])
            finally:
                data.close()


if __name__ == "__main__":
    unittest.main()
