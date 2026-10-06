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
from evals.exemplar import EOT_TOKEN, evaluate  # noqa: E402
from evals.register_extract import (  # noqa: E402
    acts_contexts,
    capture_block_outputs,
    context_layer_means,
    direction_delta,
    greedy_continuation,
    layer_covariance,
    offtarget_windows,
    otm_values,
    rand_direction,
    save_covariance,
    steer_hook,
    steer_row,
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


class SteerHookTests(unittest.TestCase):
    def test_zero_delta_leaves_logits_unchanged_and_no_hooks_left(self):
        model = make_model()
        torch.manual_seed(3)
        inputs = torch.randint(0, 50304, (2, 10), dtype=torch.int64)
        with torch.no_grad():
            base = model.forward_logits(inputs).clone()
        zero = torch.zeros(64)
        with steer_hook(model, 1, zero):
            with torch.no_grad():
                steered = model.forward_logits(inputs)
        self.assertTrue(torch.equal(base, steered))
        for block in model.blocks:
            self.assertFalse(block._forward_hooks)

    def test_nonzero_delta_adds_exactly_bf16_delta_to_block_output(self):
        model = make_model()
        torch.manual_seed(4)
        inputs = torch.randint(0, 50304, (2, 10), dtype=torch.int64)
        delta = torch.randn(64)
        # reference: block 1's unhooked output, then the same forward
        with torch.no_grad():
            ref_outputs = list(manual_outputs(model, inputs))
        with steer_hook(model, 1, delta):
            with torch.no_grad():
                steered_outputs = list(manual_outputs(model, inputs))
        # block 0 unaffected, block 1 shifted by delta.to(bf16), block 2
        # downstream of the shift (not compared bit-exactly)
        self.assertTrue(torch.equal(ref_outputs[0], steered_outputs[0]))
        expected = ref_outputs[1] + delta.to(ref_outputs[1].dtype)
        self.assertTrue(torch.equal(steered_outputs[1], expected))
        for block in model.blocks:
            self.assertFalse(block._forward_hooks)

    def test_hook_removed_on_exception(self):
        model = make_model()
        delta = torch.ones(64)
        with self.assertRaises(RuntimeError):
            with steer_hook(model, 2, delta):
                raise RuntimeError("boom")
        for block in model.blocks:
            self.assertFalse(block._forward_hooks)


class EvaluateUnderHookTests(unittest.TestCase):
    def _files(self, tmp: Path):
        ex = tmp / "ex.txt"
        ex.write_text(
            "## pair.contrast\nwrong member text here\n---\n"
            "## pair\ntarget member text here\n---\n",
            encoding="utf-8")
        cr = tmp / "cr.txt"
        cr.write_text("## craft\ncraft essay text here\n---\n",
                      encoding="utf-8")
        return ex, cr

    def test_evaluate_zero_hook_equals_unhooked(self):
        model = make_model()
        enc = FakeEnc()
        with tempfile.TemporaryDirectory() as tmp:
            ex, cr = self._files(Path(tmp))
            base = evaluate(model, enc, exemplar_file=ex, craft_file=cr,
                            seq_len=64, device="cpu")
            with steer_hook(model, 1, torch.zeros(64)):
                hooked = evaluate(model, enc, exemplar_file=ex,
                                  craft_file=cr, seq_len=64, device="cpu")
        self.assertEqual(base["mean_loss_gap"], hooked["mean_loss_gap"])
        self.assertEqual(base["contrast_gaps"], hooked["contrast_gaps"])


class OtmValuesTests(unittest.TestCase):
    def _windows(self, n=3, window=16, seed=5):
        torch.manual_seed(seed)
        return torch.randint(0, 50304, (n, window), dtype=torch.int64)

    def test_alpha_zero_is_exactly_zero(self):
        model = make_model()
        v = np.random.default_rng(0).standard_normal(64)
        values = otm_values(model, self._windows(), 1, v, [0.0],
                            "cpu", batch=2)
        self.assertEqual(values, [0.0])

    def test_values_nonnegative_and_hand_checked(self):
        model = make_model()
        windows = self._windows()
        v = np.random.default_rng(1).standard_normal(64)
        values = otm_values(model, windows, 1, v, [0.5, 1.0, 2.0],
                            "cpu", batch=2)
        self.assertEqual(len(values), 3)
        for x in values:
            self.assertGreaterEqual(x, 0.0)
        # hand-written KL for the FULL window set (not one window): the
        # mean over positions 1..15 of every window, in plain torch
        delta = torch.tensor(0.5 * v, dtype=torch.float32).to(torch.bfloat16)
        with torch.no_grad():
            logp0 = torch.log_softmax(model.forward_logits(windows), -1)
            with steer_hook(model, 1, delta):
                logpa = torch.log_softmax(model.forward_logits(windows), -1)
            kl = (logp0.exp() * (logp0 - logpa)).sum(-1)[:, 1:]
        self.assertAlmostEqual(values[0], float(kl.mean()), places=6)

    def test_position_zero_excluded(self):
        # a delta whose ONLY effect is at position 0 is not counted: the
        # KL at positions 1.. is what the mean reads. Assert the divisor:
        # kept positions per window = window − 1, not window.
        model = make_model()
        windows = self._windows()
        v = np.random.default_rng(2).standard_normal(64)
        values = otm_values(model, windows, 2, v, [1.0], "cpu", batch=4)
        self.assertGreaterEqual(values[0], 0.0)
        # divisor: every kept position contributes; count == 3*15
        delta = torch.tensor(v, dtype=torch.float32).to(torch.bfloat16)
        with torch.no_grad():
            logp0 = torch.log_softmax(model.forward_logits(windows), -1)
            with steer_hook(model, 2, delta):
                logpa = torch.log_softmax(model.forward_logits(windows), -1)
            kl = (logp0.exp() * (logp0 - logpa)).sum(-1)
        self.assertAlmostEqual(values[0],
                                float(kl[:, 1:].sum() / (3 * 15)), places=6)
        # and it is NOT the mean including position 0
        self.assertNotAlmostEqual(values[0], float(kl.mean()), places=6)


class RandDirectionTests(unittest.TestCase):
    def test_unit_norm_and_deterministic(self):
        a = rand_direction(20261009, dim=64)
        b = rand_direction(20261009, dim=64)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(a)), 1.0, places=12)
        self.assertEqual(a.dtype, np.float64)

    def test_seeds_distinct(self):
        a = rand_direction(20261009, dim=64)
        b = rand_direction(20261009 + 1000, dim=64)
        self.assertFalse(np.allclose(a, b))
        self.assertGreater(float(np.dot(a, b)), -0.99)
        self.assertLess(float(np.dot(a, b)), 0.99)


class GreedyContinuationTests(unittest.TestCase):
    def test_equals_handwritten_argmax_loop(self):
        model = make_model()
        torch.manual_seed(6)
        prefix = [EOT_TOKEN] + [ord(c) % 1000 for c in "an eerie prefix"]
        cont = greedy_continuation(model, prefix, 8, "cpu")
        # hand-written loop: full re-forward each step, no KV cache
        seq = torch.tensor([prefix], dtype=torch.int64)
        expected = []
        with torch.no_grad():
            for _ in range(8):
                nxt = int(torch.argmax(
                    model.forward_logits(seq)[0, -1]).item())
                expected.append(nxt)
                seq = torch.cat([seq, torch.tensor([[nxt]],
                                  dtype=torch.int64)], dim=1)
        self.assertEqual(cont, expected)
        for block in model.blocks:
            self.assertFalse(block._forward_hooks)

    def test_steered_continuation_differs_for_nonzero_delta(self):
        model = make_model()
        torch.manual_seed(7)
        prefix = [EOT_TOKEN] + [ord(c) % 1000 for c in "another prefix"]
        delta = 5.0 * torch.randn(64)
        with steer_hook(model, 2, delta):
            steered = greedy_continuation(model, prefix, 6, "cpu")
        plain = greedy_continuation(model, prefix, 6, "cpu")
        # a big delta at layer 2 should change SOME generated token
        self.assertNotEqual(steered, plain)


class SteerRowTests(unittest.TestCase):
    def _row(self, direction, otm=None):
        model = make_model()
        enc = FakeEnc()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ex = tmp / "ex.txt"
            ex.write_text(
                "## pair.contrast\nwrong member text here\n---\n"
                "## pair\ntarget member text here\n---\n",
                encoding="utf-8")
            cr = tmp / "cr.txt"
            cr.write_text("## craft\ncraft essay text here\n---\n",
                          encoding="utf-8")
            v = np.random.default_rng(8).standard_normal(64)
            windows = torch.randint(0, 50304, (2, 16), dtype=torch.int64)
            return steer_row(
                model, enc, regime="R1", run_id="run-b", step=3250,
                theta_A={"run_id": "run-a", "step": 1000},
                layer=1, direction=direction, alpha=1.0,
                v=v, v_norm=float(np.linalg.norm(v)),
                exemplar_file=ex, craft_file=cr, seq_len=64,
                device="cpu", windows=windows,
                window_source="gutenberg_train_000.bin:tokens[0:32]:2x16")

    def test_exact_key_set_and_otm_null_off_v(self):
        row = self._row("neg_v")
        self.assertEqual(
            set(row.keys()),
            {"regime", "run_id", "step", "theta_A", "layer", "direction",
             "alpha", "v_norm", "mean_loss_gap_steered", "contrast_gaps",
             "otm"})
        self.assertIsNone(row["otm"])
        self.assertEqual(row["theta_A"], {"run_id": "run-a", "step": 1000})

    def test_v_row_otm_metadata(self):
        row = self._row("v")
        otm = row["otm"]
        self.assertEqual(otm["alphas"], [0.5, 1.0, 2.0])
        self.assertEqual(otm["kl_direction"], "unsteered||steered")
        self.assertEqual(otm["off_target_windows"], 2)
        self.assertEqual(otm["positions_per_window"], 15)
        self.assertEqual(otm["window_source"],
                         "gutenberg_train_000.bin:tokens[0:32]:2x16")
        self.assertEqual(len(otm["values"]), 3)

    def test_rand_direction_uses_seed(self):
        row = self._row("rand:20262010")
        self.assertEqual(row["direction"], "rand:20262010")
        self.assertIsNone(row["otm"])
        self.assertEqual(row["alpha"], 1.0)


class DirectionDeltaTests(unittest.TestCase):
    def test_all_directions(self):
        v = np.random.default_rng(9).standard_normal(64)
        norm = float(np.linalg.norm(v))
        np.testing.assert_array_equal(direction_delta("v", v, norm), v)
        np.testing.assert_array_equal(direction_delta("neg_v", v, norm), -v)
        np.testing.assert_array_equal(direction_delta("zero", v, norm),
                                      np.zeros(64))
        r = direction_delta("rand:20261009", v, norm)
        self.assertAlmostEqual(float(np.linalg.norm(r)), norm, places=9)
        with self.assertRaisesRegex(ValueError, "unknown direction"):
            direction_delta("sideways", v, norm)


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
