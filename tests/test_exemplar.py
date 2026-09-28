import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from evals.exemplar import (  # noqa: E402
    EOT_TOKEN,
    Passage,
    _encode_windows,
    _find_matches,
    append_jsonl,
    check_contamination,
    eval_set_digest,
    evaluate,
    load_passages,
    score_passage,
    split_pairs,
)
from training_state import copy_jsonl_through_step  # noqa: E402


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


class EvalSetParsingTests(unittest.TestCase):
    def test_load_passages_reads_sections_and_ignores_preamble(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ex.txt"
            p.write_text(
                "# header comment\npreamble prose that is not a passage\n\n"
                "## alpha\nfirst passage body\n\n## beta\nsecond passage\n",
                encoding="utf-8",
            )
            passages = load_passages(p)
            self.assertEqual([x.id for x in passages], ["alpha", "beta"])
            self.assertEqual(passages[0].text, "first passage body")
            self.assertEqual(passages[1].text, "second passage")

    def test_load_passages_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ex.txt"
            p.write_text("no sections here", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "No '## <id>'"):
                load_passages(p)

    def test_split_pairs_groups_by_contrast_suffix(self):
        passages = [
            Passage("a", "x"), Passage("a.contrast", "y"),
            Passage("b", "z"),
        ]
        singles, pairs = split_pairs(passages)
        self.assertEqual([s.id for s in singles], ["b"])
        self.assertEqual([(t.id, w.id) for t, w in pairs], [("a", "a.contrast")])

    def test_orphan_contrast_member_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no target passage"):
            split_pairs([Passage("a.contrast", "y")])

    def test_digest_changes_when_texts_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.txt"
            b = Path(tmp) / "b.txt"
            a.write_text("one", encoding="utf-8")
            b.write_text("two", encoding="utf-8")
            d1 = eval_set_digest([a, b])
            self.assertEqual(d1, eval_set_digest([a, b]))
            a.write_text("ONE", encoding="utf-8")
            self.assertNotEqual(d1, eval_set_digest([a, b]))


class WindowTests(unittest.TestCase):
    def test_windows_are_eot_anchored_and_cover_the_text(self):
        enc = FakeEnc()
        text = "abcdefgh"  # 8 tokens
        windows = _encode_windows(text, enc, seq_len=4, device="cpu")
        # first window: [EOT, a, b, c] as inputs -> targets [a, b, c, d]
        # (seq_len+1 slice: inputs are the first seq_len, targets the last seq_len)
        first_inputs, first_targets = windows[0]
        self.assertEqual(first_inputs.tolist(),
                         [EOT_TOKEN, ord("a") % 1000, ord("b") % 1000, ord("c") % 1000])
        self.assertEqual(first_targets.tolist(),
                         [ord("a") % 1000, ord("b") % 1000, ord("c") % 1000, ord("d") % 1000])
        # every text token appears as a target at least once
        seen = set()
        for _, targets in windows:
            seen.update(targets.tolist())
        self.assertEqual(seen, {ord(c) % 1000 for c in text})

    def test_short_text_yields_one_window(self):
        windows = _encode_windows("ab", FakeEnc(), seq_len=1024, device="cpu")
        self.assertEqual(len(windows), 1)
        inputs, targets = windows[0]
        self.assertEqual(inputs[0].item(), EOT_TOKEN)
        self.assertEqual(targets.numel(), 2)


class _TinyScorer(torch.nn.Module):
    """forward_logits stub: uniform logits -> PPL == vocab size."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab
        self.training = True

    def forward_logits(self, inputs):
        return torch.zeros(inputs.size(0), inputs.size(1), self.vocab)


class ScoringTests(unittest.TestCase):
    def test_uniform_model_ppl_equals_vocab_size(self):
        enc = FakeEnc()
        model = _TinyScorer(vocab=1000)
        scored = score_passage(model, enc, Passage("t", "abcdef"), seq_len=4, device="cpu")
        # uniform distribution over 1000 symbols: NLL = ln(1000)
        import math
        self.assertAlmostEqual(scored.ppl, 1000.0, places=3)
        self.assertAlmostEqual(scored.nll_sum / scored.tokens, math.log(1000), places=4)
        # eval mode was restored
        self.assertTrue(model.training)

    def test_zero_token_passage_is_an_error(self):
        model = _TinyScorer(vocab=1000)
        with self.assertRaisesRegex(ValueError, "zero tokens"):
            score_passage(model, FakeEnc(), Passage("t", ""), seq_len=4, device="cpu")


class ContaminationGuardTests(unittest.TestCase):
    def test_exact_match_is_found_at_any_alignment(self):
        corpus = torch.tensor([5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
        query = torch.tensor([9, 10, 11, 12, 13, 14])  # matches corpus[4:10]
        hits = _find_matches(query, corpus, match_len=6)
        self.assertEqual(hits, [(0, 4)])
        # shifted alignment: query starts at odd offset
        query2 = torch.tensor([10, 11, 12, 13, 14, 15])
        hits2 = _find_matches(query2, corpus, match_len=6)
        self.assertEqual(hits2, [(0, 5)])

    def test_no_match_returns_empty(self):
        corpus = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        query = torch.tensor([9, 9, 9, 9, 9, 9])
        self.assertEqual(_find_matches(query, corpus, match_len=6), [])

    def test_multiple_passage_offsets_reported(self):
        corpus = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        query = torch.tensor([1, 2, 3, 4, 5, 1, 2, 3, 4, 5])
        hits = _find_matches(query, corpus, match_len=6)
        # the full query occurs at corpus 0; and the query's own second copy
        # of the pattern (q=4) also matches the corpus head — every offset on
        # BOTH sides is covered by the sweep, not just query offset 0.
        self.assertIn((0, 0), hits)
        self.assertIn((4, 4), hits)
        # corpus-offset 5 can only host a 5-token match with this corpus length
        self.assertNotIn((0, 5), hits)

    def test_guard_reports_and_assert_clean_raises(self):
        enc = FakeEnc()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "data" / "shards"
            shard_dir.mkdir(parents=True)
            # passage text tokenizes to [ord('a')%1000, ...]; embed it verbatim
            body = "abc" + "x" * 20
            body_ids = enc.encode(body)
            write_tiny_shard(shard_dir / "gutenberg_train_000.bin",
                             [EOT_TOKEN] + body_ids + [EOT_TOKEN])
            passage = Passage("p", body)
            hits = check_contamination([passage], enc,
                                      "data/shards/gutenberg_train_*.bin",
                                      root=root)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["passage"], "p")
            self.assertTrue(hits[0]["snippet"])

            clean_passage = Passage("q", "zzz" + "y" * 30)
            self.assertEqual(
                check_contamination([clean_passage], enc,
                                    "data/shards/gutenberg_train_*.bin",
                                    root=root),
                [],
            )

            from evals.exemplar import assert_clean
            with self.assertRaisesRegex(RuntimeError, "contamination guard"):
                assert_clean([passage], enc,
                             "data/shards/gutenberg_train_*.bin", root=root)

    def test_missing_shard_glob_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shard_dir = root / "data" / "shards"
            shard_dir.mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "no shards match"):
                check_contamination([Passage("p", "hello")], FakeEnc(),
                                    "data/shards/gutenberg_train_*.bin",
                                    root=root)


class JsonlTests(unittest.TestCase):
    def test_copy_jsonl_through_step_filters_and_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.jsonl"
            dst = Path(tmp) / "dst.jsonl"
            rows = [
                {"step": 250, "v": 1.0},
                {"step": 500, "v": 2.0},
                {"step": 750, "v": 3.0},
            ]
            src.write_text("\n".join(json.dumps(r) for r in rows) + "\n",
                           encoding="utf-8")
            kept = copy_jsonl_through_step(src, dst, 500)
            self.assertEqual(kept, 2)
            out = [json.loads(l) for l in dst.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["step"] for r in out], [250, 500])

    def test_copy_jsonl_missing_source_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                copy_jsonl_through_step(Path(tmp) / "nope.jsonl",
                                       Path(tmp) / "dst.jsonl", 10),
                0,
            )

    def test_append_jsonl_writes_strict_json_and_nulls_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.jsonl"
            append_jsonl(p, {"step": 1, "x": float("nan"), "nested": {"y": float("inf")}})
            with open(p, encoding="utf-8") as f:
                line = f.readline()
            # strict-JSON parse must succeed and carry the divergence explicitly
            record = json.loads(
                line, parse_constant=lambda s: (_ for _ in ()).throw(
                    ValueError(f"non-standard constant {s}"))
            )
            self.assertIsNone(record["x"])
            self.assertTrue(record["x_nonfinite"])
            self.assertIsNone(record["nested"]["y"])
            self.assertTrue(record["nested"]["y_nonfinite"])
            self.assertEqual(record["step"], 1)

    def test_append_jsonl_finite_values_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "out.jsonl"
            append_jsonl(p, {"step": 250, "ppl": 12.5, "gap": -0.3})
            record = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(record["step"], 250)
            self.assertAlmostEqual(record["ppl"], 12.5)
            self.assertAlmostEqual(record["gap"], -0.3)
            self.assertNotIn("ppl_nonfinite", record)


class _RepeatModel:
    """forward_logits that always predicts the CURRENT input token as the
    next: loss on a passage is then content-dependent (it depends only on
    how often targets[i] == inputs[i]), so a test can prove WHICH passages
    entered an aggregate."""

    def __init__(self, vocab):
        self.vocab = vocab
        self.training = False

    def eval(self):
        return self

    def train(self):
        return self

    def forward_logits(self, inputs):
        # B,T -> B,T,V one-hot of the input token at each position
        b, t = inputs.shape
        logits = torch.full((b, t, self.vocab), -20.0)
        logits.scatter_(2, inputs.unsqueeze(2), 20.0)
        return logits


class EvaluateSemanticsTests(unittest.TestCase):
    """Lock §7.2 quantity 1: exemplar PPL aggregates REGISTER TARGETS only."""

    def _write_set(self, tmp: Path, target_text: str, wrong_text: str,
                   craft_text: str) -> tuple[Path, Path]:
        ex = tmp / "ex.txt"
        ex.write_text(f"## pair.contrast\n{wrong_text}\n---\n## pair\n{target_text}\n---\n",
                      encoding="utf-8")
        cr = tmp / "cr.txt"
        cr.write_text(f"## craft\n{craft_text}\n---\n", encoding="utf-8")
        return ex, cr

    def test_exemplar_ppl_is_target_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            ex, cr = self._write_set(
                tmp,
                target_text="ABAB",      # target: alternation, low repeat-loss
                wrong_text="AAAAAAAAAA",  # mundane: high repeat-loss
                craft_text="CDCD",
            )
            enc = FakeEnc()
            result = evaluate(_RepeatModel(vocab=50357), enc,
                              exemplar_file=ex, craft_file=cr,
                              seq_len=8, device="cpu")
            # If the wrong member had entered the exemplar aggregate, the
            # token-weighted mean would shift toward its (much lower) loss.
            # Compute the target-only expectation and require it exactly.
            tgt = result["contrast_gaps"][0]
            self.assertAlmostEqual(
                result["exemplar_ppl_nats_per_tok"],
                tgt["target_nll_per_tok"], places=5)
            # the wrong member was scored and reported per-pair
            self.assertEqual(len(result["contrast_gaps"]), 1)
            # 10 source tokens; the EOT-anchored half-stride windowing
            # re-scores overlap, so the counted tokens exceed the raw count
            self.assertGreater(result["contrast_gaps"][0]["wrong_tokens"], 10)
            self.assertIsNotNone(result["mean_loss_gap"])



if __name__ == "__main__":
    unittest.main()
