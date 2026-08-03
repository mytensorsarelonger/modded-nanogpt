import hashlib
import json
import struct
import tempfile
import unittest
import zipfile
from pathlib import Path

from data.prepare_colab_smoke import (
    DEFAULT_SOURCE_FILES,
    SHARD_MAGIC,
    SHARD_VERSION,
    build_bundle,
    inspect_shard,
    rewrite_manifest,
)


EOT = 50256


def write_tiny_shard(path: Path, tokens: list[int]) -> dict:
    header = [0] * 256
    header[0] = SHARD_MAGIC
    header[1] = SHARD_VERSION
    header[2] = len(tokens)
    payload = struct.pack(f"<{len(tokens)}H", *tokens)
    blob = struct.pack("<256i", *header) + payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    return {
        "shard_path": str(path),
        "num_tokens": len(tokens),
        "file_size": len(blob),
        "sha256": hashlib.sha256(blob).hexdigest(),
        "magic": SHARD_MAGIC,
        "version": SHARD_VERSION,
    }


class PrepareColabSmokeTests(unittest.TestCase):
    def test_rewrite_manifest_preserves_recipe_but_not_full_corpus_totals(self):
        source = {
            "train_shards": [{"num_tokens": 100}],
            "val_shards": [{"num_tokens": 50}],
            "total_train_tokens": 999,
            "total_val_tokens": 555,
            "num_train_books": 90,
            "num_val_books": 10,
            "tokenizer": "gpt2",
            "vocab_size": 50304,
            "quality_thresholds": {"min_chars": 5000},
            "dedup": {
                "method": "minhash_lsh_plus_blockwise_prefilter",
                "jaccard_threshold": 0.8,
                "clusters": 20,
                "documents_dropped": 20,
            },
            "slices": {"backbone": 900, "register": 99},
        }
        train = [{"shard_path": "data/shards/train.bin", "num_tokens": 100}]
        val = [{"shard_path": "data/shards/val.bin", "num_tokens": 50}]
        reduced = rewrite_manifest(
            source,
            train,
            val,
            train_document_starts=3,
            val_document_starts=2,
            source_manifest_sha256="abc123",
        )

        self.assertEqual(reduced["total_train_tokens"], 100)
        self.assertEqual(reduced["total_val_tokens"], 50)
        self.assertEqual(reduced["num_train_books"], 3)
        self.assertEqual(reduced["num_val_books"], 2)
        self.assertEqual(reduced["tokenizer"], "gpt2")
        self.assertEqual(reduced["quality_thresholds"], {"min_chars": 5000})
        self.assertNotIn("slices", reduced)
        self.assertNotIn("clusters", reduced["dedup"])
        self.assertEqual(reduced["dedup"]["jaccard_threshold"], 0.8)
        self.assertEqual(reduced["source_corpus"]["total_train_tokens"], 999)
        self.assertEqual(reduced["source_corpus"]["slices"]["register"], 99)
        self.assertEqual(
            reduced["source_corpus"]["dedup_outcome"],
            {"clusters": 20, "documents_dropped": 20},
        )

    def test_build_bundle_selects_first_shards_counts_eot_and_zips_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            shards = repo / "data" / "shards"
            train0 = write_tiny_shard(shards / "train_000.bin", [EOT, 1, 2, EOT, 3])
            train1 = write_tiny_shard(shards / "train_001.bin", [EOT, 4, 5])
            val0 = write_tiny_shard(shards / "val_000.bin", [EOT, 6, EOT, 7])
            source_manifest = {
                "train_shards": [train0, train1],
                "val_shards": [val0],
                "total_train_tokens": 8,
                "total_val_tokens": 4,
                "num_train_books": 3,
                "num_val_books": 2,
                "tokenizer": "gpt2",
                "vocab_size": 50304,
                "eot_token": EOT,
                "doc_separator": "eot_prefix_per_document",
                "quality_filter": "tier1_heuristic",
                "quality_thresholds": {"min_chars": 5000},
                "dedup": {"method": "minhash", "clusters": 1,
                          "documents_dropped": 1},
                "slices": {"backbone": 7, "register": 1},
            }
            manifest_path = shards / "manifest.json"
            manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
            (repo / "train_baseline.py").write_text("# trainer\n", encoding="utf-8")
            (repo / "config.py").write_text("# config\n", encoding="utf-8")

            output = root / "bundle"
            archive = root / "bundle.zip"
            result = build_bundle(
                manifest_path,
                output,
                repo_root=repo,
                source_files=("train_baseline.py", "config.py"),
                zip_path=archive,
            )

            reduced = json.loads(
                (output / "data" / "shards" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(result["train_tokens"], 5)
            self.assertEqual(result["val_tokens"], 4)
            self.assertEqual(result["train_books"], 2)
            self.assertEqual(result["val_books"], 2)
            self.assertEqual(result["copy"], 2)
            self.assertEqual(result["hardlink"], 0)
            self.assertEqual(len(reduced["train_shards"]), 1)
            self.assertEqual(
                reduced["train_shards"][0]["shard_path"],
                "data/shards/train_000.bin",
            )
            self.assertEqual(
                reduced["val_shards"][0]["shard_path"],
                "data/shards/val_000.bin",
            )
            self.assertFalse((output / "data" / "shards" / "train_001.bin").exists())
            self.assertTrue((output / "train_baseline.py").is_file())
            self.assertTrue((output / "config.py").is_file())
            self.assertTrue((output / "COLAB_SMOKE.md").is_file())
            readme = (output / "COLAB_SMOKE.md").read_text(encoding="utf-8")
            self.assertIn("source `data/manifest.jsonl` is included", readme)
            self.assertIn("actual reduced shard subset", readme)
            self.assertEqual(
                inspect_shard(output / "data" / "shards" / "train_000.bin", EOT)[
                    "document_starts"
                ],
                2,
            )

            self.assertTrue(archive.is_file())
            with zipfile.ZipFile(archive) as zipped:
                names = set(zipped.namelist())
            self.assertIn("bundle/train_baseline.py", names)
            self.assertIn("bundle/data/shards/manifest.json", names)
            self.assertIn("bundle/data/shards/train_000.bin", names)
            self.assertNotIn("bundle/data/shards/train_001.bin", names)

            with self.assertRaises(FileExistsError):
                build_bundle(
                    manifest_path,
                    output,
                    repo_root=repo,
                    materialization="copy",
                    source_files=(),
                )

            stale = output / "stale.txt"
            stale.write_text("old", encoding="utf-8")
            build_bundle(
                manifest_path,
                output,
                repo_root=repo,
                materialization="copy",
                source_files=(),
                force=True,
            )
            self.assertFalse(stale.exists())

    def test_output_cannot_replace_an_ancestor_of_the_repository(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            manifest = repo / "data" / "shards" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protected output"):
                build_bundle(manifest, root, repo_root=repo)

    def test_output_cannot_live_in_git_or_active_shard_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            manifest = repo / "data" / "shards" / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("{}", encoding="utf-8")
            for output in (repo / ".git" / "bundle",
                           manifest.parent / "bundle"):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "protected directory"):
                        build_bundle(manifest, output, repo_root=repo)
            with self.assertRaisesRegex(ValueError, "protected directory"):
                build_bundle(
                    manifest,
                    repo / "bundle",
                    repo_root=repo,
                    zip_path=repo / ".git" / "bundle.zip",
                )

    def test_default_sources_include_full_per_document_manifest(self):
        self.assertIn("data/manifest.jsonl", DEFAULT_SOURCE_FILES)


if __name__ == "__main__":
    unittest.main()
