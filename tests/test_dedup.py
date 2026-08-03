import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data import dedup


class DedupTests(unittest.TestCase):
    def test_sparse_content_sample_never_switches_signature_universe(self):
        text = " ".join(f"word{i}" for i in range(20))
        full = dedup._shingle_hashes(text)
        sampled = full[(full % np.uint64(dedup.KEEP_MOD)) == 0]
        self.assertLess(sampled.size, dedup.MIN_SAMPLED_SHINGLES)
        np.testing.assert_array_equal(dedup._signature_shingles(full), sampled)

    def test_sparse_candidate_path_bridges_seven_to_eight_sample_boundary(self):
        signatures = np.vstack([
            np.arange(dedup.N_PERM, dtype=np.uint64),
            np.arange(dedup.N_PERM, dtype=np.uint64) + np.uint64(1_000),
            np.arange(dedup.N_PERM, dtype=np.uint64) + np.uint64(2_000),
        ])
        candidates = dedup._candidate_pairs(
            signatures,
            full_sizes=[100, 105, 500],
            sample_sizes=[7, 8, 20],
        )
        self.assertIn((0, 1), candidates)
        self.assertNotIn((0, 2), candidates)

    def test_blockwise_prefilter_catches_matches_scattered_across_bands(self):
        first = np.arange(dedup.N_PERM, dtype=np.uint64)
        second = first.copy()
        # 101/120 matches, but deliberately no complete 10-row band.
        changed = [band * dedup.ROWS + dedup.ROWS - 1
                   for band in range(dedup.BANDS)]
        changed += [band * dedup.ROWS for band in range(7)]
        second[changed] += np.uint64(10_000)
        signatures = np.vstack([first, second])

        self.assertEqual(int((first == second).sum()), 101)
        self.assertNotIn((0, 1), dedup._lsh_candidates(signatures))
        self.assertIn((0, 1), dedup._candidate_pairs(signatures))

    def test_exact_full_set_verification_rejects_a_forced_false_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.txt"
            b = root / "b.txt"
            a.write_text(" ".join(f"word{i}" for i in range(100)), encoding="utf-8")
            b.write_text(
                " ".join([*(f"word{i}" for i in range(70)),
                          *(f"other{i}" for i in range(30))]),
                encoding="utf-8",
            )
            books = [
                {"gutenberg_id": 1, "path": a, "char_count": a.stat().st_size,
                 "quality_eligible": True},
                {"gutenberg_id": 2, "path": b, "char_count": b.stat().st_size,
                 "quality_eligible": True},
            ]
            with patch.object(dedup, "_candidate_pairs", return_value={(0, 1)}):
                drop, clusters = dedup.find_duplicates(books, verbose=False)
            self.assertEqual(drop, set())
            self.assertEqual(clusters, {})

    def test_transitive_edge_does_not_create_transitive_drop(self):
        books = [
            {"gutenberg_id": 1, "char_count": 300, "quality_eligible": True},
            {"gutenberg_id": 2, "char_count": 200, "quality_eligible": True},
            {"gutenberg_id": 3, "char_count": 100, "quality_eligible": True},
        ]
        drop, clusters = dedup._select_clusters(books, {(0, 1), (1, 2)})
        self.assertEqual(drop, {2})
        self.assertEqual(clusters, {1: [2]})

    def test_quality_eligible_copy_beats_longer_dirty_copy(self):
        books = [
            {"gutenberg_id": 1, "char_count": 1_000, "quality_eligible": False},
            {"gutenberg_id": 2, "char_count": 900, "quality_eligible": True},
        ]
        drop, clusters = dedup._select_clusters(books, {(0, 1)})
        self.assertEqual(drop, {1})
        self.assertEqual(clusters, {2: [1]})

    def test_all_dirty_component_is_left_to_quality_filter(self):
        books = [
            {"gutenberg_id": 1, "char_count": 1_000, "quality_eligible": False},
            {"gutenberg_id": 2, "char_count": 900, "quality_eligible": False},
        ]
        drop, clusters = dedup._select_clusters(books, {(0, 1)})
        self.assertEqual(drop, set())
        self.assertEqual(clusters, {})

    def test_end_to_end_prefers_clean_copy_even_when_dirty_copy_is_longer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "clean.txt"
            dirty = root / "dirty.txt"
            content = " ".join(f"word{i}" for i in range(100))
            clean.write_text(content, encoding="utf-8")
            dirty.write_text(content + ("\n" * 100), encoding="utf-8")
            books = [
                {"gutenberg_id": 1, "path": dirty,
                 "char_count": dirty.stat().st_size, "quality_eligible": False},
                {"gutenberg_id": 2, "path": clean,
                 "char_count": clean.stat().st_size, "quality_eligible": True},
            ]
            drop, clusters = dedup.find_duplicates(books, verbose=False)
            self.assertEqual(drop, {1})
            self.assertEqual(clusters, {2: [1]})

    def test_manifest_resolution_is_last_wins_and_excludes_copyright(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "old.txt"
            new = root / "new.txt"
            copyrighted = root / "copyrighted.txt"
            failed_old = root / "failed-old.txt"
            copyright_old = root / "copyright-old.txt"
            for path in (old, new, copyrighted, failed_old, copyright_old):
                path.write_text(path.stem, encoding="utf-8")
            entries = [
                {"gutenberg_id": 1, "local_path": str(old), "title": "old"},
                {"gutenberg_id": 2, "local_path": str(copyrighted),
                 "title": "copyrighted", "copyright": True},
                {"gutenberg_id": 3, "local_path": str(failed_old),
                 "title": "older usable"},
                {"gutenberg_id": 4, "local_path": str(copyright_old),
                 "title": "older public domain"},
                {"gutenberg_id": 1, "local_path": str(new), "title": "new"},
                {"gutenberg_id": 3, "status": "failed", "title": "final failed"},
                {"gutenberg_id": 4, "local_path": str(copyrighted),
                 "title": "final copyright", "copyright": True},
            ]
            books = dedup.resolve_manifest_books(entries)
            self.assertEqual(len(books), 1)
            self.assertEqual(books[0]["path"], new)
            self.assertEqual(books[0]["title"], "new")


if __name__ == "__main__":
    unittest.main()
