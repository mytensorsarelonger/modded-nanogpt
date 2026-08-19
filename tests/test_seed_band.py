"""Tests for analysis/seed_band.py (PLAN.md §4.1)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.seed_band import collect, load_runs, val_curve  # noqa: E402


def _log(steps_vals, total=1000):
    lines = ["some preamble"]
    for st, v in steps_vals:
        lines.append(f"step:{st}/{total} val_loss:{v} train_loss:9.9 train_time:1.0s")
    return "\n".join(lines) + "\n"


def _fixture(tmp_path, runs):
    logs = tmp_path / "_logs"
    logs.mkdir()
    idx = tmp_path / "index.jsonl"
    with idx.open("w", encoding="utf-8") as f:
        for r in runs:
            f.write(json.dumps(r["entry"]) + "\n")
            if r.get("log") is not None:
                name = Path(r["entry"]["logfile"]).name
                (logs / name).write_text(r["log"], encoding="utf-8")
    return idx, logs


def test_val_curve_parses_and_drops_nan(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text(_log([(0, "10.5"), (125, "4.2"), (250, "nan")]), encoding="utf-8")
    assert val_curve(p) == {0: 10.5, 125: 4.2}


def test_val_curve_missing_file_is_empty(tmp_path):
    assert val_curve(tmp_path / "nope.txt") == {}


def _entry(rid, seed, steps=1000, **kw):
    e = {"run_id": rid, "init_seed": seed, "train_steps": steps,
         "logfile": f"logs/{rid}.txt", "final_val_loss": 3.0}
    e.update(kw)
    return e


def test_collects_seeds_at_one_length(tmp_path):
    runs = [
        {"entry": _entry("a", 1), "log": _log([(0, "10.8"), (500, "4.10")])},
        {"entry": _entry("b", 2), "log": _log([(0, "10.8"), (500, "4.13")])},
    ]
    idx, logs = _fixture(tmp_path, runs)
    g = collect(load_runs(idx), logs, None)
    assert set(g) == {1000}
    assert set(g[1000]) == {1, 2}


def test_groups_are_separated_by_train_steps(tmp_path):
    """The band is length-specific: LR schedule is relative to train_steps."""
    runs = [
        {"entry": _entry("a", 1, steps=1000), "log": _log([(500, "4.1")], 1000)},
        {"entry": _entry("b", 2, steps=1000), "log": _log([(500, "4.2")], 1000)},
        {"entry": _entry("c", 1, steps=3250), "log": _log([(500, "3.9")], 3250)},
    ]
    idx, logs = _fixture(tmp_path, runs)
    g = collect(load_runs(idx), logs, None)
    assert set(g) == {1000, 3250}
    assert set(g[3250]) == {1}


def test_mixed_runs_are_excluded(tmp_path):
    """A mixed run trains on different data; it is not a seed replicate."""
    runs = [
        {"entry": _entry("a", 1), "log": _log([(500, "4.1")])},
        {"entry": _entry("b", 2, mix="mix-a"), "log": _log([(500, "9.9")])},
    ]
    idx, logs = _fixture(tmp_path, runs)
    g = collect(load_runs(idx), logs, None)
    assert set(g[1000]) == {1}


def test_resumed_runs_are_excluded(tmp_path):
    """A resumed run shares its parent's trajectory and would double-count."""
    runs = [
        {"entry": _entry("a", 1), "log": _log([(500, "4.1")])},
        {"entry": _entry("b", 2, start_step=300), "log": _log([(500, "4.1")])},
    ]
    idx, logs = _fixture(tmp_path, runs)
    g = collect(load_runs(idx), logs, None)
    assert set(g[1000]) == {1}


def test_runs_without_a_seed_are_skipped(tmp_path):
    e = _entry("a", 1)
    del e["init_seed"]
    runs = [{"entry": e, "log": _log([(500, "4.1")])}]
    idx, logs = _fixture(tmp_path, runs)
    assert collect(load_runs(idx), logs, None) == {}


def test_train_steps_filter(tmp_path):
    runs = [
        {"entry": _entry("a", 1, steps=1000), "log": _log([(500, "4.1")], 1000)},
        {"entry": _entry("c", 1, steps=3250), "log": _log([(500, "3.9")], 3250)},
    ]
    idx, logs = _fixture(tmp_path, runs)
    g = collect(load_runs(idx), logs, 3250)
    assert set(g) == {3250}


def test_logfile_found_under_either_spelling(tmp_path):
    """Registry says logs/<id>.txt; the volume stores _logs/<id>.txt."""
    runs = [{"entry": _entry("a", 1), "log": _log([(500, "4.1")])}]
    idx, logs = _fixture(tmp_path, runs)
    alt = tmp_path / "logs"
    alt.mkdir()
    (alt / "a.txt").write_text(_log([(500, "4.1")]), encoding="utf-8")
    (logs / "a.txt").unlink()
    g = collect(load_runs(idx), logs, None)
    assert set(g[1000]) == {1}


def test_missing_index_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch"):
        load_runs(tmp_path / "absent.jsonl")
