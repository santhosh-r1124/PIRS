"""run.py: one command per stage, with a clear message for stages not built yet."""

import sys

import run


def test_spec_stages_exist():
    for stage in ["data", "train", "evaluate", "api", "dashboard", "test"]:
        assert stage in run.STAGES


def test_unknown_stage_prints_usage(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["run.py", "no-such-stage"])
    assert run.main() == 1
    assert "Stages:" in capsys.readouterr().out


def test_missing_step_reports_phase(monkeypatch, capsys):
    monkeypatch.setitem(run.STAGES, "fake", [("src/not_there.py", 9, ["echo"])])
    assert run.run_stage("fake", []) == 2
    assert "added in Phase 9" in capsys.readouterr().out


def test_existing_step_runs(monkeypatch):
    ok = [("run.py", 0, [sys.executable, "-c", "import sys; sys.exit(0)"])]
    monkeypatch.setitem(run.STAGES, "fake", ok)
    assert run.run_stage("fake", []) == 0


def test_all_stage_is_the_full_pipeline():
    steps = run.STAGES["all"]
    cmds = [" ".join(c[1:]) for _, _, c in steps]
    # order: data first, then training, then evaluation, then the recommender index
    order = [
        "src.data.generate_synthetic",
        "src.data.build_dataset",
        "src.train --quiet",
        "src.evaluate",
        "src.recommender.build_index",
        "src.recommender.recommend --evaluate",
    ]
    pos = [next(i for i, c in enumerate(cmds) if c.startswith(f"-m {o}")) for o in order]
    assert pos == sorted(pos)
    # main training (all models) + 3 ablation commands + 4 rolling-origin folds
    train = [c for c in cmds if "src.train" in c]
    assert len(train) == 1 + 3 + 4
    assert sum("--split fold" in c for c in train) == 4
    assert all((run.ROOT / req).exists() for req, _, _ in steps)


def test_gitignore_does_not_hide_source_code():
    """`models/` (unanchored) would also ignore src/models/ and break a clean clone."""
    lines = (run.ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert "/models/" in lines and "models/" not in lines
