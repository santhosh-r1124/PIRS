"""Temporal models: output shapes, no information flow between modules, training loop."""

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the environment test already fails if torch is missing

from src.data.build_dataset import build_daily, chronological_split  # noqa: E402
from src.data.generate_synthetic import generate  # noqa: E402
from src.models.temporal import TemporalModel  # noqa: E402
from src.models.trainer import Objective, fit  # noqa: E402
from src.train import run_neural  # noqa: E402
from src.utils import load_config  # noqa: E402

B, L, N, F, H, C = 4, 28, 10, 16, 7, 5


def model(kind):
    torch.manual_seed(0)
    return TemporalModel(kind, N, F, C, H, max_lookback=L, d=16, layers=2, heads=2, ff=32)


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_output_shape(kind):
    m = model(kind).eval()
    out = m(torch.randn(B, L, N, F), torch.randn(B, H, C))
    assert out.shape == (B, H, N)
    short = m(torch.randn(B, 14, N, F), torch.randn(B, H, C))  # look-back ablation
    assert short.shape == (B, H, N)


@pytest.mark.parametrize("kind", ["lstm", "transformer"])
def test_modules_do_not_see_each_other(kind):
    """Temporal-only: changing module 3's history must not change any other module's output."""
    m = model(kind).eval()
    x, xf = torch.randn(B, L, N, F), torch.randn(B, H, C)
    x2 = x.clone()
    x2[:, :, 3] += 5.0
    with torch.no_grad():
        a, b = m(x, xf), m(x2, xf)
    others = [i for i in range(N) if i != 3]
    torch.testing.assert_close(a[:, :, others], b[:, :, others])
    assert not torch.allclose(a[:, :, 3], b[:, :, 3])


def test_objectives():
    y = np.array([0.0, 1.0, 3.0])
    reg = Objective("regression", y)
    assert reg.bias == pytest.approx(np.log(4 / 3))
    out = torch.tensor([0.0, float(np.log(2.0))])  # float32, like model outputs
    torch.testing.assert_close(reg.predict(out), torch.tensor([1.0, 2.0]))
    cls = Objective("classification", np.array([0, 0, 0, 1.0]))
    assert cls.pos_weight.item() == pytest.approx(3.0)
    assert 0 < cls.predict(torch.tensor([0.3])).item() < 1


@pytest.mark.parametrize("target", ["regression", "classification"])
def test_short_training_run(tmp_path, target):
    cfg = copy.deepcopy(load_config())
    cfg["paths"].update(results=str(tmp_path), checkpoints=str(tmp_path / "ckpt"))
    cfg["training"].update(max_epochs=3, patience=2)
    cfg["synthetic"]["n_days"] = 400
    d = generate(cfg, seed=1)
    daily = build_daily(
        d.incidents,
        d.signals,
        d.calendar,
        cfg["data"]["modules"],
        cfg["dataset"]["rolling_windows"],
    )
    split = chronological_split(len(daily.dates), 28, H, cfg["dataset"]["split"])
    rec = run_neural("lstm", cfg, daily, split, target, seed=1, verbose=False)
    assert 1 <= rec["training"]["best_epoch"] <= rec["training"]["epochs_run"] <= 3
    saved = json.loads((tmp_path / "runs" / f"lstm_{target}_seed1.json").read_text())
    assert saved["metrics"]["test"]["h1"] == rec["metrics"]["test"]["h1"]
    preds = np.load(tmp_path / "predictions" / f"lstm_{target}_seed1.npz")
    assert preds["test"].shape == (len(split.test), H, N) and np.all(preds["test"] >= 0)
    assert (tmp_path / "ckpt" / f"lstm_{target}_seed1.pt").exists()


def test_resume_gives_identical_result(tmp_path):
    """Pausing after every epoch and resuming must give exactly the uninterrupted result."""
    rng = np.random.default_rng(0)
    data = {
        p: {
            "X": rng.normal(size=(n, 14, N, F)).astype("float32"),
            "Xf": rng.normal(size=(n, H, C)).astype("float32"),
            "Y": rng.poisson(1.0, size=(n, H, N)).astype("float32"),
        }
        for p, n in [("train", 40), ("val", 12)]
    }
    tcfg = dict(load_config()["training"], max_epochs=4, patience=100, batch_size=8)

    def fresh():  # dropout > 0, so the random-number state must be restored too
        torch.manual_seed(0)
        return TemporalModel(
            "transformer", N, F, C, H, max_lookback=14, d=8, layers=1, heads=2, ff=16, dropout=0.2
        )

    a = fresh()
    sa = fit(a, data, "regression", tcfg, seed=3, verbose=False)
    calls, ck = 0, tmp_path / "partial.pt"
    while True:
        b = fresh()  # a new process would also build a new model before resuming
        sb = fit(b, data, "regression", tcfg, seed=3, verbose=False, checkpoint=ck, time_budget=0.0)
        calls += 1
        if sb["status"] == "done":
            break
    assert calls == 4  # paused after epochs 1, 2, 3; finished after epoch 4
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
    assert sa["history"] == sb["history"]
