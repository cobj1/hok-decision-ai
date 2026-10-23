r"""线性价值基线（严格按局切分，S2-1）。

用途：在注意力模型过拟合/负收益时，提供一个可复现的线性 logistic 基线。
输入特征 = CONTEXT_FEATURES(13) + VALUE_FEATURES(10)，只用 train 的均值/方差标准化。
输出：<HOKDATA_ROOT>/value_linear_report.json

运行：
    HOKDATA_ROOT=D:/work/aiKings/hokdata python scripts/win_train_linear_baseline.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("HOKDATA_ROOT", str(PROJECT_ROOT.parent / "hokdata"))

from hok_decision.model.attention_value import CONTEXT_FEATURES, VALUE_FEATURES  # noqa: E402

HOKDATA_ROOT = Path(os.environ["HOKDATA_ROOT"])


def _load_train_module():
    path = PROJECT_ROOT / "scripts" / "win_train_attention_real.py"
    spec = importlib.util.spec_from_file_location("win_train_attention_real", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--out", default=str(HOKDATA_ROOT / "value_linear_report.json"))
    parser.add_argument("--model-out", default=str(HOKDATA_ROOT / "value_linear_model.json"))
    args = parser.parse_args()

    module = _load_train_module()
    c, f, y, report, bvs = module._build_dataset()
    splits = json.loads((HOKDATA_ROOT / "splits.json").read_text(encoding="utf-8"))
    game_splits = {**splits.get("tencent_rt_base", {}), **splits.get("tencent_rt", {})}

    def split_of(bv: str) -> str | None:
        return game_splits.get(bv) or game_splits.get(str(bv).split("#", 1)[0])

    train_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "train"]
    val_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "val"]
    test_idx = [i for i, bv in enumerate(bvs) if split_of(bv) == "test"]
    if not train_idx or not val_idx or not test_idx:
        raise SystemExit(f"split failed: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    x = torch.cat([c, f], dim=1)
    train_t = torch.tensor(train_idx, dtype=torch.long)
    mu = x[train_t].mean(dim=0)
    sigma = x[train_t].std(dim=0).clamp_min(1e-6)
    x = (x - mu) / sigma

    def accuracy(ids, weights, bias) -> float:
        tensor = torch.tensor(ids, dtype=torch.long)
        with torch.no_grad():
            pred = (torch.sigmoid(x[tensor] @ weights + bias) > 0.5).float()
            return float((pred == y[tensor]).float().mean())

    def majority(ids) -> float:
        tensor = torch.tensor(ids, dtype=torch.long)
        rate = float(y[tensor].mean())
        return max(rate, 1.0 - rate)

    per_seed = []
    best = None
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        weights = torch.zeros(x.shape[1], requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([weights, bias], lr=args.lr, weight_decay=args.weight_decay)
        for _ in range(args.steps):
            idx = train_t[torch.randint(0, len(train_t), (64,))]
            logits = x[idx] @ weights + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_acc = accuracy(val_idx, weights, bias)
        row = {
            "seed": seed,
            "val_acc": val_acc,
            "val_majority": majority(val_idx),
            "test_acc": accuracy(test_idx, weights, bias),
            "test_majority": majority(test_idx),
        }
        per_seed.append(row)
        if best is None or val_acc > best["val_acc"]:
            best = {
                "val_acc": val_acc,
                "weights": weights.detach().clone(),
                "bias": bias.detach().clone(),
            }
        print(
            f"seed {seed}: val={row['val_acc']:.4f}/{row['val_majority']:.4f} "
            f"test={row['test_acc']:.4f}/{row['test_majority']:.4f}",
            flush=True,
        )

    result = {
        "model": "linear_logistic",
        "features": CONTEXT_FEATURES + VALUE_FEATURES,
        "samples": int(x.shape[0]),
        "train_frames": len(train_idx),
        "val_frames": len(val_idx),
        "test_frames": len(test_idx),
        "train_games": len({bvs[i] for i in train_idx}),
        "val_games": len({bvs[i] for i in val_idx}),
        "test_games": len({bvs[i] for i in test_idx}),
        "weight_decay": args.weight_decay,
        "steps": args.steps,
        "seeds": per_seed,
        "mean_val_acc": sum(row["val_acc"] for row in per_seed) / len(per_seed),
        "mean_val_majority": sum(row["val_majority"] for row in per_seed) / len(per_seed),
        "mean_test_acc": sum(row["test_acc"] for row in per_seed) / len(per_seed),
        "mean_test_majority": sum(row["test_majority"] for row in per_seed) / len(per_seed),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    model_payload = {
        "model": "linear_logistic",
        "features": CONTEXT_FEATURES + VALUE_FEATURES,
        "mean": mu.tolist(),
        "sigma": sigma.tolist(),
        "weights": best["weights"].tolist(),
        "bias": float(best["bias"]),
        "selected_by": "val_acc",
        "val_acc": best["val_acc"],
        "test_acc": result["mean_test_acc"],
        "test_majority": result["mean_test_majority"],
    }
    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(json.dumps(model_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"mean val={result['mean_val_acc']:.4f}/{result['mean_val_majority']:.4f} "
        f"test={result['mean_test_acc']:.4f}/{result['mean_test_majority']:.4f}"
    )
    print("saved:", out)
    print("model:", model_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
