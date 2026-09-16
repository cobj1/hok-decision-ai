"""层数堆叠消融（Win torch 运行，不保存模型）。

用法:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_depth_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.model.attention_value import (  # noqa: E402
    AttentionValueModel,
    make_training_data,
)
from hok_decision.model.next_token import (  # noqa: E402
    NextTokenTransformer,
    encode_situation,
    rule_action,
)


def run_attention_depth_ablation(device: torch.device) -> None:
    c, f, y = make_training_data(n=6144, device=device)
    print("\n=== 注意力二阶权重：context 编码深度 ===")
    print("depth | acc")
    for depth in (0, 1, 2):
        torch.manual_seed(20260828)
        model = AttentionValueModel(context_depth=depth).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=0.02)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(1200):
            idx = torch.randint(0, c.shape[0], (256,), device=device)
            opt.zero_grad()
            out = model(c[idx], f[idx])
            loss = loss_fn(out, y[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(c, f)) > 0.5).float()
            acc = (pred == y).float().mean().item()
        print(f"{depth:>5} | {acc:.4f}")


def run_token_depth_ablation(device: torch.device) -> None:
    torch.manual_seed(20260828)
    context = torch.rand(4096, 12, device=device)
    tokens = encode_situation(context)
    targets = rule_action(context)
    print("\n=== 下一决策 token：Transformer 层数 ===")
    print("layers | acc    | loss")
    for layers in (1, 2, 4):
        torch.manual_seed(20260828)
        model = NextTokenTransformer(n_layers=layers).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(800):
            idx = torch.randint(0, tokens.shape[0], (128,), device=device)
            logits = model(tokens[idx])
            loss = loss_fn(logits[:, -1], targets[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(tokens)
            pred = logits[:, -1].argmax(dim=-1)
            acc = (pred == targets).float().mean().item()
            final_loss = loss_fn(logits[:, -1], targets).item()
        print(f"{layers:>6} | {acc:.4f} | {final_loss:.4f}")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device}")
    run_attention_depth_ablation(device)
    run_token_depth_ablation(device)
    print("\nDEPTH_ABLATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
