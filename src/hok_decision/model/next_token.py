"""GPT 式"预测下一个决策 token"（torch，Win GPU 环境运行）。

把当前局势编码为 token 序列：
  [CLS, 时间桶, 经济饱和桶, 敌方输出桶, 己方血量桶, 塔差桶, 龙桶]
用带因果掩码的 Transformer 预测下一个 token = 宏观决策动作
（attack / retreat / farm / dragon / tower）。

这是后续接入真实对局序列训练的动作先验/策略头骨架。
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

PAD_TOKEN = 0
CLS_TOKEN = 1
NUM_BUCKETS = 8
STATE_DIM = 13
ACTION_TOKENS: dict[str, int] = {
    "attack": 50,
    "retreat": 51,
    "farm": 52,
    "dragon": 53,
    "tower": 54,
}
VOCAB_SIZE = 2 + STATE_DIM * NUM_BUCKETS
MAX_LEN = 16


def encode_situation(context: torch.Tensor) -> torch.Tensor:
    """context: (B,6) 各维 0~1 → token 序列 (B, 1+6)。"""
    batch = context.shape[0]
    buckets = (context.clamp(0.0, 0.999999) * NUM_BUCKETS).long()
    state_tokens = 2 + torch.arange(STATE_DIM, device=context.device) * NUM_BUCKETS + buckets
    cls = torch.full((batch, 1), CLS_TOKEN, dtype=torch.long, device=context.device)
    return torch.cat([cls, state_tokens], dim=1)


def rule_action(context: torch.Tensor) -> torch.Tensor:
    """合成数据的"专家动作"规则，用于验证训练管线。"""
    own_hp = context[:, 3]
    econ_sat = context[:, 1]
    time_norm = context[:, 0]
    dragon = context[:, 5]
    tower_adv = context[:, 4]
    action = torch.full((context.shape[0],), ACTION_TOKENS["attack"], dtype=torch.long, device=context.device)
    action = torch.where(own_hp < 0.3, torch.full_like(action, ACTION_TOKENS["retreat"]), action)
    action = torch.where((econ_sat < 0.5) & (own_hp >= 0.3), torch.full_like(action, ACTION_TOKENS["farm"]), action)
    action = torch.where((time_norm > 0.6) & (dragon > 0.5) & (own_hp >= 0.5), torch.full_like(action, ACTION_TOKENS["dragon"]), action)
    action = torch.where((tower_adv < -0.3) & (own_hp >= 0.5), torch.full_like(action, ACTION_TOKENS["tower"]), action)
    return action


class NextTokenTransformer(nn.Module):
    """因果 Transformer：局势 token → 下一决策 token。"""

    def __init__(
        self,
        *,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 64,
        n_head: int = 4,
        n_layers: int = 2,
        max_len: int = MAX_LEN,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self._causal: torch.Tensor | None = None

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device) * float("-inf"),
            diagonal=1,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (B,T)。返回 (B,T,vocab) logits（带因果掩码）。"""
        batch, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.encoder(x, mask=self._causal_mask(length, input_ids.device))
        return self.head(x)

    @torch.no_grad()
    def predict_next(self, input_ids: torch.Tensor) -> torch.Tensor:
        """取最后一个位置的 logits → 下一 token id。"""
        logits = self.forward(input_ids)
        return logits[:, -1].argmax(dim=-1)

    def save_artifacts(self, path: Path, metrics: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)


def train_demo(
    *,
    n_samples: int = 8192,
    steps: int = 1200,
    batch_size: int = 128,
    device: torch.device | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """训练下一决策 token 预测模型，返回指标。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260828)
    model = NextTokenTransformer().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    context = torch.rand(n_samples, STATE_DIM, device=device)
    input_ids = encode_situation(context).to(device)
    targets = rule_action(context).to(device)

    for _ in range(steps):
        idx = torch.randint(0, n_samples, (batch_size,), device=device)
        logits = model(input_ids[idx])
        loss = loss_fn(logits[:, -1], targets[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        pred = logits[:, -1].argmax(dim=-1)
        acc = (pred == targets).float().mean().item()
        final_loss = loss_fn(logits[:, -1], targets).item()

    metrics = {
        "device": str(device),
        "steps": steps,
        "n_samples": n_samples,
        "accuracy": acc,
        "final_loss": final_loss,
    }
    if artifact_dir is not None:
        model.save_artifacts(artifact_dir / "next_token.pt", metrics)
    return metrics
