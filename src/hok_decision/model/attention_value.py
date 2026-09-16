"""Transformer 注意力二阶权重函数（torch，Win GPU 环境运行）。

思路：
  一阶特征 F(s) = [hp_diff, economy_diff, small_camp_diff,
                    red_buff_diff, blue_buff_diff,
                    tyrant_diff, overlord_diff, storm_dragon_diff,
                    tower_diff, kill_diff]
  二阶权重 w(s) = base_w * (1 + gamma * softmax(Q(s)·K/√d))
     Q = 局势上下文编码（时间、经济饱和度、敌方输出、己方血量、
         塔差、龙可用、己方普攻强度、己方技能强度）
     K = 每个价值维度的可学习键向量
  V(s) = Σ w_j(s)·F_j(s)

训练目标：logistic 拟合胜负标签；w(s) 随局势/阵容变化的形态是训练重点。
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

CONTEXT_FEATURES: list[str] = [
    "time_norm",        # t/900，15分钟=1
    "econ_sat",         # min(1, max(time_norm, 平均经济/12000))
    "enemy_threat",     # 敌方平均输出/300
    "own_hp_frac",      # 己方平均血量比例
    "tower_adv",        # 敌方塔损-我方塔损（塔级加权）
    "dragon_avail",     # 龙是否可争夺
    "team_aa_power",    # 己方普攻强度（红buff收益）
    "team_caster_power",  # 己方技能强度（蓝buff收益）
    "build_attack_power",  # 出装攻击强度（来自 items_vector 属性向量）
    "build_defense_power", # 出装防御强度
    "build_utility_power", # 出装功能强度（冷却/移速/吸血/特效）
    "skill_power",         # 技能强度（来自 skills_vector 伤害/效果向量）
    "match_level",         # 对局层级 0~1（青铜→荣耀），前置分层模块输出
]
VALUE_FEATURES: list[str] = [
    "hp_diff",
    "economy_diff",
    "small_camp_diff",
    "red_buff_diff",
    "blue_buff_diff",
    "tyrant_diff",
    "overlord_diff",
    "storm_dragon_diff",
    "tower_diff",
    "kill_diff",
]

DEFAULT_BASE_WEIGHTS = [4.8, 100.0, 50.0, 20.0, 20.0, 15.0, 25.0, 50.0, 100.0, 2.7]


class AttentionValueModel(nn.Module):
    """Q=局势，K=价值维度，softmax attention 输出动态权重。"""

    def __init__(
        self,
        *,
        context_dim: int = len(CONTEXT_FEATURES),
        feature_dim: int = len(VALUE_FEATURES),
        d_model: int = 32,
        context_depth: int = 0,
        base_weights: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.context_dim = context_dim
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.context_depth = context_depth
        self.query_proj = nn.Linear(context_dim, d_model)
        if context_depth > 0:
            self.context_mlp = nn.Sequential(
                *[
                    layer
                    for _ in range(context_depth)
                    for layer in (nn.Linear(d_model, d_model), nn.ReLU())
                ]
            )
        else:
            self.context_mlp = nn.Identity()
        self.keys = nn.Parameter(torch.empty(feature_dim, d_model).normal_(0.0, 0.1))
        init = base_weights or DEFAULT_BASE_WEIGHTS
        self.base_weights = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.5))
        self.bias = nn.Parameter(torch.zeros(1))

    def dynamic_weights(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (w(s), attention)。w 形状 (B, feature_dim)。"""
        q = self.context_mlp(self.query_proj(context))  # (B,d)
        scores = q @ self.keys.t() / math.sqrt(self.d_model)
        attn = torch.softmax(scores, dim=-1)
        weights = self.base_weights * (1.0 + self.gamma * attn)
        return weights, attn

    def forward(self, context: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        weights, _ = self.dynamic_weights(context)
        return (weights * features).sum(dim=-1) + self.bias

    def save_artifacts(self, path: Path, metrics: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        summary = path.with_suffix(".json")
        summary.write_text(
            json.dumps(
                {
                    "context_features": CONTEXT_FEATURES,
                    "value_features": VALUE_FEATURES,
                    "d_model": self.d_model,
                    "metrics": metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


def make_training_data(
    n: int = 8192, *, seed: int = 20260828, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """合成训练数据：验证训练管线与目标权重形态。

    关键规则：
    - 经济在饱和（econ_sat→1，即 15 分钟/满经济）后权重归零；
    - 红buff 收益随己方普攻强度，蓝buff 随己方技能强度；
    - 暴君/主宰随时间升值，主宰（兵线/推塔）略高于暴君（伤害）；
    - 风暴龙王 20 分钟后出现且价值最高。
    """
    torch.manual_seed(seed)
    c = torch.rand(n, len(CONTEXT_FEATURES), device=device)
    f = torch.randn(n, len(VALUE_FEATURES), device=device)
    time_norm = c[:, 0]
    # econ_sat 与时间独立采样，保证模型能学到"饱和→经济降权"的独立效应
    econ_sat = torch.rand(n, device=device)
    c[:, 1] = econ_sat
    c[:, 6] = 0.4 + torch.rand(n, device=device) * 1.1
    c[:, 7] = 0.4 + torch.rand(n, device=device) * 1.1
    # loadout 特征（来自装备/技能向量，独立采样）
    c[:, 8] = torch.rand(n, device=device) * 2.0
    c[:, 9] = torch.rand(n, device=device) * 2.0
    c[:, 10] = torch.rand(n, device=device) * 2.0
    c[:, 11] = torch.rand(n, device=device) * 2.0
    c[:, 12] = torch.rand(n, device=device)  # match_level 0~1

    build_attack = c[:, 8]
    build_defense = c[:, 9]
    skill_power = c[:, 11]
    level = c[:, 12]
    after_20 = (time_norm > 1200.0 / 900.0).float()
    logit = (
        4.0 * f[:, 0] * (0.7 + 0.6 * build_attack)       # 血量差：输出越强权重越高
        + 80.0 * f[:, 1] * (1.0 - econ_sat) * (1.3 - 0.5 * level)  # 经济：低分段更值钱
        + 6.0 * f[:, 2] * (1.0 - econ_sat) * (1.3 - 0.5 * level)  # 小野
        + 20.0 * f[:, 3] * c[:, 6]                       # 红buff：看己方普攻强度
        + 20.0 * f[:, 4] * c[:, 7]                       # 蓝buff：看己方技能强度
        + 15.0 * f[:, 5] * (0.6 + 0.8 * time_norm) * (0.7 + 0.5 * level)  # 暴君：高分段更抢
        + 25.0 * f[:, 6] * (0.6 + 1.0 * time_norm) * (0.7 + 0.5 * level)  # 主宰
        + 50.0 * f[:, 7] * after_20                      # 风暴龙王
        + 30.0 * f[:, 8] * (0.6 + 1.2 * time_norm) * (0.7 + 0.5 * build_defense)  # 推塔
        + 3.0 * f[:, 9] * (0.5 + 0.75 * skill_power) * (0.6 + 0.8 * level)  # 击杀：高分段更值钱
    )
    y = (torch.sigmoid(logit) > torch.rand(n, device=device)).float()
    return c, f, y


def train_demo(
    *,
    steps: int = 2500,
    batch_size: int = 256,
    device: torch.device | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, Any]:
    """训练注意力价值模型，返回指标与权重形态。"""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionValueModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    c, f, y = make_training_data(device=device)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        idx = torch.randint(0, c.shape[0], (batch_size,), device=device)
        optimizer.zero_grad()
        out = model(c[idx], f[idx])
        loss = loss_fn(out, y[idx])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        out = model(c, f)
        pred = (torch.sigmoid(out) > 0.5).float()
        acc = (pred == y).float().mean().item()

        def ctx(**overrides: float) -> torch.Tensor:
            tensor = torch.zeros(1, len(CONTEXT_FEATURES), device=device)
            for key, value in overrides.items():
                tensor[0, CONTEXT_FEATURES.index(key)] = value
            return tensor

        w_low, _ = model.dynamic_weights(ctx(econ_sat=0.0))
        w_high, _ = model.dynamic_weights(ctx(econ_sat=1.0))
        w_aa_low, _ = model.dynamic_weights(ctx(team_aa_power=0.4))
        w_aa_high, _ = model.dynamic_weights(ctx(team_aa_power=1.5))
        w_caster_low, _ = model.dynamic_weights(ctx(team_caster_power=0.4))
        w_caster_high, _ = model.dynamic_weights(ctx(team_caster_power=1.5))
        w_att_low, _ = model.dynamic_weights(ctx(build_attack_power=0.0))
        w_att_high, _ = model.dynamic_weights(ctx(build_attack_power=2.0))
        w_skill_low, _ = model.dynamic_weights(ctx(skill_power=0.0))
        w_skill_high, _ = model.dynamic_weights(ctx(skill_power=2.0))
        w_level_low, _ = model.dynamic_weights(ctx(match_level=0.0))
        w_level_high, _ = model.dynamic_weights(ctx(match_level=1.0))

    metrics = {
        "device": str(device),
        "steps": steps,
        "accuracy": acc,
        "economy_weight_low_sat": w_low[0, 1].item(),
        "economy_weight_high_sat": w_high[0, 1].item(),
        "economy_weight_ratio": (w_high[0, 1] / (w_low[0, 1] + 1e-6)).item(),
        "red_buff_weight_low_aa": w_aa_low[0, 3].item(),
        "red_buff_weight_high_aa": w_aa_high[0, 3].item(),
        "blue_buff_weight_low_caster": w_caster_low[0, 4].item(),
        "blue_buff_weight_high_caster": w_caster_high[0, 4].item(),
        "hp_weight_low_attack": w_att_low[0, 0].item(),
        "hp_weight_high_attack": w_att_high[0, 0].item(),
        "kill_weight_low_skill": w_skill_low[0, 9].item(),
        "kill_weight_high_skill": w_skill_high[0, 9].item(),
        "economy_weight_low_level": w_level_low[0, 1].item(),
        "economy_weight_high_level": w_level_high[0, 1].item(),
        "tyrant_weight_low_level": w_level_low[0, 5].item(),
        "tyrant_weight_high_level": w_level_high[0, 5].item(),
        "kill_weight_low_level": w_level_low[0, 9].item(),
        "kill_weight_high_level": w_level_high[0, 9].item(),
    }
    if artifact_dir is not None:
        model.save_artifacts(artifact_dir / "attention_value.pt", metrics)
    return metrics
