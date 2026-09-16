r"""Win 环境 torch 模型自测脚本（用 E:\py312\python.exe 运行）。

运行:
    E:\py312\python.exe E:\project\hok-decision-ai\scripts\win_torch_demo.py
验证:
    - 注意力二阶权重模型：训练收敛、经济权重随饱和度下降
    - GPT 式下一决策 token 模型：训练收敛、预测合法动作
产物:
    data/S45/models/attention_value.pt(+.json)
    data/S45/models/next_token.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hok_decision.model.attention_value import (  # noqa: E402
    CONTEXT_FEATURES,
    VALUE_FEATURES,
    AttentionValueModel,
    train_demo as train_attention_demo,
)
from hok_decision.model.next_token import (  # noqa: E402
    ACTION_TOKENS,
    STATE_DIM,
    NextTokenTransformer,
    encode_situation,
    rule_action,
    train_demo as train_next_token_demo,
)


def check_shapes(device: torch.device) -> None:
    context = torch.rand(4, len(CONTEXT_FEATURES), device=device)
    features = torch.randn(4, len(VALUE_FEATURES), device=device)
    model = AttentionValueModel().to(device)
    weights, attn = model.dynamic_weights(context)
    assert weights.shape == (4, len(VALUE_FEATURES))
    assert attn.shape == (4, len(VALUE_FEATURES))
    assert torch.allclose(attn.sum(dim=-1), torch.ones(4, device=device), atol=1e-5)
    value = model(context, features)
    assert value.shape == (4,)

    tokens = encode_situation(context)
    assert tokens.shape == (4, 1 + STATE_DIM)
    transformer = NextTokenTransformer().to(device)
    logits = transformer(tokens)
    assert logits.shape == (4, 1 + STATE_DIM, transformer.head.out_features)
    pred = transformer.predict_next(tokens)
    assert pred.shape == (4,)
    assert all(int(token) in set(ACTION_TOKENS.values()) for token in rule_action(context))
    print("shape checks OK")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch={torch.__version__} device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")
    check_shapes(device)

    artifact_dir = PROJECT_ROOT / "data" / "S45" / "models"
    metrics_a = train_attention_demo(device=device, artifact_dir=artifact_dir)
    print("attention metrics:", metrics_a)
    assert metrics_a["accuracy"] > 0.75, metrics_a
    # 用户指定的目标形态
    assert metrics_a["economy_weight_ratio"] < 1.0, metrics_a  # 经济饱和后降权
    assert metrics_a["red_buff_weight_high_aa"] > metrics_a["red_buff_weight_low_aa"], metrics_a
    assert metrics_a["blue_buff_weight_high_caster"] > metrics_a["blue_buff_weight_low_caster"], metrics_a
    # 装备/技能向量作为权重：高攻出装→血量差权重升；高技能→击杀权重升
    assert metrics_a["hp_weight_high_attack"] > metrics_a["hp_weight_low_attack"], metrics_a
    assert metrics_a["kill_weight_high_skill"] > metrics_a["kill_weight_low_skill"], metrics_a
    # 对局水平分层：低分段经济权重高；高分段暴君/击杀权重高
    assert metrics_a["economy_weight_low_level"] > metrics_a["economy_weight_high_level"], metrics_a
    assert metrics_a["tyrant_weight_high_level"] > metrics_a["tyrant_weight_low_level"], metrics_a
    assert metrics_a["kill_weight_high_level"] > metrics_a["kill_weight_low_level"], metrics_a

    metrics_t = train_next_token_demo(device=device, artifact_dir=artifact_dir)
    print("next_token metrics:", metrics_t)
    assert metrics_t["accuracy"] > 0.85, metrics_t
    print("TORCH_DEMO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
