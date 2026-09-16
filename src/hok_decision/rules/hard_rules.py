"""出装硬规则（不参与学习，直接约束候选生成）。

规则（用户确认）：
  1. 未携带召唤师技能「惩击」→ 禁止购买打野类装备；
  2. 辅助类装备只有在**局内角色输入为辅助/游走**时允许购买。
     角色在模型启动时手动输入，与英雄职业无关。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from hok_decision.config import NORMALIZED_ROOT

ROLE_OPTIONS = ("对抗路", "打野", "中路", "发育路", "游走", "辅助")
SUPPORT_ROLES = {"游走", "辅助"}
ROLE_INPUT_FILE = NORMALIZED_ROOT / "role_input.json"


@dataclass(frozen=True)
class PurchaseRuleEngine:
    """基于正式数据文件 + 手动局内角色输入的出装硬规则。"""

    summoners: dict[str, int] = field(default_factory=dict)  # 名称 -> id
    item_categories: dict[int, str] = field(default_factory=dict)  # item_id -> category

    @classmethod
    def load_defaults(cls) -> "PurchaseRuleEngine":
        summoners: dict[str, int] = {}
        item_categories: dict[int, str] = {}
        for filename, loader in (
            ("summoners.json", lambda d: summoners.update({s["summoner_name"]: int(s["summoner_id"]) for s in d["data"]})),
            ("items_vector.json", lambda d: item_categories.update({r["equipment_id"]: r.get("category", "normal") for r in d["data"]})),
        ):
            try:
                payload = json.loads((NORMALIZED_ROOT / filename).read_text(encoding="utf-8"))
                loader(payload)
            except (FileNotFoundError, KeyError, json.JSONDecodeError):
                continue
        return cls(summoners=summoners, item_categories=item_categories)

    def category_of(self, item_id: int) -> str:
        return self.item_categories.get(item_id, "normal")

    def can_buy(
        self,
        *,
        item_id: int,
        summoner_spell_names: list[str],
        role_input: str,
    ) -> tuple[bool, str]:
        """返回 (是否可买, 原因)。role_input ∈ ROLE_OPTIONS，启动时手动输入。"""
        category = self.category_of(item_id)
        if category == "jungle":
            if "惩击" not in summoner_spell_names:
                return False, "未携带惩击，不能购买打野装备"
            return True, ""
        if category == "support":
            if role_input not in SUPPORT_ROLES:
                return False, f"局内角色为{role_input or '未输入'}，非辅助/游走，不能购买辅助装备"
            return True, ""
        return True, ""


    def filter_item_candidates(
        self,
        *,
        item_ids: list[int],
        summoner_spell_names: list[str],
        role_input: str,
    ) -> tuple[list[int], list[dict[str, object]]]:
        """前置动作掩码：把非法装备从候选集中剔除（不进入模型/搜索）。

        返回 (allowed_item_ids, rejected_reports)。硬规则永远不参与训练。
        """
        allowed: list[int] = []
        rejected: list[dict[str, object]] = []
        for item_id in item_ids:
            ok, reason = self.can_buy(
                item_id=item_id,
                summoner_spell_names=summoner_spell_names,
                role_input=role_input,
            )
            if ok:
                allowed.append(item_id)
            else:
                rejected.append({"item_id": item_id, "reason": reason})
        return allowed, rejected


def write_role_input(role: str) -> None:
    """模型启动前手动输入局内角色，写入 role_input.json。"""
    if role not in ROLE_OPTIONS:
        raise ValueError(f"非法角色: {role}，可选 {ROLE_OPTIONS}")
    NORMALIZED_ROOT.mkdir(parents=True, exist_ok=True)
    ROLE_INPUT_FILE.write_text(
        json.dumps(
            {
                "version": "S45",
                "role": role,
                "note": "局内角色在模型启动时手动输入；辅助装规则以此为准",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_role_input() -> str | None:
    if not ROLE_INPUT_FILE.exists():
        return None
    try:
        payload = json.loads(ROLE_INPUT_FILE.read_text(encoding="utf-8"))
        role = payload.get("role")
        return role if role in ROLE_OPTIONS else None
    except (json.JSONDecodeError, OSError):
        return None
