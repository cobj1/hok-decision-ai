"""硬规则包（不参与训练，直接约束决策空间）。"""

from hok_decision.rules.hard_rules import (
    ROLE_OPTIONS,
    PurchaseRuleEngine,
    load_role_input,
    write_role_input,
)

__all__ = ["ROLE_OPTIONS", "PurchaseRuleEngine", "load_role_input", "write_role_input"]
