# hok-decision-ai

王者荣耀对局决策建议系统的数据层与模拟环境：

- 爬取并结构化官方英雄/装备/召唤师技能数据与 5v5 王者峡谷地图资源信息；
- 以版本标签（当前锁定 S45）管理全部数据，数值与代码低耦合；
- 提供确定性模拟引擎：移动、普攻、伤害结算（我方无暴击下界 / 敌方期望或必暴击）。

## 目录

```
src/hok_decision/
  config.py            版本标签与数据源 URL
  schema/models.py     结构化数据模型（Pydantic v2）
  crawlers/            官方 JSON / 英雄数值与技能 / 营地导入 / 地图机制 / 地形图源
  simulator/           地图加载、伤害公式、单位、资源刷新调度与步进引擎
  search/              P2 搜索骨架：宏观候选动作 + 束搜索 + 可校准价值函数
  model/               Transformer 模型（Win torch：注意力二阶权重 + 下一决策 token）
data/S45/
  raw/                 原始抓取产物（官方地图图片资产、技能页、二手数值源、地图候选图）
  normalized/          归一化 JSON（英雄/装备/召唤师技能/地图/英雄技能/英雄数值/地形锚点）
tests/                 43 项测试（伤害手算、地图加载、引擎确定性、资源调度、搜索、价值校准、上下文价值）
docs/tech-plan-v0.1.md 技术方案
```

## 使用

```bash
pip install -e ".[dev]"

# 爬取全部数据（版本标签见 src/hok_decision/config.py；hero_stats 源较旧且网络请求多，耗时较长）
PYTHONPATH=src python3 -m hok_decision.cli crawl all

# 按数据源分开爬取
PYTHONPATH=src python3 -m hok_decision.cli crawl official
PYTHONPATH=src python3 -m hok_decision.cli crawl map
PYTHONPATH=src python3 -m hok_decision.cli crawl hero_stats
PYTHONPATH=src python3 -m hok_decision.cli crawl camp
PYTHONPATH=src python3 -m hok_decision.cli crawl terrain
PYTHONPATH=src python3 -m hok_decision.cli crawl timings

# 查看已入库数据
PYTHONPATH=src python3 -m hok_decision.cli status

# 校准价值公式权重（简化1v1模拟对局 logistic 拟合，写入 value_weights.json）
PYTHONPATH=src python3 -m hok_decision.cli calibrate

# 运行测试
python3 -m pytest -q
```

## 数据现状与已知缺口

- 英雄技能：132/132，当前使用王者营地本地抓取（updateTime=20260827，538 条，verified）；
- **装备/技能向量**：`equip_details.json`（126 件官方游戏内装备详情：属性+主动/被动+适用英雄+合成）→ `items_vector.json`（20 维属性 + 20 维效果 one-hot）；`skills_vector.json`（538 技能，冷却/伤害成长表 + 20 维效果 one-hot）；
- 英雄基础数值：**132/132 全部 verified**，来源王者营地本地抓取（`crawl camp` 可重跑）；攻击范围只有"近程/远程"标签，数值待标定；
- 地图锚点：**40 个点已由人工在小地图（416×424，左上原点）上标定并写入 `terrain_anchors.json`**（18 塔 + 12 小型资源 + 4 Buff + 2 龙 + 2 水晶 + 2 中立特殊），`storm_dragon` 仍为待标定；
- 坐标为小地图像素坐标，**尚未换算为游戏地图单位**（比例尺待录屏/游戏内测距标定）；
- 资源点刷新时间已写入 `map.json`（开局/刷新间隔/动态规则），来源为公开攻略（9game S23 表/远古生物/野区介绍），需随赛季版本复核；
- `ResourceScheduler` 已接入模拟引擎：支持 static/fixed/once/timed_window/dynamic 五类刷新规则，**风暴龙王 20 分钟后双龙坑随机刷新**用 dynamic + 实时小地图观测回填实现；
- P2 搜索使用**可校准价值函数**：权重存于 `value_weights.json`（logistic 校准，400 局简化模拟，准确率约 0.85），planner 自动加载；
- 新增**上下文价值函数**（`value_context.ContextualValue`）：`value_rules.json` 按游戏阶段（前期/中期/后期/大后期）、英雄职业（经济需求/击杀价值）、敌方输出（血量差权重）、塔级（一塔/二塔/高地塔/水晶）调节权重，规则来自用户思路文档；
- **Transformer 架构**（Win torch cu128）：注意力二阶权重模型 + GPT 式下一决策 token 模型，产物在 `data/S45/models/`，训练入口 `scripts/win_torch_demo.py`（Win 运行）；
- `raw/map_images/` 有完整地图候选图源与比例尺二手线索（均未证实），供后续配准；
- 刷新/持续时间来自公开机制文章（已标注来源，需随赛季公告复核）。
