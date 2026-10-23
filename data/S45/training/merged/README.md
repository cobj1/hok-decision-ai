# training/merged/ —— 中间产物，不要直接用于训练

本目录是 `merge_wzryagnt_frames.py` / `merge_tencent_rt_frames.py` 的**输出中间态**，
唯一消费者是 `scripts/build_hokdata.py`（它会把这里的内容规整成 `hokdata/` 下的统一数据集）。

## 权威关系

| 位置 | 角色 |
|---|---|
| **本目录** `wzry_frames.jsonl` | 上游·可重跑的中间产物 |
| `hokdata/behavior/dataset_behavior_v2.jsonl` | 下游·**消费方应读这一份** |

两者内容应当逐字节一致（`build_hokdata.py` 用 `shutil.copy2` 镜像）。若不一致，
先重跑 `python scripts/build_hokdata.py`，**不要直接改 `hokdata` 那份**。

随时校验：

```bash
python D:/work/aiKings/hokdata/check_mirror.py
```

> 2026-10-24 曾出现漂移（11 055 vs 11 043 行），因为上游被重跑但未重新镜像。
> 当前两份均为 59 795 701 B / 11 043 行，md5 `e821e10ad82f`。

## 注意

- 本目录**不在训练输入路径上**：`win_train_attention_real.py` 读的是 `hokdata/dataset/dataset_final_v4.jsonl`。
- 文件较大（57 MB / 份），已被 `.gitignore` 排除，不入库。
