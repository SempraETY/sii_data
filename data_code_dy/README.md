# 0518_dataBuilder_liubw

Symbolic-regression 数据生成工坊。产出与 dev 集 (`0518_test_liubw/data/task/dev/`) 同形态的 `samples.jsonl` + `images/`，下游可直接喂给：

- SFT/RL 转换器：`/inspire/qb-ilm2/project/26summer-camp-21/26210880/symreg_experiments/scripts/build_symreg_data.py`
- 评测脚本：`/inspire/qb-ilm2/project/26summer-camp-21/26210880/0518_test_liubw/eval.py`

## 项目分块

```
in_domain/        IND 数据生成器（已实现）
out_domain/       OOD 数据生成（占位，待 OOD 定义敲定）
data_analysis/    数据分布统计与对比工具
```

## Quickstart

```bash
pip install -r requirements.txt

# 生成 1000 条 IND 数据（5 难度各 200）
python in_domain/generate.py --n 1000 --seed 42 --out in_domain/output/ind_v1

# 与 dev 分布对比
python data_analysis/analyze.py in_domain/output/ind_v3/samples.jsonl \
    --compare /inspire/qb-ilm2/project/26summer-camp-21/26210880/0518_test_liubw/data/task/dev/samples.jsonl

# 转成 SFT/RL 训练 jsonl
python /inspire/qb-ilm2/project/26summer-camp-21/26210880/symreg_experiments/scripts/build_symreg_data.py \
    --samples in_domain/output/ind_v3/samples.jsonl \
    --out-dir in_domain/output/ind_v3/sft_rl
```

详见各子目录 README。
