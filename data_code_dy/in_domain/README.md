# in_domain — 工作逻辑

生成与 dev 集 (`0518_test_liubw/data/task/dev/`) 同形态的 IND 符号回归数据集。下游可直接喂给：

- `symreg_experiments/scripts/build_symreg_data.py` 做 SFT/RL 训练数据转换
- `0518_test_liubw/eval.py` 做评测

## 一条样本是怎么来的

```
   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌────────────────┐
   │ dev_distribution │ →  │  expressions.py  │ →  │ sampling.py  │ →  │ rendering.py   │
   │ 学 dev 经验分布  │    │ 按权重选函数+模板│    │ 采 ref/test  │    │ 画 590×390 PNG │
   └──────────────────┘    └──────────────────┘    └──────────────┘    └────────────────┘
                                       │                    │
                                       └─→ generate.py 把后面三步串成 build_one()
                                           ├─ 失败重抽（最多 max_retries 次）
                                           └─ 成功后 写一行 JSONL + 存一张 PNG
```

`dev_distribution.py` 在导入时一次性读 dev 集，按难度提取每函数 / 每结构元素的经验概率，并对 13-函数全集做 Laplace +0.1 平滑（dev 60 条里没出过的函数仍以 ~0.2% 概率出现，符合"不出现 ≠ 不能出"的原则）。

`build_one(diff, sid, rng, out_dir, max_retries, template)` 的循环：

1. **挑表达式骨架**：`expressions.sample_expression(diff, rng)` 调用难度专属模板（`_easy / _medium / _hard / _expert / _extreme`），其中函数槽位通过 `dev_distribution.pick_function(weights, rng, among=SAFE_FUNCS)` 加权采样；是否嵌 `abs / log / sqrt / exp(-x²) / pow` 由 `dev_distribution.structural_flags(diff)` 给的 dev 经验概率决定；top-level term 数从 dev `term_pmf` 采。
2. **采 x 窗口 + 采点**：`sampling.pick_ref_window` / `pick_test_window`，ref 6–20 点（非均匀），test 50 点（完美均匀）。
3. **求值 + 验证**：`eval(expr, {"x":x, "np":np, "__builtins__":{}})` 沙箱 → 三连验证（`isfinite` / `var(y_test) > 1e-6` / `max|y| ≤ 1e6`）。
4. **失败 → 回到 1**，最多 `--max-retries`（默认 50）次。
5. **成功 → 渲染**：`rendering.render_png` 用 `figsize=(5.9,3.9), dpi=100` 输出 590×390 RGBA PNG，白底 + 浅灰网格 (`#ededed`) + steelblue 曲线（`linewidth=3.0`）+ x/y 轴标签，视觉与 dev 图对齐（曲线宽度 / 网格灰像素 / 黑色刻度像素三项偏差均 <1.2k）。命名 `images/ind_<diff>_<id4>.png`。
6. **生成 hints**：`expressions.make_function_hints(true_hints, rng, diff)` 调 `dev_distribution.sample_distractors`，按难度从 dev `hint_len_pmf` 采长度，按 dev `prompt_hint_freq` 加权采干扰项。
7. **渲染 prompt**：`prompt_template.render_prompt(record, template)` 用 `prompt.txt` 生成最终用户文本，字段替换规则与 `0518_test_liubw/eval.py` 的 `build_message` 完全一致（`function_hints` / `data_points` / `axis_note` 三个占位符，值已自带前缀）。
8. **拼记录**：返回完整 dict（10 个字段，多了 `prompt`）。

`generate.py` 的主循环按 `--counts`（默认五难度各 200）顺序调 `build_one`，逐行 flush 写 `samples.jsonl`，进度每 50 条打一次。

## 为什么这样设计分布

完全从 dev 集 (`0518_test_liubw/data/task/dev/samples.jsonl`，300 条) 学经验分布——这是用户指定的"真实分布"基准。`dev_distribution.py` 在导入时离线计算并固化所有权重：

| 维度                     | 来源 |
|--------------------------|------|
| 函数权重（13 个全集）     | dev 各难度 `expression_numpy` 中函数出现频率 + Laplace +0.1 平滑 |
| 结构元素概率              | dev 各难度 `has_pow / has_abs / has_log / has_sqrt / has_exp_neg_xsq` 实测占比 |
| top-level term 分布       | dev 各难度顶层 `+ / -` 项数的经验 PMF |
| function_hints 长度       | dev 各难度 `function_hints` 字段长度的经验 PMF |
| function_hints 干扰项分布 | dev 各难度 `function_hints` 中各函数的出现率（含干扰项） |
| 常数权重                  | `{1,2,3}` 70% / `{0.5,1.5,…}` 20% / `{4,5,0}` 10`（仍是手工，不影响函数分布对齐） |

**Laplace +0.1 平滑**：dev 60 条里没出过的函数（如 `np.tan` 在 easy）仍以 ~0.16% 概率被采到。这是"未出现 ≠ 不能出"的应对：一组 60 样本不足以判定某函数概率严格为零。

**dev 标注约定的口径差异**：dev 的 `true_function_hints` 字段里**不包含 `np.abs`**（abs 被视为结构性包裹，不算"基函数"），所以 `extract_true_hints` 也排除 `np.abs`。这一点在 v2 已对齐。

## 数值安全（写在模板里）

避免重抽爆炸的关键：模板本身就保证表达式不会在 ref/test 窗口内炸。

- `np.exp(b*x)`：`|b| ≤ 1.5`
- `np.exp(-c*x**2)`：负号写死，`c > 0`，输出 ∈ (0,1]
- `np.log(...)`：包成 `np.log(np.abs(...) + 1)`
- `np.sqrt(...)`：包成 `np.sqrt(np.abs(...) + 1)`
- `**`：仅 2 或 3

## 用法

```bash
# 默认 1000 条，5 难度均匀 200×5
python in_domain/generate.py --n 1000 --seed 42 --out in_domain/output/ind_v1

# 自定义难度配比
python in_domain/generate.py \
    --counts easy=300,medium=300,hard=200,expert=100,extreme=100 \
    --seed 42 --out in_domain/output/ind_skewed

# 端到端 sanity（生成 10 条 + 校验 R² + 校验图片尺寸 + 校验下游解析）
python in_domain/smoke_test.py
```

CLI 参数：`--n / --counts / --seed / --out / --max-retries / --prompt-file`。`--prompt-file` 默认指向 `in_domain/prompt.txt`；这个文件是 prompt 的唯一可编辑源——同一份模板既被 `eval.py` 在评测时读取，也被 `generate.py` 在生成时用来填 `prompt` 字段。改 prompt 只需改这一处。

## 输出 schema（每行）

```json
{
  "id": 0,
  "split": "ind",
  "expression_str":  "0.5 * sin(2 * x)",
  "expression_numpy": "0.5 * np.sin(2 * x)",
  "true_function_hints": ["np.sin"],
  "function_hints":      ["np.sin", "np.cos", "np.tanh"],
  "data_points_text":    [[x1,y1], ...],
  "image_path":          "images/ind_easy_0000.png",
  "prompt":              "...rendered from prompt.txt; eval.py-compatible...",
  "test_points":         [[x1,y1], ... 50 个，x 完美均匀]
}
```

## 模块速查

| 文件                  | 关键 API                                                          |
|-----------------------|-------------------------------------------------------------------|
| `dev_distribution.py` | `expr_function_weights(diff)` / `true_hint_weights(diff)` / `prompt_hint_weights(diff)` / `structural_flags(diff)` / `sample_n_terms(diff,rng)` / `sample_hint_len(diff,rng)` / `pick_function(weights,rng,among=)` / `sample_distractors(diff,true_hints,rng)` / `DEV_STATS` |
| `expressions.py`      | `sample_expression(diff,rng)` / `make_function_hints(true,rng,diff)` / `extract_true_hints(expr)` / `sample_constant(rng)` |
| `sampling.py`         | `pick_ref_window` / `pick_test_window` / `make_ref_xs` / `make_test_xs` / `eval_expr` / `WINDOWS` 常量 |
| `rendering.py`        | `render_png(expr_numpy, x_min, x_max, out_path)`                  |
| `prompt_template.py`  | `load_template(path=None)` / `render_prompt(sample, template)` / `DEFAULT_PROMPT_PATH` |
| `prompt.txt`          | eval.py-compatible 模板，唯一可编辑的 prompt 源                    |
| `generate.py`         | `main(argv)` / `build_one(...)` / `resolve_counts(args)`          |
| `smoke_test.py`       | `run(out, n)` —— 10 条样本端到端验证                              |

## 与 dev 分布的偏差（v2，1000 条）

跑 `data_analysis/analyze.py --compare` 可见：

- **Overall 三视角函数频率**（`expr_freq` / `true_hint_freq` / `prompt_hint_freq`）：每个口径下 7 个主要函数的整体偏差均在 ±10pp 内，多数 ±5pp 内
- 表达式长度 / 深度 / term 数 / x 窗口偏差：overall 与 medium / easy 都在 ±5% 以内
- `eval_r2_mean = 1.0` —— 数据完整性 canary 通过

剩余可见偏差：
- `hard` 的 `np.exp` 比 dev 多 ~33pp（_hard 默认就大概率走 `exp(b*x) * f` 模板）
- `expert` / `extreme` 的 `np.sin` 比 dev 少 ~15-28pp（替换函数槽位时被其他 SAFE_FUNCS 分掉了一些权重）

如需进一步收紧，调 `expressions.py` 里 `_hard / _expert` 的分支概率即可，无需动 `dev_distribution.py`。
