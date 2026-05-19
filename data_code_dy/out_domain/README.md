# out_of_domain (占位)

OOD 数据生成尚未实现，待 OOD 定义敲定后开工。

候选方向：

- **窗口外推**：在比 ref 窗口更宽的 x 范围采样 test_points，考察模型外推能力。
- **新函数组合**：训练时未出现的函数簇（如 `np.sinh / np.cosh / np.arctan` 主导）。
- **加噪观测**：在 `data_points_text` 上加可控噪声。
- **跨域形态**：分段函数、阶跃、不连续点。

实现时复用 `in_domain/expressions.py`、`sampling.py`、`rendering.py` 的模块化设计。
