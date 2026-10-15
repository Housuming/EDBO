# EDBO 贝叶斯优化实验推荐

作者：Housuming & Deepseek v4.1 flash

版本：1.0 (2026.10.15)

基于反应空间编码（Mordred 分子描述符 / One-Hot）与高斯过程（GP）的反应条件
贝叶斯优化工具。脚本支持分段运行：每轮推荐一批实验，完成实验后填入产率，
再次运行即可获得下一批推荐。

- 支持两种反应编码：Mordred 描述符与 One-Hot。
- 通过 `BO_GP.py` 中 `ENCODING` 变量一键切换，Gamma 超参先验自动匹配。
- 支持并行批量推荐（EI + Kriging Believer），适合人工实验或 HTE 平台。

## 一、环境安装

推荐 Python 3.9 。

```bash
pip install -r requirements.txt
```

注意：

- `numpy` 不要升到 2.x，`mordred==1.2.0` 可能不兼容。
- UMAP 的导入名是 `umap`，但 pip 包名是 `umap-learn`，不要装成 `umap`。
- 若仅使用 One-Hot 编码（只跑 `DP_OHE.py`），可跳过 `rdkit` 与 `mordred`。

## 二、文件与目录

```text
EDBO/
├── data_DP_Mordred/
│   ├── reaction_space.json         # 输入：反应空间定义
│   ├── <dim>_mordred.csv           # 输出：各维度 Mordred 编码表
│   ├── search_space.csv            # 输出：候选实验条件表
│   └── X.csv                       # 输出：最终特征矩阵
├── data_DP_OHE/
│   ├── reaction_space.json         # 输入：反应空间定义
│   ├── <dim>_ohe.csv               # 输出：各维度 One-Hot 编码表
│   ├── search_space.csv
│   └── X.csv
├── data_BO_GP/
│   ├── suggested_reactions.csv     # 输出：本批推荐实验
│   └── finished_reactions.csv      # 输入：已完成实验与产率，可选
├── data_TSNE/                      # t-SNE 可视化输出
└── data_UMAP/                      # UMAP 可视化输出
```

主要脚本：

| 脚本 | 用途 |
|---|---|
| `DP_Mordred.py` | 构建搜索空间，计算 Mordred 描述符，生成 `X.csv` 与 `<dim>_mordred.csv` |
| `DP_OHE.py`     | 构建搜索空间，生成 One-Hot 编码，生成 `X.csv` 与 `<dim>_ohe.csv` |
| `BO_GP.py`      | 贝叶斯优化主循环，分批推荐实验 |
| `TSNE_vis.py`   | t-SNE 可视化搜索空间与已完成实验 |
| `UMAP_vis.py`   | UMAP 可视化搜索空间与已完成实验 |

## 三、快速开始

以下以 Mordred 编码为例。使用 One-Hot 编码时把 `DP_Mordred.py` 换成
`DP_OHE.py`，并把 `BO_GP.py` 中的 `ENCODING` 改为 `"OHE"`。

1. 编辑反应空间定义：

   `EDBO/data_DP_Mordred/reaction_space.json`，例如：

   ```json
   {
     "additive":     { "additive-0": "SMILES...", "additive-1": "SMILES..." },
     "ligand":       { "ligand-0":   "SMILES...", "ligand-1":   "SMILES..." },
     "base":         { "base-0":     "SMILES...", "base-1":     "SMILES..." },
     "aryl_halide":  { "aryl_halide-0": "SMILES...", "aryl_halide-1": "SMILES..." }
   }
   ```

   每个维度可以是 `{名称: SMILES}` 字典，也可以是 `["名称1", "名称2"]`
   或 `[{"name": ..., "smiles": ...}, ...]` 列表。

2. 生成搜索空间与特征矩阵：

   ```bash
   python DP_Mordred.py
   ```

   生成：

   - `EDBO/data_DP_Mordred/search_space.csv`
   - `EDBO/data_DP_Mordred/X.csv`
   - `EDBO/data_DP_Mordred/<dim>_mordred.csv`（每个维度一份编码表）

3. 首次运行贝叶斯优化：

   ```bash
   python BO_GP.py
   ```

   若 `finished_reactions.csv` 不存在或为空，会随机推荐一批实验，
   写入 `EDBO/data_BO_GP/suggested_reactions.csv`。

4. 按推荐条件完成实验，把条件与产率写入
   `EDBO/data_BO_GP/finished_reactions.csv`：

   - 可直接复制 `suggested_reactions.csv`，填写 `yield` 后另存为
     `finished_reactions.csv`。
   - `entry` 列仅为线下追踪，可把模板 `RC-RD*-*` 的 `*` 替换为实际编号。
   - `yield` 建议填 0\~1（可视化色标默认按 0\~1）。

5. 再次运行：

   ```bash
   python BO_GP.py
   ```

   已有已完成实验后，脚本会拟合 GP，使用 EI + Kriging Believer
   推荐下一批。重复步骤 4\~5。

6. 可选可视化：

   ```bash
   python TSNE_vis.py
   python UMAP_vis.py
   ```

   输出图片与坐标文件到 `EDBO/data_TSNE/` 或 `EDBO/data_UMAP/`。

## 四、编码切换

`BO_GP.py` 通过 `ENCODING` 变量切换编码，Gamma 先验与初始超参自动匹配：

```python
ENCODING = "Mordred"   # 或 "OHE"
```

先验参数来源：Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590 (7844), 89–96.

| 参数 | Mordred（文献 Table S3） | One-Hot（文献 Table S2） |
|---|---|---|
| length scale 先验 | Gamma(a=2.0, rate=0.1)，众数 10 | Gamma(a=3.0, rate=1.0)，众数 2 |
| output scale 先验 | Gamma(a=2.0, rate=0.1)，众数 10 | Gamma(a=5.0, rate=0.2)，众数 20 |
| noise 先验        | Gamma(a=1.5, rate=0.1)，众数 5  | Gamma(a=1.5, rate=0.1)，众数 5 |

切换 `ENCODING` 后需保证对应目录下的 `X.csv` 与 `search_space.csv` 已生成。

## 五、输入输出说明

### `DP_Mordred.py` / `DP_OHE.py`

输入：

- `EDBO/data_DP_Mordred/reaction_space.json`（或 `data_DP_OHE` 下同名文件）

输出：

- `EDBO/data_DP_<编码>/search_space.csv`
- `EDBO/data_DP_<编码>/X.csv`
- `EDBO/data_DP_<编码>/<dim>_mordred.csv` 或 `<dim>_ohe.csv`

流程：

- 按反应空间各维度做笛卡尔积，生成候选实验条件。
- 对唯一 SMILES 计算 Mordred 描述符（或生成 One-Hot 映射）。
- 拼接特征矩阵。
- 去常数特征、去相关（Pearson > 0.95）、MinMax 归一化到 [0, 1]。

配置项：

- `USE_CONTINUOUS`：是否启用温度、浓度等连续维度。
- `CORR_THRESHOLD`：去相关阈值，默认 `0.95`。
- `SAVE_MORDRED_CSV` / `SAVE_OHE_CSV`：是否输出各维度编码表。

注意：搜索空间为笛卡尔积，维度或取值过多时注意内存与耗时。

### `BO_GP.py`

输入：

- `EDBO/data_DP_<编码>/X.csv`
- `EDBO/data_DP_<编码>/search_space.csv`
- `EDBO/data_BO_GP/finished_reactions.csv`，可选

输出：

- `EDBO/data_BO_GP/suggested_reactions.csv`

逻辑：

- 无已完成实验：随机推荐 `BATCH_SIZE` 个实验。
- 有已完成实验：按条件列精确匹配到搜索空间，拟合带 Gamma 先验的
  GP（MAP 估计），使用 EI + Kriging Believer 批量推荐。
- EI 探索系数 `xi` 支持静态（默认 `XI_STATIC = 0.01`）与动态衰减两种模式。

主要配置：

- `ENCODING`：`"Mordred"` 或 `"OHE"`。
- `BATCH_SIZE`：每批推荐数量，默认 5。
- `USE_DYNAMIC_XI`、`XI_STATIC`、`XI_MAX`、`XI_MIN`、`XI_DECAY_ROUNDS`：
  EI 探索系数。
- `N_RESTARTS`、`RANDOM_SEED`：GP 优化器与随机种子。

### `TSNE_vis.py` / `UMAP_vis.py`

输入：

- `EDBO/data_DP_Mordred/X.csv`
- `EDBO/data_DP_Mordred/search_space.csv`
- `EDBO/data_BO_GP/finished_reactions.csv`，可选

输出：

- `tsne_search_space.png` / `tsne_coords.csv`
- `umap_search_space.png` / `umap_coords.csv`

用途：观察搜索空间覆盖情况与已完成实验的产率分布。

若需可视化 One-Hot 编码下的搜索空间，把脚本中 `DP_DIR` 改为
`EDBO\data_DP_OHE` 即可。

## 六、数据格式要求

- `search_space.csv`：包含条件列与对应的 `*_smiles` 列。
- `finished_reactions.csv`：
  - 必须包含与 `search_space.csv` 一致的条件列，以及 `yield` 列。
  - `yield` 建议为 0\~1 的数值；空字符串与 NaN 会被自动过滤。
  - 同一条件的重复实验会按均值聚合。
  - `entry` 列不参与匹配，可保留用于线下追踪。
- 条件匹配会规范化数值列（如 `105` 与 `105.0`、`0.1` 与 `0.10` 视为相同）。

## 七、注意事项

- `X.csv` 行数必须与 `search_space.csv` 行数一致，否则无法对齐。
- 修改 `reaction_space.json` 后，需要重新运行 `DP_Mordred.py` 或 `DP_OHE.py`。
- 若已完成实验条件无法匹配到搜索空间，`BO_GP.py` 会跳过该条记录；
  若全部无法匹配，会退化为随机推荐。
- 候选池剩余数量少于 `BATCH_SIZE` 时，`BO_GP.py` 会自动推荐全部剩余实验。
- 默认路径为 Windows 风格，跨平台运行时请调整脚本中的路径配置。
- 当前 `BO_GP.py` 的 Mordred 与 OHE 两种配置共用 `EDBO\data_BO_GP\`；
  若两套编码并行优化，建议在 `CONFIG` 里为它们分别指定独立的 `BO_DIR`。
- 本 README 以当前脚本内容为准。
