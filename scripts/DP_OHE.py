# -*- coding: utf-8 -*-

"""
DP_OHE.py

功能
----
把反应空间 JSON 转换成贝叶斯优化所需的 One-Hot 特征矩阵 X.csv。
流程：构建搜索空间 → 生成各维度 One-Hot 映射 → 拼接特征矩阵 → 去常数/去相关/归一化

运行顺序（推荐）
----------------
    1) 运行本脚本：生成 <dim>_ohe.csv、search_space.csv 与 X.csv
    2) 运行 BO_GP.py：按贝叶斯优化流程推荐实验
    3) 可选：运行 TSNE_vis.py / UMAP_vis.py 可视化探索进度

输入
----
    EDBO/data_DP_OHE/reaction_space.json
        反应空间定义。每个维度可以是：
            - dict 形式：{试剂名: SMILES}
            - list 形式：["试剂A", "试剂B"]，或 [{"name": ..., "smiles": ...}, ...]

输出
----
    EDBO/data_DP_OHE/<dim>_ohe.csv
        各维度的 One-Hot 编码表，第一列为 <dim>_SMILES，
        其余列为 <dim>=<SMILES>，每类一列，取值 0/1。
    EDBO/data_DP_OHE/search_space.csv
        所有候选实验条件组合（含各维度的 *_smiles 列）。
    EDBO/data_DP_OHE/X.csv
        最终特征矩阵，供 BO_GP.py / TSNE_vis.py / UMAP_vis.py 读取。

说明
----
    - USE_CONTINUOUS=False 时，温度、浓度等连续维度不参与搜索空间与特征。
    - 搜索空间大小 = 各维度取值数量的笛卡尔积；维度或取值多时注意内存与耗时。
    - SAVE_OHE_CSV=True 时，额外输出各维度的 One-Hot 编码表。
"""

import os
import json
from itertools import product

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# 0. 配置区
# ==========================================================

JSON_PATH = r"EDBO\data_DP_OHE\reaction_space.json"
DATA_DIR  = r"EDBO\data_DP_OHE"

SEARCH_SPACE_CSV = os.path.join(DATA_DIR, "search_space.csv")
X_CSV            = os.path.join(DATA_DIR, "X.csv")

# 连续维度开关：为 True 时启用下面的温度 / 浓度列表
USE_CONTINUOUS         = False
TEMPERATURES: list     = []
CONCENTRATIONS: list   = []

ALL_CONTINUOUS_NAMES  = ["temperature", "concentration"]
ALL_CONTINUOUS_RANGES = [TEMPERATURES, CONCENTRATIONS]

# 特征预处理参数
VAR_EPS        = 1e-12   # 方差阈值：低于该值视为常数特征
CORR_THRESHOLD = 0.95    # 相关性阈值：超过该值的列会被剔除

# 是否为每个维度输出 <dim>_ohe.csv
SAVE_OHE_CSV = True


# ==========================================================
# 1. 工具函数
# ==========================================================

def is_missing(x):
    """判断值是否为 None / NaN / 空字符串。"""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def resolve_continuous_config(use_continuous):
    """根据开关返回 (生效的连续维度名, 生效的连续取值列表)。"""
    if not use_continuous:
        return [], []
    return list(ALL_CONTINUOUS_NAMES), [list(v) for v in ALL_CONTINUOUS_RANGES]


def parse_reaction_space(reaction_space):
    """
    解析 reaction_space JSON，返回：
        dimensions     : {维度名: [取值1, 取值2, ...]}
        reagent_smiles : {维度名: {取值: SMILES}}
    """
    if not isinstance(reaction_space, dict) or not reaction_space:
        raise ValueError("reaction_space.json 为空或格式不正确。")

    dimensions, reagent_smiles = {}, {}

    for dim_name, reagents in reaction_space.items():

        if isinstance(reagents, dict):
            if not reagents:
                raise ValueError(f"维度 '{dim_name}' 为空字典。")
            dimensions[dim_name]     = list(reagents.keys())
            reagent_smiles[dim_name] = dict(reagents)

        elif isinstance(reagents, list):
            if not reagents:
                raise ValueError(f"维度 '{dim_name}' 为空列表。")
            names, smi_map = [], {}
            for item in reagents:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("reagent")
                    smi  = item.get("smiles", "")
                else:
                    name, smi = item, ""
                if name is None:
                    raise ValueError(
                        f"维度 '{dim_name}' 中存在缺少 name/reagent 的条目。"
                    )
                names.append(name)
                smi_map[name] = smi
            dimensions[dim_name]     = names
            reagent_smiles[dim_name] = smi_map

        else:
            raise ValueError(
                f"无法解析维度 '{dim_name}' 的结构: {type(reagents)}"
            )

    return dimensions, reagent_smiles


def build_search_space_table(dimensions, continuous_names, continuous_values):
    """按各维度取值做笛卡尔积，生成候选条件表 df。"""
    reagent_dims = list(dimensions.keys())
    value_lists  = [dimensions[d] for d in reagent_dims] + continuous_values
    column_names = reagent_dims + continuous_names

    rows = list(product(*value_lists))
    df   = pd.DataFrame(rows, columns=column_names)
    return df, reagent_dims


def build_onehot_encoding(dimensions, reagent_smiles):
    """
    为每个维度构建 One-Hot 编码映射。

    Returns
    -------
    ohe_index  : {dim: {smiles: 列索引}}
    ohe_smiles : {dim: [smiles, ...]}   按原始顺序排列的唯一 SMILES 列表
    """
    ohe_index, ohe_smiles = {}, {}

    for dim, names in dimensions.items():
        smiles_list, seen = [], set()
        for name in names:
            smi = reagent_smiles[dim].get(name, "")
            if is_missing(smi):
                raise ValueError(
                    f"维度 '{dim}' 的取值 '{name}' 缺少 SMILES。"
                )
            if smi not in seen:
                seen.add(smi)
                smiles_list.append(smi)

        if not smiles_list:
            raise ValueError(f"维度 '{dim}' 没有可用的 SMILES。")

        ohe_smiles[dim] = smiles_list
        ohe_index[dim]  = {smi: i for i, smi in enumerate(smiles_list)}

    return ohe_index, ohe_smiles


def save_ohe_csvs(ohe_smiles, data_dir):
    """
    为每个维度保存 <dim>_ohe.csv。

    格式（与文献 One-Hot 示例一致）：
        第一列：<dim>_SMILES
        其余列：<dim>=<SMILES>，每个类别一列，取值 0/1
    """
    for dim, smiles_list in ohe_smiles.items():
        n = len(smiles_list)

        mat = np.zeros((n, n), dtype=int)
        np.fill_diagonal(mat, 1)

        cols = [f"{dim}={smi}" for smi in smiles_list]
        df   = pd.DataFrame(mat, columns=cols)
        df.insert(0, f"{dim}_SMILES", smiles_list)

        path = os.path.join(data_dir, f"{dim}_ohe.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[OHE] 已保存: {path} ({n} 个类别)")


def build_feature_matrix(df, reagent_dims, continuous_names, ohe_index, ohe_smiles):
    """
    对每行实验拼接特征：
        按 reagent_dims 顺序拼接各试剂维度的 One-Hot 向量
        + 连续维度数值（若 continuous_names 非空）
    """
    smiles_cols = [f"{d}_smiles" for d in reagent_dims]

    rows = []
    for _, row in df.iterrows():
        parts = []
        for dim, col in zip(reagent_dims, smiles_cols):
            smi = row[col]
            if is_missing(smi):
                raise ValueError(
                    f"第 {row.name} 行缺少 SMILES（列 '{col}'）。"
                )
            if smi not in ohe_index[dim]:
                raise ValueError(
                    f"第 {row.name} 行的 '{dim}' SMILES 未出现在 OHE 映射中: {smi}"
                )

            vec = np.zeros(len(ohe_smiles[dim]), dtype=float)
            vec[ohe_index[dim][smi]] = 1.0
            parts.append(vec)

        if continuous_names:
            parts.append(np.asarray(
                [row[c] for c in continuous_names], dtype=float
            ))

        rows.append(np.concatenate(parts))

    if not rows:
        raise ValueError("搜索空间为空，无法构建特征矩阵。")

    return np.asarray(rows, dtype=float)


def preprocess_features(X_raw, var_eps=VAR_EPS, corr_threshold=CORR_THRESHOLD):
    """
    特征预处理：
        1) 去除方差为 0 的常数特征；
        2) 贪心去相关，保留与已选特征相关性均不超过 corr_threshold 的列；
        3) 用 MinMaxScaler 归一化到 [0, 1]。
    """
    if X_raw.size == 0:
        raise ValueError("特征矩阵为空，无法预处理。")

    # 1) 去常数特征
    variances  = np.var(X_raw, axis=0)
    var_mask   = variances > var_eps
    X_nonconst = X_raw[:, var_mask]

    if X_nonconst.shape[1] == 0:
        raise ValueError("所有特征方差均为 0，无法进行预处理。")

    # 2) 贪心去相关
    if X_nonconst.shape[1] == 1:
        keep = [0]
    else:
        corr = np.corrcoef(X_nonconst, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)
        keep = []
        for i in range(corr.shape[0]):
            if all(abs(corr[i, j]) <= corr_threshold for j in keep):
                keep.append(i)

    X_selected = X_nonconst[:, keep]

    # 3) 归一化到 [0, 1]
    X = MinMaxScaler().fit_transform(X_selected)

    return X, X_nonconst.shape[1], X_selected.shape[1]


# ==========================================================
# 2. 主流程
# ==========================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---------- 第 1 段：构建搜索空间 ----------
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        reaction_space = json.load(f)

    dimensions, reagent_smiles = parse_reaction_space(reaction_space)
    continuous_names, continuous_values = resolve_continuous_config(USE_CONTINUOUS)

    df, reagent_dims = build_search_space_table(
        dimensions, continuous_names, continuous_values
    )

    if len(df) == 0:
        raise ValueError("搜索空间为空，请检查 reaction_space.json。")

    # 为每个试剂维度补充对应的 SMILES 列
    for d in reagent_dims:
        df[f"{d}_smiles"] = df[d].map(reagent_smiles[d])

    print(f"[第1段] 连续维度开关 USE_CONTINUOUS = {USE_CONTINUOUS}")
    print(f"[第1段] 生效的连续维度: {continuous_names}")
    print(f"[第1段] 总实验数: {len(df)}")
    print(df.head())

    df.to_csv(SEARCH_SPACE_CSV, index=False, encoding="utf-8-sig")
    print(f"[第1段] 已保存: {SEARCH_SPACE_CSV}")

    # ---------- 第 2 段：构建 One-Hot 编码 ----------
    ohe_index, ohe_smiles = build_onehot_encoding(dimensions, reagent_smiles)

    print(f"[第2段] 各维度 One-Hot 维度数:")
    for dim in reagent_dims:
        print(f"        {dim}: {len(ohe_smiles[dim])}")

    if SAVE_OHE_CSV:
        save_ohe_csvs(ohe_smiles, DATA_DIR)

    # ---------- 第 3 段：拼接原始特征矩阵 ----------
    X_raw = build_feature_matrix(
        df, reagent_dims, continuous_names, ohe_index, ohe_smiles
    )

    print(f"[第3段] 连续维度开关 USE_CONTINUOUS = {USE_CONTINUOUS}")
    print(f"[第3段] 原始特征矩阵形状: {X_raw.shape}")

    # ---------- 第 4 段：特征预处理 ----------
    X, n_nonconst, n_selected = preprocess_features(X_raw)

    print(f"[第4段] 去除常数特征后特征数: {n_nonconst}")
    print(f"[第4段] 去相关后特征数: {n_selected}")
    print(f"[第4段] 归一化后形状: {X.shape}")

    pd.DataFrame(X).to_csv(X_CSV, index=False, encoding="utf-8-sig")
    print(f"[第4段] 已保存: {X_CSV}")


if __name__ == "__main__":
    main()