# -*- coding: utf-8 -*-

"""
DP_Mordred.py

功能
----
把反应空间 JSON 转换成贝叶斯优化所需的数值特征矩阵 X.csv。
流程：构建搜索空间 → 计算 Mordred 描述符 → 拼接特征矩阵 → 去常数/去相关/归一化

运行顺序（推荐）
----------------
    1) 运行本脚本：生成 <dim>_mordred.csv、search_space.csv 与 X.csv
    2) 运行 BO_GP.py：按贝叶斯优化流程推荐实验
    3) 可选：运行 TSNE_vis.py / UMAP_vis.py 可视化探索进度

输入
----
    EDBO/data_DP_Mordred/reaction_space.json
        反应空间定义。每个维度可以是：
            - dict 形式：{试剂名: SMILES}
            - list 形式：["试剂A", "试剂B"]，或 [{"name": ..., "smiles": ...}, ...]

输出
----
    EDBO/data_DP_Mordred/<dim>_mordred.csv
        各维度的 Mordred 编码表，第一列为 <dim>_SMILES，
        其余列为 Mordred 描述符名，每行一个唯一 SMILES。
    EDBO/data_DP_Mordred/search_space.csv
        所有候选实验条件组合（含各维度的 *_smiles 列）。
    EDBO/data_DP_Mordred/X.csv
        最终特征矩阵，供 BO_GP.py / TSNE_vis.py / UMAP_vis.py 读取。

说明
----
    - USE_CONTINUOUS=False 时，温度、浓度等连续维度不参与搜索空间与特征。
    - 搜索空间大小 = 各维度取值数量的笛卡尔积；维度或取值多时注意内存与耗时。
    - SAVE_MORDRED_CSV=True 时，额外输出各维度的 Mordred 编码表。
"""

import os
import json
from itertools import product

import numpy as np
import pandas as pd

from rdkit import Chem
from mordred import Calculator, descriptors
from sklearn.preprocessing import MinMaxScaler


# ==========================================================
# 0. 配置区
# ==========================================================

JSON_PATH = r"EDBO\data_DP_Mordred\reaction_space.json"
DATA_DIR  = r"EDBO\data_DP_Mordred"

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

# 是否为每个维度输出 <dim>_mordred.csv
SAVE_MORDRED_CSV = True


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


def compute_unique_descriptors(smiles_set, calc):
    """
    对每个唯一 SMILES 计算一次 Mordred 描述符，返回 {SMILES: 描述符向量}。
    非数值 / NaN / Inf 统一置为 0.0。
    """
    def _descriptor_vector(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"无效 SMILES: {smiles}")

        raw = calc(mol)
        vals = []
        for d in raw:
            try:
                v = float(d)
                if np.isnan(v) or np.isinf(v):
                    v = 0.0
            except (TypeError, ValueError):
                v = 0.0
            vals.append(v)
        return np.asarray(vals, dtype=np.float64)

    return {smi: _descriptor_vector(smi) for smi in smiles_set}


def get_descriptor_names(calc):
    """返回与 calc(mol) 输出顺序一致的描述符名列表。"""
    return [str(d) for d in calc.descriptors]


def unique_smiles_in_dim(smi_map):
    """按出现顺序返回某维度中唯一的、非缺失的 SMILES 列表。"""
    seen, smiles_list = set(), []
    for smi in smi_map.values():
        if is_missing(smi):
            continue
        if smi not in seen:
            seen.add(smi)
            smiles_list.append(smi)
    return smiles_list


def save_mordred_csvs(reagent_smiles, smiles_to_desc, desc_names, data_dir):
    """
    为每个维度保存 <dim>_mordred.csv。

    格式（与文献 One-Hot 示例风格一致）：
        第一列：<dim>_SMILES
        其余列：Mordred 描述符名
        每一行：一个唯一 SMILES 及其描述符向量
    """
    for dim, smi_map in reagent_smiles.items():
        smiles_list = unique_smiles_in_dim(smi_map)
        if not smiles_list:
            print(f"[Mordred] 维度 '{dim}' 无有效 SMILES，跳过。")
            continue

        rows = [smiles_to_desc[smi] for smi in smiles_list]
        df   = pd.DataFrame(rows, columns=desc_names)
        df.insert(0, f"{dim}_SMILES", smiles_list)

        path = os.path.join(data_dir, f"{dim}_mordred.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"[Mordred] 已保存: {path} "
              f"({len(smiles_list)} 个分子 × {len(desc_names)} 个描述符)")


def build_feature_matrix(df, reagent_dims, continuous_names, smiles_to_desc):
    """
    对每行实验拼接特征：
        按 reagent_dims 顺序拼接各试剂维度的 Mordred 描述符
        + 连续维度数值（若 continuous_names 非空）
    """
    smiles_cols = [f"{d}_smiles" for d in reagent_dims]

    rows = []
    for _, row in df.iterrows():
        parts = []
        for col in smiles_cols:
            smi = row[col]
            if is_missing(smi):
                raise ValueError(
                    f"第 {row.name} 行缺少 SMILES（列 '{col}'）。"
                )
            if smi not in smiles_to_desc:
                raise ValueError(
                    f"第 {row.name} 行的 SMILES 未计算描述符: {smi}"
                )
            parts.append(smiles_to_desc[smi])

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

    # ---------- 第 2 段：计算 Mordred 描述符 ----------
    calc       = Calculator(descriptors, ignore_3D=True)
    desc_names = get_descriptor_names(calc)

    unique_smiles = {
        smi
        for smi_map in reagent_smiles.values()
        for smi in smi_map.values()
        if not is_missing(smi)
    }

    if not unique_smiles:
        raise ValueError("没有可计算的唯一 SMILES，请检查反应空间定义。")

    smiles_to_desc = compute_unique_descriptors(unique_smiles, calc)

    print(f"[第2段] 唯一 SMILES 数量: {len(unique_smiles)}")
    print(f"[第2段] 每个分子的描述符数量: {len(desc_names)}")

    # ---------- 第 2.5 段：为每个维度输出 <dim>_mordred.csv ----------
    if SAVE_MORDRED_CSV:
        save_mordred_csvs(reagent_smiles, smiles_to_desc, desc_names, DATA_DIR)

    # ---------- 第 3 段：拼接原始特征矩阵 ----------
    X_raw = build_feature_matrix(df, reagent_dims, continuous_names, smiles_to_desc)

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