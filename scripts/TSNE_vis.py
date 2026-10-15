# -*- coding: utf-8 -*-

"""
TSNE_vis.py

功能
----
用 t-SNE 把搜索空间特征矩阵 X.csv 投影到二维，并按产率给已完成实验着色，
用于观察贝叶斯优化的探索进度与覆盖情况。

运行顺序（推荐）
----------------
    1) 先运行 DP_Mordred.py 或 DP_OHE.py 生成 X.csv 与 search_space.csv；
    2) 运行 BO_GP.py 并完成实验，维护 finished_reactions.csv；
    3) 运行本脚本生成可视化。

输入
----
    EDBO/data_DP_Mordred/X.csv
    EDBO/data_DP_Mordred/search_space.csv
    EDBO/data_BO_GP/finished_reactions.csv        可选；
        不存在、为空或无有效产率时，仅绘制搜索空间。

输出
----
    EDBO/data_TSNE/tsne_search_space.png
    EDBO/data_TSNE/tsne_coords.csv
"""

import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # 无 GUI 环境下也可出图
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE


# ==========================================================
# 0. 配置区
# ==========================================================

DP_DIR   = r"EDBO\data_DP_OHE"
BO_DIR   = r"EDBO\data_BO_GP"
TSNE_DIR = r"EDBO\data_TSNE"

X_CSV            = os.path.join(DP_DIR, "X.csv")
SEARCH_SPACE_CSV = os.path.join(DP_DIR, "search_space.csv")
FINISHED_CSV     = os.path.join(BO_DIR, "finished_reactions_OHE.csv")

OUT_PNG    = os.path.join(TSNE_DIR, "tsne_search_space_OHE.png")
OUT_COORDS = os.path.join(TSNE_DIR, "tsne_coords_OHE.csv")

ENTRY_COL = "entry"

# t-SNE 超参
# 说明：perplexity 一般取 5~50 且必须小于样本数；下面会自动收缩到安全范围。
#       样本量很大（> 数千）时，可考虑先 PCA 预降维以加速。
TSNE_PERPLEXITY    = 30
TSNE_LEARNING_RATE = "auto"
TSNE_MAX_ITER      = 1000        # 旧名 n_iter，sklearn >= 1.5 起改名 max_iter
TSNE_INIT          = "pca"
TSNE_METRIC        = "euclidean"
RANDOM_SEED        = 42

# 绘图超参
FIG_SIZE      = (10, 8)
DPI           = 200
POINT_SIZE_BG = 18
POINT_SIZE_FG = 70

YIELD_MIN = 0.0
YIELD_MAX = 1.0

# 产率色标：蓝（低）→ 中灰 → 红（高）
CMAP_BLUE_RED = LinearSegmentedColormap.from_list(
    "blue_to_red",
    [(0.0, "#1f4fff"), (0.5, "#c8c8c8"), (1.0, "#e02020")],
)

# 图像中统一使用英文，避免字体缺失导致乱码
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================
# 1. 数据加载
# ==========================================================

def load_finished_reactions():
    """
    读取 finished_reactions.csv。
    以下情况视为“尚无已完成实验”，返回 None：
        - 文件不存在 / 内容为空 / 无法解析
        - 缺少 yield 列
        - 所有行 yield 均为空
    """
    if not os.path.exists(FINISHED_CSV):
        print(f"[Vis] 未找到 {FINISHED_CSV}")
        return None

    try:
        df = pd.read_csv(FINISHED_CSV)
    except pd.errors.EmptyDataError:
        print(f"[Vis] {FINISHED_CSV} 为空文件，按无已完成实验处理。")
        return None
    except pd.errors.ParserError as e:
        print(f"[Vis] {FINISHED_CSV} 解析失败（{e}），按无已完成实验处理。")
        return None

    if ENTRY_COL in df.columns:
        df = df.drop(columns=[ENTRY_COL])

    if "yield" not in df.columns:
        print(f"[Vis] {FINISHED_CSV} 缺少 'yield' 列。")
        return None

    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna(subset=["yield"])
    return None if df.empty else df


def match_to_search_space(finished_df, search_space_df, condition_cols):
    """
    把已完成实验按条件列精确匹配回 search_space_df 的行号。
    返回 (row_idx, y)，其中 row_idx 是 search_space_df / X 中的行号。
    """
    key_to_idx = {
        tuple(str(v) for v in row[condition_cols]): i
        for i, row in search_space_df.iterrows()
    }

    row_idx, y = [], []
    for _, row in finished_df.iterrows():
        key = tuple(str(v) for v in row[condition_cols])
        if key not in key_to_idx:
            print(f"[Warn] 已完成反应未在搜索空间中找到，跳过: {key}")
            continue
        row_idx.append(key_to_idx[key])
        y.append(float(row["yield"]))

    return np.asarray(row_idx, dtype=int), np.asarray(y, dtype=float)


# ==========================================================
# 2. t-SNE 降维
# ==========================================================

def compute_tsne_embedding(X):
    n_samples = X.shape[0]

    # perplexity 必须 < n_samples，自动收缩并留出安全裕量
    perplexity = min(TSNE_PERPLEXITY, max(5, (n_samples - 1) // 3))
    if perplexity >= n_samples:
        perplexity = max(1.0, float(n_samples - 1))
    perplexity = float(perplexity)

    if n_samples < 4:
        print(f"[t-SNE][Warn] n_samples={n_samples} 过小，t-SNE 结果可能不稳定。")

    # init="pca" 需要 n_samples > n_components，否则退回 random
    init = TSNE_INIT
    if init == "pca" and n_samples <= 2:
        init = "random"

    print(f"\n[t-SNE] Start: n_samples={n_samples}, perplexity={perplexity}, "
          f"learning_rate={TSNE_LEARNING_RATE}, max_iter={TSNE_MAX_ITER}, "
          f"init={init}, metric={TSNE_METRIC}")

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=TSNE_LEARNING_RATE,
        max_iter=TSNE_MAX_ITER,     # sklearn >= 1.5 使用 max_iter
        init=init,
        metric=TSNE_METRIC,
        random_state=RANDOM_SEED,
    )
    embedding = tsne.fit_transform(X)

    print(f"[t-SNE] Done, embedding shape: {embedding.shape}, "
          f"KL divergence: {tsne.kl_divergence_:.4f}")
    return embedding


# ==========================================================
# 3. 绘图
# ==========================================================

def plot_embedding(coords_df, has_finished, out_png):
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    bg = coords_df[~coords_df["done"]]
    fg = coords_df[coords_df["done"]]

    # 背景：整个搜索空间
    ax.scatter(
        bg["tsne_1"], bg["tsne_2"],
        s=POINT_SIZE_BG, c="lightgray", alpha=0.6,
        edgecolors="none", label=f"Unexplored (n={len(bg)})", zorder=1,
    )

    if has_finished:
        sc = ax.scatter(
            fg["tsne_1"], fg["tsne_2"],
            s=POINT_SIZE_FG, c=fg["yield"],
            cmap=CMAP_BLUE_RED, vmin=YIELD_MIN, vmax=YIELD_MAX,
            edgecolors="black", linewidths=0.6,
            label=f"Finished (n={len(fg)})", zorder=3,
        )

        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Yield")
        cbar.set_ticks(np.arange(YIELD_MIN, YIELD_MAX + 1e-9, 0.1))
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylim(YIELD_MIN, YIELD_MAX)

        # 标注当前最优产率点
        best_row = coords_df.loc[fg["yield"].idxmax()]
        ax.annotate(
            f"best = {best_row['yield']:.2f}",
            xy=(best_row["tsne_1"], best_row["tsne_2"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, color="black",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            zorder=4,
        )
    else:
        # 没有已完成实验时也画一个 0~1 的占位 colorbar，观感统一
        sm = plt.cm.ScalarMappable(
            cmap=CMAP_BLUE_RED,
            norm=plt.Normalize(vmin=YIELD_MIN, vmax=YIELD_MAX),
        )
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Yield")
        cbar.set_ticks(np.arange(YIELD_MIN, YIELD_MAX + 1e-9, 0.1))
        cbar.ax.tick_params(labelsize=9)

        ax.scatter(
            [], [], s=POINT_SIZE_FG, c="tab:red",
            edgecolors="black", linewidths=0.6,
            label="Finished (n=0)",
        )

    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")
    ax.set_title("t-SNE Projection of Search Space and Finished Reactions")
    ax.legend(loc="best", frameon=True)
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# ==========================================================
# 4. 主流程
# ==========================================================

def main():
    os.makedirs(TSNE_DIR, exist_ok=True)

    # ---- 加载搜索空间与特征矩阵 ----
    X = pd.read_csv(X_CSV).to_numpy(dtype=float)
    search_space_df = pd.read_csv(SEARCH_SPACE_CSV)

    condition_cols = [
        c for c in search_space_df.columns
        if not c.endswith("_smiles") and c.lower() != ENTRY_COL
    ]

    print(f"[Load] X shape: {X.shape}")
    print(f"[Load] search_space size: {len(search_space_df)}")
    print(f"[Load] condition columns: {condition_cols}")

    if X.shape[0] != len(search_space_df):
        raise ValueError(
            f"X.csv 行数 ({X.shape[0]}) 与 search_space.csv 行数 "
            f"({len(search_space_df)}) 不一致，无法对齐。"
        )

    # ---- t-SNE 降维 ----
    embedding = compute_tsne_embedding(X)

    # ---- 组装绘图数据 ----
    coords_df = pd.DataFrame({
        "tsne_1": embedding[:, 0],
        "tsne_2": embedding[:, 1],
        "done":   False,
        "yield":  np.nan,
    })

    finished_df = load_finished_reactions()
    has_finished = False

    if finished_df is not None:
        row_idx, y_done = match_to_search_space(
            finished_df, search_space_df, condition_cols
        )
        if len(row_idx) > 0:
            coords_df.loc[row_idx, "done"]  = True
            coords_df.loc[row_idx, "yield"] = y_done
            has_finished = True
            print(f"[Vis] 匹配已完成反应 {len(row_idx)} 条，"
                  f"产率范围 {y_done.min():.2f} ~ {y_done.max():.2f}")
        else:
            print("[Vis] 无已完成反应匹配到搜索空间。")
    else:
        print("[Vis] 无已完成反应，仅绘制搜索空间。")

    # 坐标文件：附带条件列，便于线下核查
    coords_out = pd.concat(
        [search_space_df[condition_cols].reset_index(drop=True), coords_df],
        axis=1,
    )
    coords_out.to_csv(OUT_COORDS, index=False, encoding="utf-8-sig")
    print(f"[Vis] 坐标已保存: {OUT_COORDS}")

    # ---- 绘图 ----
    plot_embedding(coords_df, has_finished, OUT_PNG)
    print(f"[Vis] 图像已保存: {OUT_PNG}")


if __name__ == "__main__":
    main()