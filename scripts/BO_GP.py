# -*- coding: utf-8 -*-

"""
BO_GP.py

功能
----
贝叶斯优化主循环（支持分段运行，支持 Mordred / OHE 两种编码切换）。
每次运行只推荐“一批”实验：
    - 若 finished_reactions.csv 不存在或为空：随机推荐 BATCH_SIZE 个实验；
    - 若已有已完成实验：拟合带 Gamma 超参先验的 GP（MAP 估计），
      使用 EI + Kriging Believer 推荐下一批。

运行顺序（推荐）
----------------
    1) 先运行 DP_Mordred.py 或 DP_OHE.py，生成对应目录下的 X.csv 与 search_space.csv；
    2) 首次运行本脚本 → 随机推荐一批实验；
    3) 完成实验后把条件与产率写入对应 BO_DIR 下的 finished_reactions.csv；
    4) 再次运行本脚本 → 推荐下一批。

输入
----
    EDBO/data_DP_<编码>/X.csv
    EDBO/data_DP_<编码>/search_space.csv
    EDBO/data_BO_GP/finished_reactions.csv        可选

输出
----
    EDBO/data_BO_GP/suggested_reactions.csv
        本批推荐条件，含 entry 模板列与空白 yield 列。

说明
----
    - 通过 ENCODING 变量切换 'Mordred' 或 'OHE'，其余配置自动适配。
    - finished_reactions.csv 中的条件列应与 search_space.csv 保持一致，
      便于按列精确匹配。
    - entry 列仅用于线下追踪，不参与条件匹配。
    - EI 的探索系数 xi 支持两种模式：
        * USE_DYNAMIC_XI = True  ：随 BO 轮次线性衰减（前期探索，后期利用）
        * USE_DYNAMIC_XI = False ：使用静态 xi = XI_STATIC（文献推荐 δ=0.01）
    - GP 使用 ARD（每个特征一个 length scale）、Matérn 5/2 核与 WhiteKernel，
      并对 length scale / output scale / noise 三个超参数施加 Gamma 先验，
      通过最大化后验（MAP）估计超参数，对应文献中 GPyTorch 的做法。
"""

import os
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel
from scipy.stats import norm, gamma


# ==========================================================
# 0. 配置区
# ==========================================================

# ---- 编码切换：'Mordred' 或 'OHE' ----
ENCODING = "Mordred"

# 各编码对应的配置：
#   DP_DIR / BO_DIR        : 数据目录
#   PRIOR_LENGTH_SCALE     : length scale 的 Gamma 先验（a, scale=1/rate）
#   PRIOR_OUTPUT_SCALE     : output scale 的 Gamma 先验
#   PRIOR_NOISE            : noise 的 Gamma 先验
#   INIT_*                 : 超参数初始值（设为对应先验众数）
#
# 先验参数来源：Bayesian Reaction Optimization as a Tool for Chemical Synthesis. Nature 2021, 590 (7844), 89–96.
#   Mordred : 文献 Table S3
#   OHE     : 文献 Table S2
# scipy.stats.gamma: a = concentration, scale = 1/rate, 众数 = (a-1)*scale
CONFIG = {
    "Mordred": dict(
        DP_DIR = r"EDBO\data_DP_Mordred",
        BO_DIR = r"EDBO\data_BO_GP",
        PRIOR_LENGTH_SCALE = dict(a=2.0, scale=10.0),   # mode = 10.0
        PRIOR_OUTPUT_SCALE = dict(a=2.0, scale=10.0),   # mode = 10.0
        PRIOR_NOISE        = dict(a=1.5, scale=10.0),   # mode = 5.0
        INIT_LENGTH_SCALE  = 10.0,
        INIT_OUTPUT_SCALE  = 10.0,
        INIT_NOISE_LEVEL   = 5.0,
    ),
    "OHE": dict(
        DP_DIR = r"EDBO\data_DP_OHE",
        BO_DIR = r"EDBO\data_BO_GP",
        PRIOR_LENGTH_SCALE = dict(a=3.0, scale=1.0),    # mode = 2.0
        PRIOR_OUTPUT_SCALE = dict(a=5.0, scale=5.0),    # mode = 20.0
        PRIOR_NOISE        = dict(a=1.5, scale=10.0),   # mode = 5.0
        INIT_LENGTH_SCALE  = 2.0,
        INIT_OUTPUT_SCALE  = 20.0,
        INIT_NOISE_LEVEL   = 5.0,
    ),
}

if ENCODING not in CONFIG:
    raise ValueError(f"未知 ENCODING = '{ENCODING}'，应为 {list(CONFIG.keys())}。")

_cfg = CONFIG[ENCODING]

DP_DIR = _cfg["DP_DIR"]
BO_DIR = _cfg["BO_DIR"]

X_CSV            = os.path.join(DP_DIR, "X.csv")
SEARCH_SPACE_CSV = os.path.join(DP_DIR, "search_space.csv")

SUGGESTED_CSV = os.path.join(BO_DIR, "suggested_reactions.csv")
FINISHED_CSV  = os.path.join(BO_DIR, "finished_reactions_Mordred.csv")

# ---- BO 超参 ----
BATCH_SIZE  = 5        # 每批推荐实验数
N_RESTARTS  = 10       # GP 优化器重启次数
RANDOM_SEED = 42

# ---- EI 探索系数 xi 配置 ----
USE_DYNAMIC_XI = False   # True: 动态衰减；False: 静态 xi = XI_STATIC
XI_STATIC      = 0.01    # 静态探索系数（文献中 δ = 0.01）

# 动态衰减参数（仅当 USE_DYNAMIC_XI = True 时生效）
XI_MAX          = 0.15
XI_MIN          = 0.02
XI_DECAY_ROUNDS = 8

# ---- entry 列 ----
ENTRY_COL      = "entry"
ENTRY_TEMPLATE = "RC-RD*-*"

# ---- Gamma 先验配置（由 ENCODING 决定） ----
PRIOR_LENGTH_SCALE = _cfg["PRIOR_LENGTH_SCALE"]
PRIOR_OUTPUT_SCALE = _cfg["PRIOR_OUTPUT_SCALE"]
PRIOR_NOISE        = _cfg["PRIOR_NOISE"]

INIT_LENGTH_SCALE = _cfg["INIT_LENGTH_SCALE"]
INIT_OUTPUT_SCALE = _cfg["INIT_OUTPUT_SCALE"]
INIT_NOISE_LEVEL  = _cfg["INIT_NOISE_LEVEL"]

np.random.seed(RANDOM_SEED)


# ==========================================================
# 1. 数据加载与匹配
# ==========================================================

def _norm_key_value(v):
    """把条件值规范化为字符串，避免 0.1 / 0.10、105 / 105.0 等不一致。"""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    if pd.isna(v):
        return ""
    if isinstance(v, (int, float, np.integer, np.floating)):
        return f"{float(v):.6g}"
    return str(v).strip()


def load_search_space():
    """读取 X.csv 与 search_space.csv，返回 (X, search_space_df, condition_cols)。"""
    if not os.path.exists(X_CSV):
        raise FileNotFoundError(
            f"未找到 {X_CSV}，请先运行对应编码的特征生成脚本"
            f"（当前 ENCODING='{ENCODING}'）。"
        )
    if not os.path.exists(SEARCH_SPACE_CSV):
        raise FileNotFoundError(
            f"未找到 {SEARCH_SPACE_CSV}，请先运行对应编码的特征生成脚本。"
        )

    X = pd.read_csv(X_CSV).to_numpy(dtype=float)
    search_space_df = pd.read_csv(SEARCH_SPACE_CSV)

    if len(search_space_df) == 0:
        raise ValueError("search_space.csv 为空。")
    if X.shape[0] != len(search_space_df):
        raise ValueError(
            f"X.csv 行数 ({X.shape[0]}) 与 search_space.csv 行数 "
            f"({len(search_space_df)}) 不一致。"
        )

    condition_cols = [
        c for c in search_space_df.columns
        if not c.endswith("_smiles") and c.lower() != ENTRY_COL
    ]

    print(f"[加载] ENCODING = {ENCODING}")
    print(f"[加载] X 形状: {X.shape}")
    print(f"[加载] 搜索空间大小: {len(search_space_df)}")
    print(f"[加载] 条件列: {condition_cols}")

    return X, search_space_df, condition_cols


def load_finished_reactions():
    """
    读取 finished_reactions.csv。
    以下情况视为“尚无已完成实验”，返回 None：
        - 文件不存在 / 内容为空 / 无法解析
        - 缺少 yield 列
        - yield 列无法转为数值，或全部为空
    返回的 DataFrame 中 entry 列已被删除。
    """
    if not os.path.exists(FINISHED_CSV):
        return None

    try:
        df = pd.read_csv(FINISHED_CSV)
    except pd.errors.EmptyDataError:
        print(f"[BO] {FINISHED_CSV} 为空文件，按无已完成实验处理。")
        return None
    except pd.errors.ParserError as e:
        print(f"[BO] {FINISHED_CSV} 解析失败（{e}），按无已完成实验处理。")
        return None

    if ENTRY_COL in df.columns:
        df = df.drop(columns=[ENTRY_COL])

    if "yield" not in df.columns:
        print(f"[BO] {FINISHED_CSV} 缺少 'yield' 列，按无已完成实验处理。")
        return None

    df["yield"] = pd.to_numeric(df["yield"], errors="coerce")
    df = df.dropna(subset=["yield"])

    if df.empty:
        print(f"[BO] {FINISHED_CSV} 无有效 yield，按无已完成实验处理。")
        return None
    return df


def match_to_search_space(finished_df, search_space_df, condition_cols):
    """
    把已完成反应按条件列精确匹配回 search_space_df 的行号。
    对同一条件的重复实验，取产率均值聚合。

    Returns
    -------
    row_idx : np.ndarray(int)    search_space_df / X 中的行号
    y       : np.ndarray(float)  对应的产率
    """
    key_to_idx = {
        tuple(_norm_key_value(v) for v in row[condition_cols]): i
        for i, row in search_space_df.iterrows()
    }

    row_idx, y = [], []
    for _, row in finished_df.iterrows():
        key = tuple(_norm_key_value(v) for v in row[condition_cols])
        if key not in key_to_idx:
            print(f"[警告] 已完成反应未在搜索空间中找到，跳过: {key}")
            continue
        row_idx.append(key_to_idx[key])
        y.append(float(row["yield"]))

    if not row_idx:
        return np.asarray([], dtype=int), np.asarray([], dtype=float)

    row_idx = np.asarray(row_idx, dtype=int)
    y       = np.asarray(y, dtype=float)

    unique_idx = np.unique(row_idx)
    agg_y = np.array([y[row_idx == i].mean() for i in unique_idx], dtype=float)

    if len(unique_idx) < len(row_idx):
        print(f"[BO] 检测到重复条件，已聚合："
              f"{len(row_idx)} 条 → {len(unique_idx)} 条。")

    return unique_idx, agg_y


# ==========================================================
# 2. 带 Gamma 先验的 GP
# ==========================================================

class GammaPriorGPR(GaussianProcessRegressor):
    """
    在 sklearn 的 GaussianProcessRegressor 上加入 Gamma 超参先验。
    优化目标由最大似然变为最大后验（MAP）：
        L(theta) = log_marginal_likelihood(theta) + log_prior(theta)
    假定核结构为 C * Matern + WhiteKernel。
    """

    def __init__(self, kernel=None, *, alpha=1e-10, optimizer="fmin_l_bfgs_b",
                 n_restarts_optimizer=0, normalize_y=False,
                 copy_X_train=True, random_state=None,
                 length_scale_prior=None, output_scale_prior=None,
                 noise_prior=None):
        super().__init__(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=normalize_y,
            copy_X_train=copy_X_train,
            random_state=random_state,
        )
        self.length_scale_prior = length_scale_prior
        self.output_scale_prior = output_scale_prior
        self.noise_prior = noise_prior

    def _extract_hyperparameters(self, kernel):
        """从 C * Matern + WhiteKernel 核中提取实际超参数值。"""
        constant_value = float(kernel.k1.k1.constant_value)
        length_scale   = np.atleast_1d(kernel.k1.k2.length_scale).astype(float)
        noise_level    = float(kernel.k2.noise_level)
        return constant_value, length_scale, noise_level

    def _log_prior_and_grad(self, theta):
        """计算 Gamma 先验的对数密度及对无约束超参数 theta 的梯度。"""
        kernel = self.kernel_.clone_with_theta(theta)
        constant_value, length_scale, noise_level = self._extract_hyperparameters(kernel)

        prior = 0.0
        grad = np.zeros_like(theta, dtype=float)

        n_ls = len(length_scale)

        # ---- output scale 先验 ----
        if self.output_scale_prior is not None:
            if constant_value <= 0:
                return -np.inf, np.zeros_like(theta)
            a    = self.output_scale_prior.kwds["a"]
            rate = 1.0 / self.output_scale_prior.kwds["scale"]
            prior += self.output_scale_prior.logpdf(constant_value)
            grad[0] += (a - 1) - rate * constant_value

        # ---- length scale 先验 ----
        if self.length_scale_prior is not None:
            a    = self.length_scale_prior.kwds["a"]
            rate = 1.0 / self.length_scale_prior.kwds["scale"]
            for j, v in enumerate(length_scale):
                if v <= 0:
                    return -np.inf, np.zeros_like(theta)
                prior += self.length_scale_prior.logpdf(v)
                grad[1 + j] += (a - 1) - rate * v

        # ---- noise 先验 ----
        if self.noise_prior is not None:
            if noise_level <= 0:
                return -np.inf, np.zeros_like(theta)
            a    = self.noise_prior.kwds["a"]
            rate = 1.0 / self.noise_prior.kwds["scale"]
            prior += self.noise_prior.logpdf(noise_level)
            grad[1 + n_ls] += (a - 1) - rate * noise_level

        return prior, grad

    def log_marginal_likelihood(self, theta=None, eval_gradient=False, clone_kernel=True):
        if theta is None:
            theta = self.kernel_.theta

        result = super().log_marginal_likelihood(theta, eval_gradient, clone_kernel)

        if eval_gradient:
            lml, grad = result
        else:
            lml = result

        lprior, grad_prior = self._log_prior_and_grad(theta)

        if eval_gradient:
            return lml + lprior, grad + grad_prior
        return lml + lprior


# ==========================================================
# 3. 采集函数与批量推荐
# ==========================================================

def get_xi(n_done, batch_size=BATCH_SIZE,
           xi_max=XI_MAX, xi_min=XI_MIN,
           decay_rounds=XI_DECAY_ROUNDS,
           use_dynamic=USE_DYNAMIC_XI, xi_static=XI_STATIC):
    """根据已完成实验数推算当前 BO 轮次，并返回对应的 EI 探索系数 xi。"""
    if not use_dynamic:
        return None, xi_static

    bo_round = max(1, n_done // batch_size)
    frac     = min((bo_round - 1) / max(decay_rounds - 1, 1), 1.0)
    xi       = xi_max - (xi_max - xi_min) * frac
    return bo_round, xi


def expected_improvement(X_cand, gp, y_best, xi):
    """标准 EI 采集函数。"""
    mu, sigma = gp.predict(X_cand, return_std=True)
    mu    = mu.reshape(-1, 1)
    sigma = sigma.reshape(-1, 1)

    with np.errstate(divide="warn"):
        imp = mu - y_best - xi
        Z   = imp / sigma
        ei  = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei.flatten()


def build_gp(n_features):
    """构造带 Gamma 超参先验的高斯过程回归器。"""
    if n_features <= 0:
        raise ValueError(f"特征数必须为正，当前为 {n_features}。")

    kernel = (
        C(INIT_OUTPUT_SCALE, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(n_features) * INIT_LENGTH_SCALE,
            length_scale_bounds=(1e-3, 1e3),
            nu=2.5,
        )
        + WhiteKernel(
            noise_level=INIT_NOISE_LEVEL,
            noise_level_bounds=(1e-5, 1e2),
        )
    )

    length_scale_prior = gamma(**PRIOR_LENGTH_SCALE)
    output_scale_prior = gamma(**PRIOR_OUTPUT_SCALE)
    noise_prior        = gamma(**PRIOR_NOISE)

    return GammaPriorGPR(
        kernel=kernel,
        n_restarts_optimizer=N_RESTARTS,
        alpha=1e-10,
        normalize_y=True,
        random_state=RANDOM_SEED,
        length_scale_prior=length_scale_prior,
        output_scale_prior=output_scale_prior,
        noise_prior=noise_prior,
    )


def kriging_believer(X_pool, X_train, y_train, y_best, batch_size, xi):
    """
    Kriging Believer 批量推荐：每轮取 EI 最大点，并把 GP 预测值作为该点的
    “虚拟产率”加入训练集，然后继续下一轮。

    Returns
    -------
    selected_local : list[int]
        X_pool 中被选中的行号（局部索引）。
    """
    selected_local = []
    y_augmented    = []

    n_pool = X_pool.shape[0]
    n_select = min(batch_size, n_pool)
    if n_select <= 0:
        return []

    for _ in range(n_select):

        if selected_local:
            X_cur = np.vstack([X_train, X_pool[selected_local]])
            y_cur = np.hstack([y_train, y_augmented])
        else:
            X_cur, y_cur = X_train, y_train

        gp = build_gp(X_cur.shape[1])
        gp.fit(X_cur, y_cur)

        pool_mask = np.ones(n_pool, dtype=bool)
        if selected_local:
            pool_mask[selected_local] = False

        ei         = expected_improvement(X_pool[pool_mask], gp, y_best, xi)
        best_local = int(np.where(pool_mask)[0][np.argmax(ei)])

        selected_local.append(best_local)
        y_pred = gp.predict(X_pool[best_local].reshape(1, -1))[0]
        y_augmented.append(float(y_pred))

    return selected_local


# ==========================================================
# 4. 保存推荐并提示
# ==========================================================

def save_suggestions(indices, search_space_df, condition_cols):
    """把选中的 search_space 行写入 suggested_reactions.csv。"""
    if len(indices) == 0:
        print("[BO] 没有可推荐的实验。")
        return

    suggested_df = search_space_df.iloc[indices][condition_cols].copy()
    suggested_df.insert(0, ENTRY_COL, ENTRY_TEMPLATE)
    suggested_df["yield"] = ""

    suggested_df.to_csv(SUGGESTED_CSV, index=False, encoding="utf-8-sig")

    print(f"\n[BO] 已保存本批推荐实验到: {SUGGESTED_CSV}")
    print(suggested_df.to_string(index=False))

    print()
    print("=" * 68)
    print(f"当前编码: {ENCODING}")
    print("请执行以下步骤：")
    print(f"  1. 按 {SUGGESTED_CSV} 中的条件进行实验。")
    print(f"  2. 将实验条件与产率写入 {FINISHED_CSV}（追加即可）。")
    print(f"     若该文件尚不存在，可直接复制 {SUGGESTED_CSV}，")
    print(f"     填写 yield 列后另存为 finished_reactions.csv。")
    print(f"  3. entry 列已给出模板 '{ENTRY_TEMPLATE}'，")
    print( "     请自行把 * 号替换为实际编号（如 RC-RD1-1），便于线下追踪。")
    print( "  4. 再次运行本脚本，获得下一批推荐。")
    print("=" * 68)


# ==========================================================
# 5. 决策：随机推荐 vs 贝叶斯优化推荐
# ==========================================================

def random_suggest(n_total):
    """首次或无可用数据时，从搜索空间中随机选取一批实验。"""
    if n_total < BATCH_SIZE:
        raise ValueError(
            f"搜索空间大小 {n_total} 小于 BATCH_SIZE={BATCH_SIZE}，"
            f"无法随机推荐。请减小 BATCH_SIZE 或扩大搜索空间。"
        )
    return np.random.choice(n_total, BATCH_SIZE, replace=False)


def bo_suggest(X, search_space_df, finished_df, condition_cols):
    """
    使用 GP（带 Gamma 先验）+ Kriging Believer 生成推荐。
    若已完成实验无法匹配到搜索空间，则退化为随机推荐。
    """
    row_idx, y_train = match_to_search_space(
        finished_df, search_space_df, condition_cols
    )

    if len(row_idx) == 0:
        print("[警告] 已完成实验均未匹配到搜索空间，退化为随机推荐。")
        return random_suggest(len(search_space_df))

    X_train = X[row_idx]
    y_best  = float(np.max(y_train))
    print(f"[BO] 训练集大小: {len(row_idx)}, 当前最优产率: {y_best:.2f}")

    bo_round, xi = get_xi(len(row_idx))
    if USE_DYNAMIC_XI:
        print(f"[BO] 当前 BO 轮次: {bo_round}, "
              f"EI 探索系数 xi = {xi:.3f} "
              f"(衰减区间 {XI_MAX} → {XI_MIN}, {XI_DECAY_ROUNDS} 轮)")
    else:
        print(f"[BO] 静态 EI 探索系数 xi = {xi:.3f} (文献 δ)")

    done_set       = set(row_idx.tolist())
    candidate_mask = np.array([i not in done_set for i in range(len(X))])
    X_pool         = X[candidate_mask]
    pool_indices   = np.where(candidate_mask)[0]

    if len(X_pool) == 0:
        raise ValueError("候选池为空：所有搜索空间中的实验都已完成。")

    if len(X_pool) < BATCH_SIZE:
        print(f"[警告] 候选池仅剩 {len(X_pool)} 个，"
              f"少于 BATCH_SIZE={BATCH_SIZE}，将推荐全部剩余实验。")

    selected_local = kriging_believer(
        X_pool, X_train, y_train, y_best, BATCH_SIZE, xi
    )
    return pool_indices[selected_local]


def main():
    os.makedirs(BO_DIR, exist_ok=True)

    X, search_space_df, condition_cols = load_search_space()
    finished_df = load_finished_reactions()

    if finished_df is None:
        print(f"\n[BO] {FINISHED_CSV} 不存在或为空，"
              f"随机推荐 {BATCH_SIZE} 个实验。")
        indices = random_suggest(len(search_space_df))
    else:
        print(f"\n[BO] 读取到 {len(finished_df)} 条已完成实验，进入贝叶斯优化。")
        indices = bo_suggest(X, search_space_df, finished_df, condition_cols)

    save_suggestions(indices, search_space_df, condition_cols)


if __name__ == "__main__":
    main()