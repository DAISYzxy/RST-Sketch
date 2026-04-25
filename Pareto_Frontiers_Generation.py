from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import numpy as np
from itertools import combinations
import math

def find_nearest(frontier, J1_star, J2_star):
    # frontier 中每个元素是 (id, J1, J2)
    best_item = None
    best_dist = float("inf")
    for item in frontier:
        _, J1, J2 = item
        dist = math.hypot(J1 - J1_star, J2 - J2_star)  # 欧式距离
        if dist < best_dist:
            best_dist = dist
            best_item = item
    return best_item


def _keys_to_ordered_list(lengths: Dict[str, float]) -> List[str]:
    """
    将 lengths 的 keys（字符串）按 int(key) 排序并返回列表。
    假定 keys 可以转换为 int，并且 '1' 对应索引 0。
    """
    keys = list(lengths.keys())
    # sort by integer value of key
    keys_sorted = sorted(keys, key=lambda k: int(k))
    return keys_sorted

def compute_pairwise_sum(sim_mat: np.ndarray, idxs: List[int]) -> float:
    """计算选中索引集合 idxs 的 pairwise similarity 之和（只计算 i<j）。"""
    s = 0.0
    for i, j in combinations(idxs, 2):
        s += sim_mat[i, j]
    return s

def select_min_similar_set(
    sim_mat: np.ndarray,
    lengths: Dict[str, float],
    L_max: float,
    method: str = 'mip',
    pulp_time_limit: float = None
) -> Tuple[List[str], float, float]:
    
    keys_sorted = _keys_to_ordered_list(lengths)
    n = len(keys_sorted)
    # 建立 index 映射：第 k 个 key 对应 sim_mat 的行索引 int(key)-1
    indices = [int(k)-1 for k in keys_sorted]
    # 检查 sim_mat 维度
    if sim_mat.shape[0] < max(indices)+1 or sim_mat.shape[1] < max(indices)+1:
        raise ValueError("sim_mat 的尺寸不足以覆盖 lengths 中的最大索引（检查 keys 与 sim_mat 的对应关系）。")

    costs = [lengths[k] for k in keys_sorted]

    if method == 'bruteforce':
        # 仅在 n 较小时使用
        if n > 25:
            raise ValueError("bruteforce 方法只适用于 n <= ~25，否则组合爆炸。")
        best_obj = float('inf')
        best_set = []
        best_cost = 0.0
        # 枚举所有子集（空集也允许）
        for r in range(n+1):
            for comb in combinations(range(n), r):
                # 计算真实的 sim_mat 索引集合
                idxs = [indices[i] for i in comb]
                total_cost = sum(costs[i] for i in comb)
                if total_cost <= L_max:
                    obj = compute_pairwise_sum(sim_mat, idxs)
                    if obj < best_obj:
                        best_obj = obj
                        best_set = list(comb)
                        best_cost = total_cost
        selected_keys = [keys_sorted[i] for i in best_set]
        return selected_keys, best_obj, best_cost

    if method == 'greedy':
        # 贪心：从空集开始，反复尝试加入在满足成本约束下使目标增量最小的 item
        selected = set()
        selected_sim_indices = []  # 存真实 sim_mat 的索引
        remaining = set(range(n))
        current_cost = 0.0
        current_obj = 0.0  # pairwise sum so far
        improved = True
        # 如果 sim 可以为负值，则有可能不断加入降低目标（好的）直到成本耗尽
        while improved:
            improved = False
            best_inc = None
            best_i = None
            best_new_cost = None
            for i in list(remaining):
                c = costs[i]
                if current_cost + c > L_max:
                    continue
                # 增量 = sum_j sim(idx_i, idx_j) over j in selected
                idx_i = indices[i]
                inc = 0.0
                for idx_j in selected_sim_indices:
                    inc += sim_mat[idx_i, idx_j]
                # 因为 objective = sum_{i<j} sim_ij, 当加入 i 时只需加 inc
                # 选择使 inc 最小的（可能为负）
                if best_inc is None or inc < best_inc:
                    best_inc = inc
                    best_i = i
                    best_new_cost = current_cost + c
            if best_i is not None:
                # 如果加入会改善目标（inc < 0）或虽然不改善但我们想填满（允许 inc >=0 但仍可能加入）
                # 这里策略：如果 inc < 0 -> 一定加入；如果没有 inc<0 的项，仍然尝试加入最小 inc 的项（可选）
                # 我们选择：加入只要 inc <= 0 或当前 selected 为空且 inc is smallest (尝试开启)
                if best_inc <= 0 or len(selected) == 0:
                    selected.add(best_i)
                    selected_sim_indices.append(indices[best_i])
                    current_cost = best_new_cost
                    current_obj += best_inc
                    remaining.remove(best_i)
                    improved = True
                else:
                    # 如果没有负增量，尝试看看是否任何正增量也可以被接受？我们停止以保守策略。
                    break
        selected_keys = [keys_sorted[i] for i in sorted(selected)]
        return selected_keys, current_obj, current_cost

    raise ValueError("未知 method, 支持 'greedy'|'bruteforce'。")



def semantic_similarity_matrix(edus, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    if not isinstance(edus, (list, tuple)) or not all(isinstance(x, str) for x in edus):
        raise ValueError("`edus` 应为字符串列表。")

    # 关闭所有可能的 notebook 小组件/进度条
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        edus,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False   # 避免 ipywidgets 依赖
    )

    sim_mat = cosine_similarity(embeddings)  # (n x n)

    def _shorten(s, maxlen=30):
        return s if len(s) <= maxlen else s[:maxlen - 1] + "…"
    index = [f"{i}: {_shorten(s)}" for i, s in enumerate(edus)]
    sim_df = pd.DataFrame(sim_mat, index=index, columns=index)
    return sim_mat, sim_df, embeddings


def select_indices(agg_edu_priors, lengths, L_max):
    indices = list(agg_edu_priors.keys())
    n = len(indices)

    # dp[i][l] 表示考虑前 i 个元素，长度限制为 l 时的最大价值
    dp = [[0] * (L_max + 1) for _ in range(n + 1)]

    # 回溯路径用
    keep = [[False] * (L_max + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        idx = indices[i - 1]
        val = agg_edu_priors[idx]
        cost = lengths[idx]
        for l in range(L_max + 1):
            # 不选
            dp[i][l] = dp[i - 1][l]
            # 选
            if cost <= l and dp[i - 1][l - cost] + val > dp[i][l]:
                dp[i][l] = dp[i - 1][l - cost] + val
                keep[i][l] = True

    # 回溯选中的索引
    chosen = []
    l = L_max
    for i in range(n, 0, -1):
        if keep[i][l]:
            idx = indices[i - 1]
            chosen.append(idx)
            l -= lengths[idx]

    chosen.reverse()
    return chosen


def select_subset_greedy(
    sem_matrix: np.ndarray,
    cost_dict: Dict,
    weight_dict: Dict,
    Lmax: float,
    *,
    w_min: float,
    seed: int = 42,   # 为了与原接口兼容，虽然纯贪心用不到随机
) -> dict:
    """
    纯贪心（无局部搜索）：
        minimize  sum_{i<j, i,j in S} s_ij
        s.t.      sum c_i <= Lmax,  sum w_i >= w_min
    """
    if w_min < 0:
        raise ValueError("w_min 必须 >= 0")

    n = sem_matrix.shape[0]
    assert sem_matrix.shape == (n, n), "sem_matrix 必须为方阵"
    assert np.allclose(sem_matrix, sem_matrix.T, atol=1e-9), "sem_matrix 必须对称"

    # ---- 规范化输入键 ----
    def normalize_dict(d: Dict, name: str) -> Dict[int, float]:
        keys = list(d.keys())
        if all(isinstance(k, int) for k in keys):
            if set(d.keys()) != set(range(n)):
                raise ValueError(f"{name} 的整数键必须覆盖 0..{n-1}")
            return {int(k): float(d[k]) for k in range(n)}
        elif all(isinstance(k, str) and k.isdigit() for k in keys):
            if set(d.keys()) != set(str(i+1) for i in range(n)):
                raise ValueError(f"{name} 的字符串键必须覆盖 '1'..'{n}'")
            return {i: float(d[str(i+1)]) for i in range(n)}
        else:
            raise ValueError(f"{name} 键必须是 0..{n-1} 或 '1'..'{n}'")

    cost_dict = normalize_dict(cost_dict, "cost_dict")
    weight_dict = normalize_dict(weight_dict, "weight_dict")
    c = np.array([cost_dict[i] for i in range(n)], dtype=float)
    w = np.array([weight_dict[i] for i in range(n)], dtype=float)

    # ---- 状态量 ----
    selected = np.zeros(n, dtype=bool)
    remain = np.ones(n, dtype=bool)
    current_cost = 0.0
    current_weight = 0.0

    # g[i] = 选择 i 时对子集内两两相似度之和的“边际增量”
    # 对称矩阵下，若当前选择集的指示向量为 x，则边际增量 = (S @ x)[i]
    g = np.zeros(n, dtype=float)

    # ---- 辅助：在 remain∧可行 的候选里，按 (增量, -权重, 成本) 取最小 ----
    def pick_next(budget_only: bool) -> int | None:
        """
        budget_only=False 时：用于攒权重阶段
            目标：优先小增量，其次权重大，再次成本低（更有机会达到 w_min）
        budget_only=True 时：可选（达到 w_min 后如仍想在预算内继续改善/维持）
        """
        feasible = remain & (current_cost + c <= Lmax + 1e-12)
        if not np.any(feasible):
            return None
        idx = np.where(feasible)[0]

        # 打分元组：(增量, -权重, 成本)
        scores = np.stack([g[idx], -w[idx], c[idx]], axis=1)
        # 逐列字典序最小
        order = np.lexsort((scores[:, 2], scores[:, 1], scores[:, 0]))
        return int(idx[order[0]])

    # ---- 阶段1：先把权重攒到 w_min ----
    while current_weight < w_min - 1e-12:
        i = pick_next(budget_only=False)
        if i is None:
            # 在预算内无法把权重攒到 w_min
            raise ValueError("在给定 Lmax 下无法达到 w_min（问题不可行）。")
        # 选择 i
        selected[i] = True
        remain[i] = False
        current_cost += c[i]
        current_weight += w[i]
        # 更新边际增量向量
        g += sem_matrix[i, :]

    # ---- 计算目标（只用上三角）----
    x = selected.astype(int)
    obj = _pairwise_sum_fast(sem_matrix, x)

    return {
        "indices": np.where(selected)[0].tolist(),
        "objective": float(obj),
        "total_cost": float(current_cost),
        "total_weight": float(current_weight),
        "W_target": float(w_min),
        "method": "greedy",
    }


def _pairwise_sum_fast(S: np.ndarray, x: np.ndarray) -> float:
    # 与你现有的 _pairwise_sum 等价，只是写法不同
    # 目标 = 0.5 * (x^T S x - sum_i S_ii x_i)
    quad = float(x @ (S @ x))
    diag = float(np.sum(np.diag(S) * x))
    return 0.5 * (quad - diag)


def generate_multiple_solutions_greedy(
    sem_matrix,
    cost_dict,
    weight_dict,
    Lmax,
    w_min,
    *,
    k: int = 5,
    base_seed: int = 0,
    jitter_std: float = 0.0,
    max_tries: int | None = None,
    one_based_indices: bool = False,
):
    if max_tries is None:
        max_tries = max(k * 3, k + 2)

    rng = np.random.default_rng(base_seed)
    n = sem_matrix.shape[0]

    solutions = []
    seen = set()

    for t in range(max_tries):
        S = sem_matrix
        if jitter_std > 0.0:
            noise = rng.normal(0.0, jitter_std, size=S.shape)
            noise = (noise + noise.T) / 2.0
            np.fill_diagonal(noise, 0.0)
            S = S + noise

        try:
            res = select_subset_greedy(
                S, cost_dict, weight_dict, Lmax,
                w_min=w_min,
                seed=base_seed + t,
            )
        except ValueError:
            # 当前抖动下不可行，跳过
            continue

        key_0based = tuple(sorted(res["indices"]))
        if key_0based in seen:
            continue
        seen.add(key_0based)

        if one_based_indices:
            res = dict(res)
            res["indices"] = [str(i + 1) for i in res["indices"]]

        solutions.append(res)
        if len(solutions) >= k:
            break

    solutions.sort(key=lambda r: r["objective"])
    return solutions


def _compare(a, b, mode):
    return a >= b if mode == 'max' else a <= b

def _strict_better(a, b, mode):
    return a > b if mode == 'max' else a < b

def pareto_frontier_bruteforce(J1: List[float], J2: List[float],
                              mode1: str='max', mode2: str='min') -> List[int]:
    """O(n^2) 暴力法：返回非支配点的原始索引（保持输入顺序）"""
    assert len(J1) == len(J2), "J1 and J2 must have same length"
    n = len(J1)
    non_dominated = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            at_least_as_good = _compare(J1[j], J1[i], mode1) and _compare(J2[j], J2[i], mode2)
            strictly_better = _strict_better(J1[j], J1[i], mode1) or _strict_better(J2[j], J2[i], mode2)
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            non_dominated.append(i)
    return non_dominated

def pareto_frontier_2d_fast(J1: List[float], J2: List[float],
                            mode1: str='max', mode2: str='min') -> List[int]:
    """
    O(n log n) 的二维快速方法：
    - 依据第一维（J1）排序（max -> 降序; min -> 升序）
    - 扫描时维护已见到的最优第二维（J2），选出非支配点
    注意：若存在完全相同的最优点，fast 方法可能只保留一个代表。
    返回的是非支配点的索引（检测顺序），如需按原始顺序可 later sort。
    """
    assert len(J1) == len(J2), "J1 and J2 must have same length"
    n = len(J1)
    if n == 0:
        return []
    pts = [(J1[i], J2[i], i) for i in range(n)]
    # 主维排序方向
    reverse_primary = True if mode1 == 'max' else False
    pts.sort(key=lambda x: (x[0], x[1]), reverse=reverse_primary)
    frontier = []
    if mode2 == 'max':
        best_secondary = float('-inf')
        for a, b, idx in pts:
            if b > best_secondary:
                frontier.append(idx)
                best_secondary = b
    else:  # mode2 == 'min'
        best_secondary = float('inf')
        for a, b, idx in pts:
            if b < best_secondary:
                frontier.append(idx)
                best_secondary = b
    return frontier

# 辅助：返回 (index, J1, J2) 并可选择按原始索引排序
def get_pareto(J1: List[float], J2: List[float],
               method: str='fast', mode1: str='max', mode2: str='min',
               sort_by_index: bool = True) -> List[Tuple[int, float, float]]:
    if method == 'fast':
        indices = pareto_frontier_2d_fast(J1, J2, mode1=mode1, mode2=mode2)
    else:
        indices = pareto_frontier_bruteforce(J1, J2, mode1=mode1, mode2=mode2)
    if sort_by_index:
        indices = sorted(indices)
    return [(i, J1[i], J2[i]) for i in indices]


