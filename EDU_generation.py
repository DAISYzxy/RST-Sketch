from Preprocessing_Four_Priors import *
from Pareto_Frontiers_Generation import *
import json
from typing import Any
import os
from tqdm import tqdm

# ------------------------
# 读取新的输入文件（单个 JSON，外层是 list）
# ------------------------
input_path = "wcep_rst.json"
with open(input_path, "r", encoding="utf-8") as f:
    try:
        data_list = json.load(f)  # <- 外层 list
        assert isinstance(data_list, list)
        print(f"成功读取: {input_path}，样本数：{len(data_list)}")
    except json.JSONDecodeError as e:
        raise RuntimeError(f"解析失败: {input_path}, 错误: {e}")

# ------------------------
# 处理并汇总保存为一个 json 文件
# ------------------------
results_all = []  # 每个元素保存一个样本的处理结果（append 到一个 list）

llm_relation_docuType_weights = {
    "Attribution": 0.08,
    "Enablement": 0.06,
    "Cause": 0.12,
    "Temporal": 0.11,
    "Condition": 0.04,
    "Elaboration": 0.05,
    "Background": 0.07,
    "Topic-Comment": 0.05,
    "Evaluation": 0.02,
    "Explanation": 0.09,
    "TextualOrganization": 0.01,
    "Contrast": 0.04,
    "Comparison": 0.03,
    "Summary": 0.10,
    "Manner-Means": 0.07,
    "Same-Unit": 0.01,
    "Joint": 0.03,
    "Topic-Change": 0.02
}

beta_n = 1.5
beta_s = 0.5
L_max = 768

# 预设 epsilon
EPSILON = 0.5

def to_zero_based(indices, n_edus: int):
    """
    把 select_indices 之类返回的 indices（可能是 "1","2"... 或 int，可能 0/1-based）
    统一转换成 0-based，用于访问 edus[idx]
    """
    inds = list(indices)
    if not inds:
        return []

    # 字符串 key: "1","2"... -> 0-based
    if all(isinstance(x, str) for x in inds):
        return sorted(int(x) - 1 for x in inds)

    inds_int = [int(x) for x in inds]

    # 推断是 0-based
    if min(inds_int) == 0 or max(inds_int) == n_edus - 1:
        return sorted(inds_int)

    # 推断是 1-based
    if min(inds_int) == 1 and max(inds_int) == n_edus:
        return sorted(i - 1 for i in inds_int)

    # 兜底：优先尝试 1-based -> 0-based（更常见）
    cand = [i - 1 for i in inds_int]
    if all(0 <= i < n_edus for i in cand):
        return sorted(cand)

    # 最后兜底：按 0-based，过滤越界
    return sorted([i for i in inds_int if 0 <= i < n_edus])

for item_idx, item in tqdm(enumerate(data_list)):
    # 新结构：每个 item 是一个 dict，包含 'id', 'input', 'segmentation', 'tree_parsing'
    try:
        toks = item["input"]
        brks = item["segmentation"]
        steps = item["tree_parsing"][0]
    except (KeyError, IndexError, TypeError) as e:
        print(f"[跳过] 第 {item_idx} 个样本字段缺失或格式不对：{e}")
        continue

    # ===== 与原流程一致 =====
    edus = edus_from(toks, brks)
    parsed = parse_rst_pairs(steps)

    EDU_roles = collect_roles(parsed)
    if not EDU_roles:
        print(f"[跳过] 第 {item_idx} 个样本 EDU_roles 为空")
        continue

    nucleus_proportion = sum(1 for v in EDU_roles.values() if v == "Nucleus") / len(EDU_roles)
    satellite_proportion = sum(1 for v in EDU_roles.values() if v == "Satellite") / len(EDU_roles)

    denom = (nucleus_proportion * beta_n + satellite_proportion * beta_s)
    if denom == 0:
        denom = 1e-8
    b_n = beta_n / denom
    b_s = beta_s / denom

    weight_acc = accumulate_weights(parsed, llm_relation_docuType_weights, list(EDU_roles.keys()))
    EDU_layers = count_EDU_layer(parsed, n_edus=len(edus))
    h_indicator = mark_nucleus_backbone(set(EDU_roles.keys()), parsed)
    dn_priors = calculate_depth_nucleus_prior(h_indicator, EDU_layers)

    scores = calculate_edu_information_density(edus, alpha_tf=0.2, kappa=2)
    info_density_scores = {str(i + 1): v for i, v in enumerate(scores)}

    normalized_relation_priors = normalize_to_mean_one(weight_acc)
    normalized_dn_priors = normalize_to_mean_one(dn_priors)
    normalized_info_density_priors = normalize_to_mean_one(info_density_scores)

    agg_edu_priors = {}
    for key in EDU_roles.keys():
        role = EDU_roles[key]
        role_prior = b_n if role == 'Nucleus' else b_s
        key_str = str(key)
        agg_edu_priors[key_str] = (
            role_prior
            * normalized_relation_priors[key_str]
            * normalized_dn_priors[key_str]
            * normalized_info_density_priors[key_str]
        )

    lengths = compute_lengths(edus)

    # ------------------------
    # 单目标：使 J1 最大的基准解（用于 J1_star，也用于 fallback 最终输出）
    # ------------------------
    results = select_indices(agg_edu_priors, lengths, L_max)

    costs_total = 0
    J1_star = 0.0
    for idx in results:
        costs_total += lengths[idx]
        J1_star += agg_edu_priors[idx]

    # ------------------------
    # 多目标：在固定 epsilon 下尝试生成解；无解则直接回退到“J1 最大”的单目标解
    # ------------------------
    sim_mat, sim_df, embeddings = semantic_similarity_matrix(edus)

    J1_J2_results = generate_multiple_solutions_greedy(
        sim_mat, lengths, agg_edu_priors, L_max,
        w_min=J1_star * EPSILON,
        k=100, base_seed=0, jitter_std=1e-3
    )

    # ===== Fallback：若预设 epsilon 下没解，放弃 J2，直接用 J1 最大的解作为最终结果 =====
    if len(J1_J2_results) == 0:
        chosen_indices = to_zero_based(results, len(edus))
        retrieved_edus = " ".join(edus[i] for i in chosen_indices).strip()

        results_all.append({
            "id": item.get("id", item_idx),
            "selected_text": retrieved_edus,
            "selected_indices": chosen_indices,  # 0-based
            "total_cost": costs_total,
            "total_weight": J1_star
        })
        continue

    # ------------------------
    # 正常情况：有多目标候选解，再走 Pareto + J2
    # ------------------------
    J1_K = [res["total_weight"] for res in J1_J2_results]
    J2_K = [res["total_cost"] for res in J1_J2_results]
    frontier = get_pareto(J1_K, J2_K, method='fast', mode1='max', mode2='min', sort_by_index=True)

    sel_keys, J2_star, cost = select_min_similar_set(
        sim_mat, lengths, L_max, method='greedy', pulp_time_limit=10
    )
    nearest = find_nearest(frontier, J1_star, J2_star)

    chosen_indices = J1_J2_results[nearest[0]]["indices"]
    retrieved_edus = " ".join(edus[idx] for idx in chosen_indices).strip()

    results_all.append({
        "id": item.get("id", item_idx),
        "selected_text": retrieved_edus,
        "selected_indices": chosen_indices,
        "total_cost": J1_J2_results[nearest[0]]["total_cost"],
        "total_weight": J1_J2_results[nearest[0]]["total_weight"]
    })

# ------------------------
# 一次性写出一个 JSON 文件（list）
# ------------------------
save_json_path = "wcep_selected_edus_0dot5.json"
with open(save_json_path, "w", encoding="utf-8") as f:
    json.dump(results_all, f, ensure_ascii=False, indent=2)

print("Epsilon: 0.5; \n")

print(f"已保存到：{save_json_path}，共 {len(results_all)} 条。")
