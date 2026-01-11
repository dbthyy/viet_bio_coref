import numpy as np
from scipy.optimize import linear_sum_assignment

def muc_score(pred_clusters, gold_clusters):
    def links(clusters):
        return sum(len(c) - 1 for c in clusters if len(c) > 1)

    def correct(pred, gold):
        score = 0
        for p in pred:
            partitions = sum(1 for g in gold if len(p & g) > 0)
            if len(p) > 0:
                score += len(p) - partitions
        return score

    recall = correct(gold_clusters, pred_clusters) / max(links(gold_clusters), 1e-8)
    precision = correct(pred_clusters, gold_clusters) / max(links(pred_clusters), 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def b_cubed_score(pred_clusters, gold_clusters, N):
    gold_map = {}
    pred_map = {}

    for c in gold_clusters:
        for m in c:
            gold_map[m] = c

    for c in pred_clusters:
        for m in c:
            pred_map[m] = c

    p_sum = r_sum = 0.0
    for i in range(N):
        if i not in gold_map or i not in pred_map:
            continue

        inter = len(gold_map[i] & pred_map[i])
        p_sum += inter / len(pred_map[i])
        r_sum += inter / len(gold_map[i])

    precision = p_sum / N
    recall = r_sum / N
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

def ceaf_phi4(pred_clusters, gold_clusters):
    def phi4(c1, c2):
        return 2 * len(c1 & c2) / (len(c1) + len(c2) + 1e-8)

    sim = np.zeros((len(gold_clusters), len(pred_clusters)))
    for i, g in enumerate(gold_clusters):
        for j, p in enumerate(pred_clusters):
            sim[i, j] = phi4(g, p)

    row, col = linear_sum_assignment(-sim)
    score = sim[row, col].sum()

    precision = score / max(len(pred_clusters), 1e-8)
    recall = score / max(len(gold_clusters), 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1