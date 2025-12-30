def pairwise_f1(pred_clusters, gold_clusters):
    def cluster_pairs(clusters):
        pairs = set()
        for c in clusters:
            c = list(c)
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    pairs.add((c[i], c[j]))
        return pairs

    gold_pairs = set()
    N = len(gold_clusters)
    for i in range(N):
        for j in range(i + 1, N):
            if gold_clusters[i] is not None and gold_clusters[i] == gold_clusters[j]:
                gold_pairs.add((i, j))

    pred_pairs = cluster_pairs(pred_clusters)

    tp = len(pred_pairs & gold_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)

    return p, r, f1