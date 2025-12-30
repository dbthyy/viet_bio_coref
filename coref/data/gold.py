def gold_clusters_from_ids(gold_clusters):
    clusters = {}
    for i, cid in enumerate(gold_clusters):
        if cid is None:
            continue
        if cid not in clusters:
            clusters[cid] = set()
        clusters[cid].add(i)
    return list(clusters.values())