from collections import defaultdict

def gold_eids_to_span_clusters(spans, gold_eids):
    clusters = defaultdict(list)

    for span, eid in zip(spans, gold_eids):
        if eid is None:
            continue
        clusters[eid].append(span)

    return list(clusters.values())

def clusters_from_ids_to_spans(pred_clusters, spans):
    span_clusters = []
    for cluster in pred_clusters:
        span_clusters.append(
            [spans[mid] for mid in cluster]
        )
    return span_clusters

def build_coref_instance(tokens, clusters):
    return {
        "tokens": tokens,
        "clusters": clusters
    }

def build_mention_to_cluster(clusters):

    m2c = {}
    for cid, cluster in enumerate(clusters):
        for m in cluster:
            m2c[tuple(m)] = cid
    return m2c