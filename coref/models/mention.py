import torch

def get_mention_embeddings(hidden, word_ids, spans, gold_clusters):
    embs, kept_clusters, kept_spans = [], [], []

    for (s, e), cluster_id in zip(spans, gold_clusters):
        idx = [i for i, wid in enumerate(word_ids) if wid is not None and s <= wid < e]
        if not idx:
            continue

        emb = hidden[idx].mean(dim=0)
        embs.append(emb)
        kept_clusters.append(cluster_id)
        kept_spans.append((s, e))  

    if not embs:
        return None, None, None

    return torch.stack(embs), kept_clusters, kept_spans

def get_test_mention_embeddings(hidden, word_ids, spans):
    embs = []
    kept_spans = []

    for s, e in spans:
        idx = [i for i, wid in enumerate(word_ids) if wid is not None and s <= wid < e]
        if not idx:
            continue

        embs.append(hidden[idx].mean(dim=0))
        kept_spans.append((s, e))

    if not embs:
        return None, None

    return torch.stack(embs), kept_spans