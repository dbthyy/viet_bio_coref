import torch

def build_pairs(mention_embs, gold_clusters, spans, tokens):
    pairs = []
    N = len(gold_clusters)

    for i in range(N):
        for j in range(i + 1, N):
            if gold_clusters[i] is None:
                continue

            label = float(gold_clusters[i] == gold_clusters[j])

            same_string = float(
                tokens[spans[i][0]:spans[i][1]] ==
                tokens[spans[j][0]:spans[j][1]]
            )

            pairs.append({
                "mi": mention_embs[i],
                "mj": mention_embs[j],
                "dist": torch.tensor([j - i], dtype=torch.float),
                "same_string": torch.tensor([same_string], dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.float)
            })

    return pairs