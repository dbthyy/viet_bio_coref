import torch.nn as nn
from utils import bio_to_spans
from coref.data.encode import coref_encode
from coref.models.mention import get_mention_embeddings
from coref.models.pair import build_pairs

def train_epoch(model, dataset, encoder, tokenizer, optimizer, device):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, steps = 0.0, 0

    for sample in dataset:
        spans = bio_to_spans(sample["bio_tags"])

        hidden, word_ids = coref_encode(
            encoder, tokenizer, sample["tokens"], device
        )

        mention_embs, clusters, kept_spans = get_mention_embeddings(
            hidden,
            word_ids,
            spans,
            sample["gold_clusters"]
        )

        if mention_embs is None or len(clusters) < 2:
            continue

        pairs = build_pairs(
            mention_embs,
            clusters,
            kept_spans,          
            sample["tokens"]
        )

        optimizer.zero_grad()
        loss = 0.0

        for p in pairs:
            mi = p["mi"].unsqueeze(0)
            mj = p["mj"].unsqueeze(0)
            dist = p["dist"].to(device).unsqueeze(0)
            same_string = p["same_string"].to(device).unsqueeze(0)
            label = p["label"].to(device).unsqueeze(0)

            score = model(mi, mj, dist, same_string)
            loss += loss_fn(score, label)

        loss = loss / len(pairs)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)