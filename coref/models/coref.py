from utils import bio_to_spans
from coref.models.mention import get_test_mention_embeddings
from coref.data.encode import coref_encode
from coref.models.scorer import compute_pairwise_scores
from coref.models.clustering import union_find_clustering

def resolve_coreference(tokens, bio_tags, encoder, tokenizer, model, device, threshold=0.5):
    spans = bio_to_spans(bio_tags)
    hidden, word_ids = coref_encode(encoder, tokenizer, tokens, device)

    mention_embs, kept_spans = get_test_mention_embeddings(
        hidden, word_ids, spans
    )

    if mention_embs is None or mention_embs.size(0) < 2:
        return {"spans": kept_spans, "pred_clusters": []}

    scores = compute_pairwise_scores(model, mention_embs)
    pred_clusters = union_find_clustering(scores, threshold=threshold)

    return {
        "spans": kept_spans,        
        "pred_clusters": pred_clusters
    }
