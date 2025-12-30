from utils import build_bio_from_raw_text, bio_to_spans

def build_coref(raw_text):
    tokens, bio_tags, token_spans = build_bio_from_raw_text(raw_text)
    spans = bio_to_spans(bio_tags)

    gold_clusters = []
    for s, e in spans:
        eid = None
        for ts in token_spans:
            if not (e <= ts["start"] or s > ts["end"]):
                eid = ts["eid"]
                break
        gold_clusters.append(eid)

    return {
        "tokens": tokens,
        "bio_tags": bio_tags,
        "token_spans": token_spans,
        "gold_clusters": gold_clusters
    }