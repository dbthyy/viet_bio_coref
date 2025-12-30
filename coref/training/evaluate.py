from coref.metrics.pairwise import pairwise_f1
from coref.metrics.conll import muc_score, b_cubed_score, ceaf_phi4
from coref.models.coref import resolve_coreference
from coref.data.gold import gold_clusters_from_ids

def evaluate(model, dataset, encoder, tokenizer, device):
    ps, rs, fs = [], [], []
    
    for sample in dataset:
        out = resolve_coreference(
            tokens=sample["tokens"],
            bio_tags=sample["bio_tags"],
            encoder=encoder,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        p, r, f1 = pairwise_f1(
            out["pred_clusters"],
            sample["gold_clusters"]
        )

        ps.append(p)
        rs.append(r)
        fs.append(f1)

    return sum(ps)/len(ps), sum(rs)/len(rs), sum(fs)/len(fs)

def evaluate_conll(dataset, encoder, tokenizer, model, device):
    muc_f, b3_f, ceaf_f = 0, 0, 0
    n_docs = 0

    for sample in dataset:
        out = resolve_coreference(
            tokens=sample["tokens"],
            bio_tags=sample["bio_tags"],
            encoder=encoder,
            tokenizer=tokenizer,
            model=model,
            device=device
        )

        pred = out["pred_clusters"]
        gold = gold_clusters_from_ids(sample["gold_clusters"])

        if len(pred) == 0 or len(gold) == 0:
            continue

        n = sum(len(c) for c in gold)
        
        f = muc_score(pred, gold)
        muc_f += f

        f = b_cubed_score(pred, gold, n)
        b3_f += f

        f = ceaf_phi4(pred, gold)
        ceaf_f += f

        conll_f = muc_f + b3_f + ceaf_f

        n_docs += 1

    return {
        "MUC":   (muc_f/n_docs),
        "B3":    (b3_f/n_docs),
        "CEAF":  (ceaf_f/n_docs),
        "CoNLL_F1": ((conll_f) / (3 * n_docs))
    }