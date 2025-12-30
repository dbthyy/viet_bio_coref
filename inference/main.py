"""
Run end-to-end evaluation: NER → Coreference → CoNLL metrics
"""
from data_loader import load_data
from utils import bio_to_spans
from config import BASE_DIR, DEVICE

from coref.data.data_builder import build_coref
from coref.training.evaluate import evaluate_conll
from coref.models.scorer import CorefScorer
from inference.convert import (
    gold_eids_to_span_clusters,
    clusters_from_ids_to_spans
)
from inference.ner import predict_bio
from inference.coref import run_coref
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel

# Load data
_, _, test = load_data(BASE_DIR)

# Load ner
ner_model = AutoModelForTokenClassification.from_pretrained("ner/ner-checkpoint")
ner_tokenizer = AutoTokenizer.from_pretrained("./ner-checkpoint")

# Load coref
coref_encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
coref_encoder.eval()
for p in encoder.parameters():
    p.requires_grad = False

hidden_dim = encoder.config.hidden_size
coref_model = CorefScorer(hidden_dim=hidden_dim).to(DEVICE)

checkpoint = torch.load("coref/coref-checkpoint/best_model.pt", map_location=DEVICE)
coref_model.load_state_dict(checkpoint["model_state_dict"])
coref_model.eval() 

gold_instances = []
pred_instances = []

for raw_text in test:
    # ----- GOLD -----
    gold = build_coref(raw_text)
    spans = bio_to_spans(gold["bio_tags"])

    gold_clusters = gold_eids_to_span_clusters(
        spans, gold["gold_clusters"]
    )

    gold_instances.append({
        "tokens": gold["tokens"],
        "clusters": gold_clusters
    })

    # ----- PRED -----
    tokens, bio_tags = predict_bio(
        raw_text, 
        ner_model, 
        TOKENIZER, 
        DEVICE
    )

    out = run_coref(
        tokens, 
        bio_tags,
        coref_encoder, 
        TOKENIZER,
        coref_model, 
        DEVICE
    )

    pred_clusters = clusters_from_ids_to_spans(
        out["pred_clusters"], out["spans"]
    )

    pred_instances.append({
        "tokens": tokens,
        "clusters": pred_clusters
    })

# Evaluate
scores = evaluate_conll(gold_instances, pred_instances)
for k, v in scores.items():
    print(k, v)