"""
Evaluate NER model using entity-level (span-based) metrics.
Optionally collect a few error samples for debugging.
"""
import torch
from utils import bio_to_spans

def evaluate(test_dataset, model, device, id2label,
    tokenizer=None, debug=False, max_debug_samples=3):
    y_true, y_pred = [], []
    error_samples = []

    model.eval()
    with torch.no_grad():
        for batch in test_dataset:
            input_ids = batch["input_ids"].unsqueeze(0).to(device)
            attention_mask = batch["attention_mask"].unsqueeze(0).to(device)
            gold_labels = batch["labels"].tolist()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pred_labels = outputs.logits.argmax(-1).squeeze(0).cpu().tolist()

            y_true.append(gold_labels)
            y_pred.append(pred_labels)

            if debug and tokenizer is not None:
                decoded = decode_tokens_and_tags(
                    input_ids.cpu().tolist(),
                    gold_labels,
                    pred_labels,
                    tokenizer,
                    id2label
                )
                if any(g != p for _, g, p in decoded):
                    error_samples.append(decoded)
                if len(error_samples) >= max_debug_samples:
                    break

    p, r, f1 = entity_f1(y_true, y_pred, id2label)

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "error_samples": error_samples
    }

def entity_f1(y_true, y_pred, id2label):
    tp = fp = fn = 0

    for t_seq, p_seq in zip(y_true, y_pred):

        valid_idx = [i for i, t in enumerate(t_seq) if t != -100]
        t_seq = [int(t_seq[i]) for i in valid_idx]
        p_seq = [int(p_seq[i]) for i in valid_idx]

        t_tags = [id2label[t] for t in t_seq]
        p_tags = [id2label[p] for p in p_seq]

        gold_spans = set(bio_to_spans(t_tags))
        pred_spans = set(bio_to_spans(p_tags))

        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)

    p = tp / (tp + fp + 1e-10)
    r = tp / (tp + fn + 1e-10)
    f1 = 2 * p * r / (p + r + 1e-10)
    return p, r, f1

def decode_tokens_and_tags(input_ids, labels, preds, tokenizer):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    assert isinstance(labels, list)
    assert isinstance(preds, list)

    result = []
    for tok, gold, pred in zip(tokens, labels, preds):
        if gold == -100:
            continue
        result.append((tok, id2label[gold], id2label[pred]))
    return result