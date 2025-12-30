import torch 
from utils import remove_M_tags, tokenize

def predict_bio(
    raw_text,
    ner_model,
    tokenizer,
    id2label,
    device
):
    ner_model.eval()
    ner_model.to(device)

    clean_text = remove_M_tags(raw_text)
    tokens = tokenize(clean_text)

    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        out = ner_model(**enc)

    preds = out.logits.argmax(-1)[0].cpu().tolist()
    word_ids = enc.word_ids()

    bio_tags = ["O"] * len(tokens)
    prev_word_id = None

    for p, wid in zip(preds, word_ids):
        if wid is None:
            continue
        if wid != prev_word_id:
            bio_tags[wid] = id2label[p]
        prev_word_id = wid

    return tokens, bio_tags
