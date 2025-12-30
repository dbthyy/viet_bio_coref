from transformers import AutoTokenizer

def ner_encode(tokenizer, tokens, tags, label2id):
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=512
    )

    word_ids = encoding.word_ids()
    labels = []
    prev_word_id = None

    for word_id in word_ids:
        if word_id is None:
            labels.append(-100)
        elif word_id != prev_word_id:
            labels.append(label2id[tags[word_id]])
        else:
            labels.append(-100)

        prev_word_id = word_id

    encoding["labels"] = labels
    return encoding
