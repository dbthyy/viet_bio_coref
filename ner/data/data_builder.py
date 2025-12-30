"""
Build NER training data from raw annotated texts:
    Transforms raw texts into BIO-labeled (utils.py), 
    chunked(ner/ner_utils.py), 
    encoded samples(ner/encoding.py), 
    then wraps them into a PyTorch Dataset(ner/dataset.py).
"""
from utils import build_bio_from_raw_text
from ner.utils import split_into_sentences, chunk_by_sentences
from ner.data.encode import ner_encode
from ner.data.dataset import NERDataset

def build_ner(raw_texts, tokenizer_name, label2id, max_len=512):
    bio = create_bio(raw_texts, max_len=max_len)

    encodings = [
        ner_encode(tokenizer_name, tokens, tags, label2id)
        for tokens, tags in bio
    ]

    dataset = NERDataset(encodings)

    return dataset


def create_bio(dataset_split, max_len):
    dataset = []

    for raw_text in dataset_split:
        # 1. Build BIO at document-level
        tokens, tags, _ = build_bio_from_raw_text(raw_text)
        # 2. Split into sentences at token-level
        sentences = split_into_sentences(tokens, tags)
        # 3. Chunk by sentence with max_len
        chunks = chunk_by_sentences(sentences, max_len=max_len)
        # 4. Collect
        for tok_chunk, tag_chunk in chunks:
            assert len(tok_chunk) == len(tag_chunk)
            dataset.append((tok_chunk, tag_chunk))

    return dataset