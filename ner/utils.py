"""
NER utility functions.
- Split token sequence into sentences
- Chunk sentences to max sequence length
"""

def split_into_sentences(tokens, tags):
    sentences, cur_tokens, cur_tags  = [], [], []
    sentence_end_tokens={".","!","?"}

    for tok, tag in zip(tokens, tags):
        cur_tokens.append(tok)
        cur_tags.append(tag)

        if tok in sentence_end_tokens:
            sentences.append((cur_tokens, cur_tags))
            cur_tokens, cur_tags = [], []

    if cur_tokens:
        sentences.append((cur_tokens, cur_tags))

    return sentences

def chunk_by_sentences(sentences, max_len):
    chunks, cur_tokens, cur_tags = [], [], []

    for sent_tokens, sent_tags in sentences:

        if len(sent_tokens) > max_len:
            for i in range(0, len(sent_tokens), max_len):
                chunks.append((
                    sent_tokens[i:i+max_len],
                    sent_tags[i:i+max_len]
                ))
            continue

        if len(cur_tokens) + len(sent_tokens) <= max_len:
            cur_tokens += sent_tokens
            cur_tags   += sent_tags
        else:
            chunks.append((cur_tokens, cur_tags))
            cur_tokens = sent_tokens
            cur_tags   = sent_tags

    if cur_tokens:
        chunks.append((cur_tokens, cur_tags))

    return chunks