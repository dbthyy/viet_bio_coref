"""
Mention detection & BIO tagging from raw text.
- Remove {Mxx ... } tags
- Tokenize and compute char offsets
- Map mentions to token spans
"""

import re
from underthesea import word_tokenize

def remove_M_tags(raw_text):
    prev = None
    clean_text = raw_text

    while prev != clean_text:
        prev = clean_text
        clean_text = re.sub(r"\{M\d+\s*", "", clean_text, flags=re.IGNORECASE).replace("}", "")

    return re.sub(r"\s+", " ", clean_text).strip()

def tokenize(clean_text):
    return word_tokenize(clean_text, format="text").split()

def compute_token_char_offsets(clean_text, tokens):
    offsets = []
    idx = 0
    for tok in tokens:
        while idx < len(clean_text) and clean_text[idx].isspace():
            idx += 1
        start = idx
        end = idx + len(tok)
        offsets.append((start, end))
        idx = end
    return offsets

def extract_identity_mentions(raw_text):
    result = []
    for m in re.finditer(r"\{(m\d+)\s+([^}]+)\}", raw_text, flags=re.IGNORECASE):
        eid = m.group(1).lower()
        mention = m.group(2)

        raw_start, raw_end = m.start(2), m.end(2)

        clean_start = len(remove_M_tags(raw_text[:raw_start]))
        clean_end   = len(remove_M_tags(raw_text[:raw_end]))

        result.append((mention, eid, clean_start, clean_end))
    return result

def mentions_to_token_spans(mentions, token_offsets):
    spans = []

    for _, eid, cs, ce in mentions:
        token_ids = [
            i for i, (ts, te) in enumerate(token_offsets)
            if not (te <= cs or ts >= ce)
        ]

        if token_ids:
            spans.append({
                "start": token_ids[0],
                "end": token_ids[-1],
                "eid": eid
            })

    return spans

def build_bio_tags_from_token_spans(num_tokens, spans):
    tags = ["O"] * num_tokens

    for span in spans:
        s, e = span["start"], span["end"]
        tags[s] = "B-M"
        for i in range(s + 1, e + 1):
            tags[i] = "I-M"

    return tags

def build_bio_from_raw_text(raw_text):
    # 1. clean text
    clean_text = remove_M_tags(raw_text)
    # 2. tokenize
    tokens = tokenize(clean_text)
    # 3. token offsets
    token_offsets = compute_token_char_offsets(clean_text, tokens)
    # 4. extract mentions (char-level)
    mentions = extract_identity_mentions(raw_text)
    # 5. map to token spans
    token_spans = mentions_to_token_spans(mentions, token_offsets)
    # 6. build BIO
    tags = build_bio_tags_from_token_spans(len(tokens), token_spans)

    return tokens, tags, token_spans

def bio_to_spans(tags):
    spans = []
    start = None

    for i, tag in enumerate(tags):
        if tag == "B-M":
            if start is not None:
                spans.append((start, i-1))
            start = i

        elif tag == "I-M":
            if start is None:
                start = i

        else:  
            if start is not None:
                spans.append((start, i-1))
                start = None

    if start is not None:
        spans.append((start, len(tags)-1))

    return spans