import torch

def coref_encode(encoder, tokenizer, tokens, device):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True
    ).to(device)

    with torch.no_grad():
        out = encoder(**enc)

    hidden = out.last_hidden_state[0]   
    word_ids = enc.word_ids()           

    return hidden, word_ids