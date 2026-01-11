import torch
from transformers import AutoTokenizer, AutoModel

from data_loader import load_data
from coref.data.data_builder import build_coref
from coref.models.scorer import CorefScorer
from coref.main import train_epoch
from coref.training.evaluate import evaluate, evaluate_conll

from config import BASE_DIR, MODEL_NAME, DEVICE, EPOCHS, LR

# 1. Load data
train, val, test = load_data(BASE_DIR)

# 2. 
train = [build_coref(t) for t in train]
val   = [build_coref(t) for t in val]
test  = [build_coref(t) for t in test]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

encoder = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

model = CorefScorer(hidden_dim=encoder.config.hidden_size).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

save_dir = "./coref-checkpoint"
os.makedirs(save_dir, exist_ok=True)

best_val_f1 = -1.0

for epoch in range(EPOCHS):
    train_loss = train_epoch(
        model,
        train,
        encoder,
        tokenizer,
        optimizer,
        DEVICE
    )

    val_p, val_r, val_f1 = evaluate(
        model,
        val,
        encoder,
        tokenizer,
        DEVICE
    )

    print(
        f"[Epoch {epoch+1}] "
        f"train loss={train_loss:.4f} | "
        f"val F1={val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            },
            os.path.join(save_dir, "best_model.pt")
        )

scores = evaluate_conll(
    test,
    encoder,
    tokenizer,
    model,
    DEVICE
)
print(scores)