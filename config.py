import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "Coref Dataset/annotated_texts"

MODEL_NAME = "xlm-roberta-base"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

ENCODER = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
encoder.eval()

LABEL2ID = {"O":0, "B-M":1, "I-M":2}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

LR = 2e-5
EPOCHS = 1