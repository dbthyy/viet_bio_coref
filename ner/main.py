"""
Train script for NER model.
- Build datasets
- Initialize model & trainer
- Run training and evaluation
"""
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer

from data_loader import load_data
from ner.data.data_builder import build_ner
from ner.training.evaluate import evaluate
from config import BASE_DIR, LABEL2ID, ID2LABEL, TOKENIZER, MODEL_NAME, DEVICE

# Load data
train, val, test = load_data(BASE_DIR)

# Build datasets
train_dataset = build_ner(train, TOKENIZER, LABEL2ID)
val_dataset   = build_ner(val, TOKENIZER, LABEL2ID)
test_dataset = build_ner(test, TOKENIZER, LABEL2ID)

# Model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL2ID),
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

# Trainer
training_args = TrainingArguments(
    output_dir="./ner-checkpoint", 
    eval_strategy="epoch", 
    save_strategy="epoch",
    learning_rate=LR, 
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8,
    num_train_epochs=EPOCHS, 
    weight_decay=0.01, 
    logging_steps=20, 
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", 
    greater_is_better=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()

# Evaluate
results = evaluate(test_dataset, model, DEVICE, ID2LABEL)
print(results)

# 6+ Debug
# results = evaluate(test_dataset, model, DEVICE, ID2LABEL, tokenizer=TOKENIZER, debug=True)