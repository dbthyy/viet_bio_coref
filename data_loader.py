"""
Data loading & preprocessing utilities.

- Load annotated text files by split (train/valid/test)
- Remove metadata lines (#color, #tokenization)
- Lowercase text
- Re-split data for training
"""

import glob, os, re
from sklearn.model_selection import train_test_split

def get_file(base_dir, split_name):
    return glob.glob(os.path.join(base_dir, f"{split_name}_*.txt"))

def load_split(base_dir, split_name):
    file_paths = get_file(base_dir, split_name)
    data_list = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data_list.append(f.read().strip())
        except Exception as e:
            print(f"Error reading {path}: {e}")
    return data_list

def clean_metadata(texts):
    cleaned_texts = []
    for text in texts:
        cleaned = re.sub(r"^(#color:.*|#tokenization.*)$", "", text, flags=re.MULTILINE|re.IGNORECASE).strip().lower()
        cleaned_texts.append(cleaned)
    return cleaned_texts

def load_data(base_dir):
    raw_train = load_split(base_dir, "train")
    raw_valid = load_split(base_dir, "valid")
    raw_test = load_split(base_dir, "test")

    all_data = raw_train + raw_valid + raw_test

    train, temp = train_test_split(all_data, test_size=0.3, random_state=42, shuffle=True)
    val, test = train_test_split(temp, test_size=2/3, random_state=42, shuffle=True)

    train = clean_metadata(train)
    val = clean_metadata(val)
    test = clean_metadata(test)

    return train, val, test