# Coreference Resolution in Vietnamese Narrative Text

---

## 📌 Table of Contents
- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Model Evaluation](#-model-evaluation)
- [Results](#-results)
- [Tech Stack](#-tech-stack)
- [Related Documents](#-related-documents)
- [Contributors](#-contributors)
- [License](#-license)

---

## 📖 Overview
This project builds a **Coreference Resolution** system for Vietnamese narrative text. The goal is to detect entity mentions within a story and link mentions that refer to the same underlying entity into unified clusters spanning the entire text.

<img width="897" height="364" alt="Overview" src="https://github.com/user-attachments/assets/eebcfe12-7587-4b34-a45e-b68b9ccde92a" />

---

## 🏗 System Architecture
The system is built as a two-stage pipeline:

<img width="1879" height="542" alt="Pipeline architecture" src="https://github.com/user-attachments/assets/53442c29-27ea-4385-8ac9-3fa53ce1d2d2" />

### 1. Mention Detection
- Framed as a **sequence labeling** problem.
- Uses a Transformer-based model (`AutoModelForTokenClassification`).
- Labels follow the **BIO tagging scheme**:
  - `B-M` (Begin Mention)
  - `I-M` (Inside Mention)
  - `O` (Outside)

### 2. Coreference Resolution
- Builds a **pairwise classification** model.
- Predicts the probability that two mentions refer to the same entity.
- Clusters mentions into entities using the **Union-Find** algorithm.

---

## 📁 Project Structure
```bash
.
├── ner/         # Mention Detection module
├── coref/       # Coreference Resolution module
├── inference/   # End-to-end inference pipeline
├── Ablation/    # Ablation experiments and analysis
└── README.md
```

---

## 📊 Model Evaluation

The system was evaluated across three phases: Mention Detection, Coreference Resolution, and full end-to-end Inference.

### Mention Detection

| Model          | Precision | Recall | F1    |
|----------------|-----------|--------|-------|
| PhoBERT        | **85.96** | **85.49** | **85.73** |
| — no word segmentation | 84.19 | 84.88 | 84.53 |
| XLM-RoBERTa    | **90.63** | **93.33** | **91.96** |
| — no word segmentation | 88.99 | 92.97 | 90.94 |
| DeBERTa v3     | **90.74** | 91.74  | **91.24** |
| — no word segmentation | 88.47 | **92.95** | 90.65 |

### Coreference Resolution
*(Ablation setups: **Full** = full feature set, **-dist** = distance feature removed, **-same** = same-sentence feature removed, **-diff** = different-sentence feature removed)*

| Model              | Setup        | MUC F1 | B³ F1 | CEAF F1 | CoNLL F1 |
|--------------------|-------------|--------|-------|---------|----------|
| PhoBERT            | Full        | **96.08** | **40.20** | **50.53** | **62.27** |
|                    | -dist       | 92.54  | 39.67 | 48.57   | 60.26 |
|                    | -same       | 92.55  | 39.54 | 50.48   | 60.86 |
|                    | -diff       | 87.81  | 37.13 | 42.62   | 55.85 |
| XLM-RoBERTa        | Full        | **94.33** | 67.64 | **66.08** | **76.02** |
|                    | -dist       | 93.36  | 66.03 | 62.25   | 73.88 |
|                    | -same       | 93.79  | **67.77** | 64.95   | 75.50 |
|                    | -diff       | 89.88  | 64.50 | 58.96   | 71.12 |
| DeBERTa v3         | Full        | 93.15  | 50.60 | 57.16   | 66.97 |
|                    | -dist       | 92.43  | 50.05 | 55.28   | 65.92 |
|                    | -same       | 91.99  | 49.29 | 50.56   | 65.28 |
|                    | -diff       | **93.34** | **51.47** | **57.85** | **67.42** |

### End-to-End Inference

| Model          | Setup          | MUC   | B³    | CEAF  | CoNLL |
|----------------|----------------|-------|-------|-------|-------|
| PhoBERT        | Full           | 89.35 | 24.89 | 32.25 | 48.83 |
| XLM-RoBERTa    | Full           | **92.67** | **29.94** | **40.87** | **54.49** |
| DeBERTa v3     | -diff feature  | 91.30 | 21.34 | 36.97 | 49.87 |

---

## 🎯 Results
- **XLM-RoBERTa** delivered the strongest overall performance across both individual stages and full end-to-end inference, achieving the highest CoNLL F1 score among the three models tested.
- The ablation studies show that removing the "different-sentence" feature (**-diff**) consistently hurt coreference resolution performance the most for PhoBERT and XLM-RoBERTa, indicating that cross-sentence signal is important for linking mentions in narrative text.
- Interestingly, DeBERTa v3 performed best in its **-diff** ablation setup rather than its full configuration, suggesting model-specific sensitivity to that feature worth investigating further.

---

## 🛠 Tech Stack
* **Language:** Python
* **Modeling Framework:** Hugging Face Transformers (`AutoModelForTokenClassification`)
* **Base Models:** PhoBERT, XLM-RoBERTa, DeBERTa v3
* **Clustering Algorithm:** Union-Find

---

## 📄 Related Documents
- Paper: [DS310_report.pdf](https://github.com/dbthyy/viet_bio_coref/blob/e0d4e6b339265f64a6e68daf033770a209392b77/DS310_report.pdf)

---

## 👥 Contributors

| Student ID | Name |
| :--- | :--- |
| **23520728** | Đặng Hoàng Gia Khiêm |
| **23521565** | Võ Ngọc Anh Thy |
| **23521563** | Đinh Bảo Thy |

---
📜 License
This repository is an academic project created for coursework purposes under course **DS304 - Natural Language Processing** at the **University of Information Technology (UIT), VNU-HCM**. All rights reserved by the project authors.
