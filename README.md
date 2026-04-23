# Nhận Diện Đồng Tham Chiếu trong Văn Bản Kể Chuyện Tiếng Việt  
Môn học DS304 - Xử lý Ngôn ngữ Tự nhiên

---

## Tổng quan

Dự án tập trung vào việc xây dựng một hệ thống **giải quyết đồng tham chiếu (Coreference Resolution)** cho văn bản kể chuyện tiếng Việt.  
Mục tiêu là nhận diện các đề cập thực thể (entity mentions) và liên kết các đề cập này thành các cụm thực thể thống nhất xuyên suốt văn bản.

<img width="897" height="364" alt="image" src="https://github.com/user-attachments/assets/eebcfe12-7587-4b34-a45e-b68b9ccde92a" />

---

## Kiến trúc Hệ thống

Hệ thống được xây dựng theo pipeline gồm 2 giai đoạn chính:

<img width="1879" height="542" alt="image" src="https://github.com/user-attachments/assets/53442c29-27ea-4385-8ac9-3fa53ce1d2d2" />

### 1. Nhận diện đề cập (Mention Detection)
- Bài toán: Sequence Labeling  
- Sử dụng mô hình Transformer (`AutoModelForTokenClassification`)  
- Gán nhãn theo chuẩn BIO:
  - B-M (Begin Mention)
  - I-M (Inside Mention)
  - O (Outside)

### 2. Giải quyết đồng tham chiếu (Coreference Resolution)
- Xây dựng mô hình phân loại cặp (pairwise classification)  
- Dự đoán xác suất hai mention có cùng thực thể  
- Gom cụm bằng thuật toán **Union-Find**  

### 3. Cấu trúc Repository
```
.
├── ner/         # Module nhận diện đề cập (Mention Detection)
├── coref/       # Module giải quyết đồng tham chiếu
├── inference/   # Pipeline suy diễn end-to-end
├── Ablation/    # Thí nghiệm ablation và phân tích
└── README.md
```

---

## Đánh giá mô hình

Hệ thống được đánh giá theo 2 pha:
- Mention Detection:
  
| Model          | Precision | Recall | F1    |
|----------------|----------|--------|-------|
| PhoBERT        | **85.96** | **85.49** | **85.73** |
| -word seg      | 84.19    | 84.88  | 84.53 |
| XLM-RoBERTa    | **90.63** | **93.33** | **91.96** |
| -word seg      | 88.99    | 92.97  | 90.94 |
| DeBERTa v3     | **90.74** | 91.74  | **91.24** |
| -word seg      | 88.47    | **92.95** | 90.65 |

- Coreference Resolution:
  
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

- Inference:

| Model          | Setup          | MUC   | B³    | CEAF  | CoNLL |
|----------------|----------------|-------|-------|-------|-------|
| PhoBERT        | Full           | 89.35 | 24.89 | 32.25 | 48.83 |
| XLM-RoBERTa    | Full           | **92.67** | **29.94** | **40.87** | **54.49** |
| DeBERTa v3     | -diff feature  | 91.30 | 21.34 | 36.97 | 49.87 |

Kết quả suy diễn trên tập kiểm tra cho thấy XLM-RoBERTa đạt hiệu năng tổng thể cao nhất với điểm CoNLL F1 vượt trội so với hai mô hình còn lại.

---

## Tác giả

| MSSV     | Họ và tên            |
|----------|---------------------|
| 23520728 | Đặng Hoàng Gia Khiêm | 
| 23521565 | Võ Ngọc Anh Thy      | 
| 23521563 | Đinh Bảo Thy         | 
---

## 🔗 Link paper: [Link](https://github.com/dbthyy/viet_bio_coref/blob/e0d4e6b339265f64a6e68daf033770a209392b77/DS310_report.pdf)
