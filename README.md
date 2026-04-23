# Nhận Diện Đồng Tham Chiếu trong Văn Bản Kể Chuyện Tiếng Việt  
Môn học DS304 - Xử lý Ngôn ngữ Tự nhiên

---

## Tổng quan

Dự án tập trung vào việc xây dựng một hệ thống **giải quyết đồng tham chiếu (Coreference Resolution)** cho văn bản kể chuyện tiếng Việt.  
Mục tiêu là nhận diện các đề cập thực thể (entity mentions) và liên kết các đề cập này thành các cụm thực thể thống nhất xuyên suốt văn bản.

Hệ thống được thiết kế theo hướng **huấn luyện mô hình encoder-based**, thay vì sử dụng các mô hình LLM dạng prompt, nhằm:
- Tăng khả năng kiểm soát mô hình  
- Giảm chi phí tính toán  
- Phù hợp với dữ liệu tiếng Việt (ngôn ngữ ít tài nguyên)  

---

## Kiến trúc Hệ thống

Hệ thống được xây dựng theo pipeline gồm 2 giai đoạn chính:

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

---

## Xử lý dữ liệu

Dữ liệu được tiền xử lý nhằm đảm bảo phù hợp với mô hình học có giám sát:

- Làm sạch văn bản (lowercase, loại bỏ nhiễu)
- Tách từ tiếng Việt bằng `underthesea`
- Tokenization theo subword (Transformer tokenizer)
- Chia dữ liệu theo tỷ lệ:
  - Train: 70%
  - Validation: 10%
  - Test: 20%

---

## Thiết kế Mô hình

### Biểu diễn đề cập
Mỗi mention được biểu diễn bằng vector ngữ nghĩa trích xuất từ hidden states của encoder.

### Đặc trưng sử dụng
- Embedding của hai mention  
- Hiệu tuyệt đối (semantic difference)  
- Khoảng cách vị trí trong văn bản  
- Đặc trưng so khớp chuỗi (string match)  

### Mô hình học
- Sử dụng **MLP (Multi-Layer Perceptron)** để tính điểm liên kết giữa các mention  
- Ngưỡng xác suất được sử dụng để quyết định liên kết  

---

## Các mô hình thử nghiệm

Dự án khảo sát nhiều kiến trúc Transformer:

- **PhoBERT**: mô hình đơn ngôn ngữ tiếng Việt  
- **XLM-RoBERTa**: mô hình đa ngôn ngữ  
- **DeBERTa v3**: cải tiến attention  

👉 **Kết quả tốt nhất: XLM-RoBERTa**

---

## Đánh giá mô hình

Hệ thống được đánh giá theo 2 pha:

### Nhận diện đề cập
- Precision  
- Recall  
- F1-score  

### Đồng tham chiếu (theo CoNLL-2012)
- MUC  
- B³  
- CEAFφ4  
- CoNLL F1  

---

## Kết quả chính

- XLM-RoBERTa đạt hiệu năng cao nhất trên tập kiểm tra  
- Word segmentation cải thiện hiệu năng trên tất cả mô hình  
- Các đặc trưng quan trọng:
  - Khoảng cách (distance)
  - Khác biệt ngữ nghĩa (semantic difference)

---

## Công nghệ sử dụng

- Python  
- PyTorch  
- HuggingFace Transformers  
- underthesea  

---

## Hướng phát triển

- Kết hợp mô hình đơn ngôn ngữ và đa ngôn ngữ  
- Tích hợp tri thức ngoài văn bản (knowledge graph)  
- Cải thiện các trường hợp đồng tham chiếu phức tạp  

---

## Tác giả

| MSSV     | Họ và tên            |
|----------|---------------------|
| 23520728 | Đặng Hoàng Gia Khiêm | 
| 23521565 | Võ Ngọc Anh Thy      | 
| 23521563 | Đinh Bảo Thy         | 
---

## 🔗 Link paper: 
