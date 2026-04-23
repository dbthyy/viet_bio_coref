# Nhận Diện Đồng Tham Chiếu trong Văn Bản Kể Chuyện Tiếng Việt  
Môn học DS304 - Xử lý Ngôn ngữ Tự nhiên

---

## Tổng quan

Dự án tập trung vào việc xây dựng một hệ thống **giải quyết đồng tham chiếu (Coreference Resolution)** cho văn bản kể chuyện tiếng Việt.  
Mục tiêu là nhận diện các đề cập thực thể (entity mentions) và liên kết các đề cập này thành các cụm thực thể thống nhất xuyên suốt văn bản.

<img width="897" height="364" alt="image" src="https://github.com/user-attachments/assets/eebcfe12-7587-4b34-a45e-b68b9ccde92a" />

Hệ thống được thiết kế theo hướng **huấn luyện mô hình encoder-based**, thay vì sử dụng các mô hình LLM dạng prompt, nhằm:
- Tăng khả năng kiểm soát mô hình  
- Giảm chi phí tính toán  
- Phù hợp với dữ liệu tiếng Việt (ngôn ngữ ít tài nguyên)  

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
---

## Đánh giá mô hình

Hệ thống được đánh giá theo 2 pha:
<img width="584" height="319" alt="image" src="https://github.com/user-attachments/assets/18db2c88-fba9-4e91-bbf8-912154bb4a68" />
<img width="735" height="600" alt="image" src="https://github.com/user-attachments/assets/c09f3e3e-21a7-40e0-875d-57f5fa3d1a39" />
<img width="802" height="231" alt="image" src="https://github.com/user-attachments/assets/061b084e-1ff5-4748-abc5-c090664136de" />

- XLM-RoBERTa đạt hiệu năng cao nhất trên tập kiểm tra  

---

## Tác giả

| MSSV     | Họ và tên            |
|----------|---------------------|
| 23520728 | Đặng Hoàng Gia Khiêm | 
| 23521565 | Võ Ngọc Anh Thy      | 
| 23521563 | Đinh Bảo Thy         | 
---

## 🔗 Link paper: 
