# Báo cáo nghiên cứu: Đánh giá mô hình và giải thích XAI trong phát hiện giả mạo khuôn mặt

## Tóm tắt

Nghiên cứu đánh giá một pipeline face anti-spoofing (FAS) kết hợp phân loại ảnh và giải thích hậu nghiệm. Mô hình sử dụng MobileNetV3-Small tiền huấn luyện ImageNet, được tinh chỉnh cho phân loại nhị phân với quy ước `0 = real`, `1 = spoof`. Hai phương pháp XAI được so sánh là Grad-CAM và Integrated Gradients (IG). Thực nghiệm chính chạy trên AWS CUDA với NVIDIA Tesla T4, sử dụng tập con CelebA-Spoof gồm 10.000 ảnh train, 1.000 ảnh validation và 1.000 ảnh test.

Mô hình đạt ROC-AUC = 0,9446, average precision = 0,9526 và accuracy = 0,766 trên test tại threshold được chọn từ validation. Tuy nhiên, APCER/FAR = 0,462 và BPCER/FFR = 0,006 cho thấy mô hình còn bỏ sót nhiều ảnh spoof. Grad-CAM ổn định hơn IG trên tất cả các chỉ số PCES. Kết quả cho thấy cần đánh giá đồng thời hiệu năng phân loại, độ ổn định, faithfulness và sanity thay vì dựa trên một metric duy nhất.

**Từ khóa:** face anti-spoofing, CelebA-Spoof, MobileNetV3-Small, Grad-CAM, Integrated Gradients, XAI.

## 1. Mục tiêu nghiên cứu

Mục tiêu là xây dựng pipeline có khả năng tái lập để: (i) đánh giá mô hình FAS; (ii) đo độ bền vững của dự đoán trước biến đổi chất lượng ảnh; (iii) so sánh Grad-CAM và IG; và (iv) kiểm tra faithfulness, sanity, runtime và sai số theo loại spoof.

## 2. Dữ liệu và thiết lập thực nghiệm

| Split | Số ảnh | Phân bố nhãn |
|---|---:|---|
| Train | 10.000 | 5.000 real, 5.000 spoof |
| Validation | 1.000 | 261 real, 739 spoof |
| Test | 1.000 | 500 real, 500 spoof |

Seed thực nghiệm là 42. Ảnh được chuyển RGB, resize 224 × 224 và chuẩn hóa theo ImageNet mean `(0,485; 0,456; 0,406)` và standard deviation `(0,229; 0,224; 0,225)`. Augmentation gồm random resized crop, horizontal flip và ColorJitter nhẹ. Các perturbation nghiên cứu không được dùng trong huấn luyện.

Ba nhóm biến đổi được áp dụng trên test là brightness `(0,8; 0,6; 0,4)`, Gaussian blur với σ `(0,5; 1,0; 2,0)` và JPEG quality `(70; 40; 20)`. Kiểm tra provenance không phát hiện duplicate relative path, duplicate image ID hoặc split overlap theo protocol; tuy nhiên có `subject_overlap = true`, là một giới hạn đối với khả năng tổng quát hóa theo chủ thể.

Môi trường chính thức gồm Python 3.11.15, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, Captum 0.9.0, CUDA 12.8 và Tesla T4. Bảy validation gate đều đạt.

## 3. Kiến trúc và huấn luyện

MobileNetV3-Small dùng trọng số ImageNet; lớp cuối được thay bằng `Linear(in_features, 2)`. Huấn luyện gồm hai giai đoạn: 2 epoch đầu chỉ cập nhật classifier, sau đó mở khóa 3 block cuối của feature extractor trong tổng số 6 epoch. Các thông số chính:

| Thông số | Giá trị |
|---|---:|
| Batch size | 32 |
| Head learning rate | 0,001 |
| Fine-tuning learning rate | 0,0001 |
| Optimizer | AdamW |
| Weight decay | 0,0001 |
| Gradient clipping | 1,0 |
| Mixed precision | Có, CUDA |
| DataLoader workers | 4 |

Checkpoint được chọn theo validation ROC-AUC. Threshold được chọn từ validation bằng chiến lược `min_acer` và giữ cố định khi đánh giá test. Artifact phân loại ghi threshold 0,9900677; threshold summary ghi 0,9904602 do được tái tính ở bước khác. Các điểm tối ưu trực tiếp trên test chỉ mang tính chẩn đoán.

## 4. Phương pháp giải thích và metric

Grad-CAM sử dụng convolutional layer cuối. IG dùng baseline zero, `n_steps = 24` và `internal_batch_size = 8`. Độ ổn định bản đồ được đo bằng cosine similarity, Spearman correlation, top-10 IoU và top-20 IoU trên các cặp prediction-conditioned, tức chỉ giữ các trường hợp dự đoán không đổi.

Faithfulness được đo bằng patch deletion và insertion với patch size 32, 10 bước, hai baseline mean và blur. Deletion AUC thấp hơn là tốt hơn; insertion AUC cao hơn là tốt hơn. Sanity test ngẫu nhiên hóa tăng dần các phần của mô hình.

## 5. Kết quả phân loại

| Chỉ số | Kết quả | Bootstrap 95% CI |
|---|---:|---:|
| Accuracy | 0,766 | [0,738; 0,792] |
| Precision | 0,989 | [0,975; 1,000] |
| Recall | 0,538 | [0,493; 0,579] |
| F1 | 0,697 | [0,658; 0,730] |
| ROC-AUC | 0,945 | [0,930; 0,957] |
| Average precision | 0,953 | — |
| APCER/FAR | 0,462 | [0,421; 0,507] |
| BPCER/FFR | 0,006 | [0,000; 0,014] |
| ACER | 0,234 | [0,214; 0,257] |
| TAR | 0,994 | [0,986; 1,000] |

Ma trận nhầm lẫn là `[[497, 3], [231, 269]]`. Mô hình chỉ từ chối nhầm 3 ảnh real nhưng bỏ sót 231 ảnh spoof. Điểm min-ACER chẩn đoán trên test đạt ACER = 0,114 tại threshold 0,0704; điểm EER chẩn đoán có FAR = FFR = 0,124 tại threshold 0,0201. Hai giá trị này không thay thế threshold validation.

## 6. Độ ổn định dự đoán

| Perturbation | Mức 1 | Mức 2 | Mức 3 |
|---|---:|---:|---:|
| Blur | 0,985 | 0,931 | 0,846 |
| Brightness | 0,977 | 0,960 | 0,941 |
| JPEG | 0,840 | 0,815 | 0,774 |

JPEG làm thay đổi dự đoán nhiều nhất; brightness gây suy giảm ít nhất.

## 7. Độ ổn định của giải thích

Trên 2.420 cặp prediction-conditioned:

| Phương pháp | Cosine | Spearman | Top-10 IoU | Top-20 IoU |
|---|---:|---:|---:|---:|
| Grad-CAM | 0,9119 | 0,8186 | 0,5608 | 0,6236 |
| IG | 0,7098 | 0,5580 | 0,3123 | 0,3770 |

Grad-CAM vượt IG ở cả bốn metric. Các kiểm định Wilcoxon paired với Holm correction đều có adjusted p-value được ghi nhận là 0,0 trong artifact, tức nhỏ hơn độ chính xác hiển thị của file.

## 8. Faithfulness, sanity và runtime

| Phương pháp | Baseline | Deletion AUC | Insertion AUC |
|---|---|---:|---:|
| Grad-CAM | Mean | 0,606 | 0,821 |
| Grad-CAM | Blur | 0,734 | 0,908 |
| IG | Mean | 0,690 | 0,695 |
| IG | Blur | 0,804 | 0,816 |

Grad-CAM có kết quả thuận lợi hơn trong thiết lập faithfulness này, nhưng metric phụ thuộc rõ rệt vào baseline. Ở randomization level cao nhất, cosine similarity và Spearman correlation của cả hai phương pháp giảm về 0 trong kết quả tổng hợp, phù hợp với yêu cầu sanity.

| Operation | Mean (ms) | Median (ms) |
|---|---:|---:|
| Classifier prediction | 6,32 | 6,32 |
| Grad-CAM | 7,62 | 7,62 |
| Integrated Gradients | 47,99 | 47,98 |
| Deletion | 13,47 | 13,47 |
| Insertion | 95,97 | 95,96 |

## 9. Phân tích theo loại spoof

APCER cao nhất ở PC (1,000), upper_body_mask (0,956), poster (0,548) và A4 (0,513). Các nhóm pad (0,081), region_mask (0,118) và three_d_mask (0,140) được phát hiện tốt hơn. Sai khác này cho thấy mô hình nhạy với loại vật liệu, hình thức trình diễn và miền dữ liệu.

## 10. Thảo luận và giới hạn

ROC-AUC cao nhưng APCER tại threshold validation vẫn lớn, vì vậy hiệu năng xếp hạng không đảm bảo operating point an toàn cho FAS. Threshold cần được chọn theo chi phí APCER/BPCER và yêu cầu triển khai. Stability không đồng nghĩa với faithfulness; faithfulness phụ thuộc baseline; sanity chỉ chứng minh explanation phụ thuộc một phần vào mô hình, không chứng minh tính nhân quả.

Các giới hạn chính gồm: tập dữ liệu chỉ là subset; có subject overlap giữa split; chỉ có một backbone và một lần chạy chính thức; chưa có cross-dataset evaluation, nhiều seed, fairness analysis hoặc điều kiện camera thực tế; chưa có log loss theo epoch nên không thể đánh giá đường học và overfitting; patch perturbation có thể tạo ảnh ngoài phân phối.

## 11. Kết luận

Pipeline đã chứng minh khả năng đánh giá đồng thời classification, robustness, explanation stability, faithfulness, sanity và runtime trên CUDA. MobileNetV3-Small đạt ROC-AUC 0,9446, nhưng mô hình hiện tại chưa phù hợp để xem là hệ thống production vì spoof detection rate tại threshold được chọn chỉ đạt 0,538. Grad-CAM ổn định hơn IG trong toàn bộ đánh giá PCES và có faithfulness thuận lợi hơn trong cấu hình hiện tại.

Các bước tiếp theo nên gồm tách subject nghiêm ngặt giữa các split, mở rộng dữ liệu và nhiều seed, calibration threshold theo chi phí lỗi, cải thiện các nhóm PC/upper_body_mask, bổ sung baseline kiến trúc và đánh giá cross-dataset.

## 12. Artifact tái lập

Thông số và kết quả được lấy từ `configs/aws_mvp.example.yaml`, các module trong `src/`, cùng các file `metrics/*.json`, `metrics/*.csv`, `validation_gates.json` và `environment_report.json` trong thư mục kết quả.
