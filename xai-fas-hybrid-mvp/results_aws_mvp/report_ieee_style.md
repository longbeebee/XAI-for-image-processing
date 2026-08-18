# Đánh giá độ tin cậy của mô hình học sâu và phương pháp giải thích trong phát hiện giả mạo khuôn mặt

**Tác giả:** ........................................................  
**Đơn vị:** ........................................................

## Tóm tắt

Phát hiện giả mạo khuôn mặt là một thành phần quan trọng trong các hệ thống xác thực sinh trắc học. Bên cạnh khả năng phân biệt ảnh thật và ảnh giả, một mô hình đáng tin cậy cần duy trì hành vi ổn định khi chất lượng ảnh thay đổi và cần cung cấp các chỉ dấu giải thích có ý nghĩa. Nghiên cứu này đánh giá một mô hình học sâu gọn nhẹ trên tập CelebA-Spoof, đồng thời so sánh hai phương pháp giải thích hậu nghiệm là Grad-CAM và Integrated Gradients. Thực nghiệm sử dụng 10.000 ảnh huấn luyện, 1.000 ảnh thẩm định và 1.000 ảnh kiểm tra. Mô hình đạt diện tích dưới đường cong ROC bằng 0,9446 và độ chính xác trung bình bằng 0,9526. Tuy nhiên, tại điểm quyết định được lựa chọn từ tập thẩm định, tỷ lệ chấp nhận nhầm ảnh giả vẫn ở mức 46,2%. Grad-CAM ổn định hơn Integrated Gradients trên tất cả các thước đo tương đồng không gian. Kết quả khẳng định sự cần thiết của một quy trình đánh giá đa chiều, trong đó hiệu năng phân loại, độ bền vững, tính trung thực của giải thích và giới hạn dữ liệu được xem xét đồng thời.

**Từ khóa—** phát hiện giả mạo khuôn mặt, học sâu, giải thích mô hình, Grad-CAM, Integrated Gradients, CelebA-Spoof, độ tin cậy.

## I. GIỚI THIỆU

Các hệ thống xác thực khuôn mặt có thể bị đánh lừa bởi ảnh in, ảnh hiển thị trên màn hình, mặt nạ hoặc các hình thức trình diễn giả mạo khác. Vì vậy, phát hiện giả mạo khuôn mặt không chỉ là một bài toán phân loại ảnh mà còn là bài toán đánh giá rủi ro trong điều kiện dữ liệu thay đổi.

Độ chính xác dự đoán chưa đủ để phản ánh mức độ đáng tin cậy của mô hình. Một mô hình có thể đạt kết quả tốt trên dữ liệu kiểm tra nhưng dựa vào dấu hiệu phụ như chất lượng nén, điều kiện chiếu sáng hoặc đặc điểm nền ảnh. Nghiên cứu này đánh giá đồng thời hiệu năng phân loại, độ ổn định của dự đoán, độ ổn định của bản đồ giải thích, tính trung thực, kiểm tra sanity và chi phí tính toán.

## II. CƠ SỞ LÝ THUYẾT

### A. Phát hiện giả mạo khuôn mặt

Mục tiêu của phát hiện giả mạo khuôn mặt là xác định liệu ảnh đầu vào được tạo ra từ một khuôn mặt thật hay từ một phương tiện trình diễn giả mạo. Các phương thức giả mạo khác nhau về vật liệu, kết cấu, phản xạ ánh sáng và quan hệ không gian giữa khuôn mặt với môi trường. Do đó, mô hình cần học các đặc trưng liên quan đến khuôn mặt thay vì ghi nhớ một số mẫu dữ liệu cụ thể.

### B. Giải thích mô hình học sâu

Grad-CAM xây dựng bản đồ nổi bật dựa trên thông tin gradient tại các lớp tích chập sâu, qua đó chỉ ra những vùng không gian có ảnh hưởng đến quyết định. Integrated Gradients ước lượng mức đóng góp của từng điểm ảnh bằng cách tích lũy gradient từ một ảnh tham chiếu đến ảnh đầu vào. Độ ổn định phản ánh mức độ bản đồ được giữ nguyên sau biến đổi ảnh; tính trung thực phản ánh mức độ dự đoán thay đổi khi các vùng quan trọng bị loại bỏ hoặc khôi phục; kiểm tra sanity xem xét phản ứng của bản đồ khi mô hình bị ngẫu nhiên hóa.

## III. PHƯƠNG PHÁP NGHIÊN CỨU

### A. Dữ liệu và thiết kế thực nghiệm

Thực nghiệm sử dụng CelebA-Spoof và bảo toàn phân chia chính thức giữa huấn luyện và kiểm tra. Tập huấn luyện gồm 10.000 ảnh, tập thẩm định gồm 1.000 ảnh và tập kiểm tra gồm 1.000 ảnh. Tập huấn luyện và tập kiểm tra được cân bằng theo nhãn. Ảnh được chuyển về RGB, điều chỉnh về kích thước 224 × 224 điểm ảnh và chuẩn hóa theo quy ước của dữ liệu tiền huấn luyện.

Ba nhóm biến đổi chất lượng được sử dụng: giảm độ sáng, làm mờ Gaussian và nén JPEG. Mỗi nhóm có ba mức độ tăng dần. Các biến đổi này không được đưa vào huấn luyện để phép đo phản ánh khả năng chống chịu trước thay đổi dữ liệu chưa được quan sát.

### B. Mô hình và huấn luyện

Mô hình sử dụng MobileNetV3-Small, một kiến trúc tích chập có chi phí tính toán thấp. Trọng số ban đầu được kế thừa từ quá trình học trên tập ảnh tự nhiên quy mô lớn. Lớp phân loại cuối được điều chỉnh cho hai lớp ảnh thật và ảnh giả.

Huấn luyện gồm hai giai đoạn. Giai đoạn đầu chỉ cập nhật phần phân loại; giai đoạn sau tinh chỉnh thêm ba khối đặc trưng cuối. Tổng số vòng huấn luyện là sáu, với kích thước nhóm 32. Bộ tối ưu AdamW, giảm trọng số, cắt biên độ gradient và tính toán độ chính xác hỗn hợp được sử dụng. Mô hình tốt nhất được lựa chọn dựa trên khả năng xếp hạng trên tập thẩm định. Điểm quyết định được giữ cố định khi đánh giá tập kiểm tra.

### C. Đánh giá giải thích

Độ tương đồng giữa các bản đồ được đo bằng tương đồng cosine, tương quan Spearman và mức giao nhau của các vùng nổi bật hàng đầu. Tính trung thực được đánh giá bằng phép lần lượt loại bỏ các vùng quan trọng và phép lần lượt khôi phục chúng từ ảnh tham chiếu. Kiểm tra sanity được thực hiện bằng cách ngẫu nhiên hóa tăng dần các phần của mô hình.

## IV. KẾT QUẢ

### A. Hiệu năng phân loại

| Thước đo | Kết quả | Khoảng tin cậy 95% |
|---|---:|---:|
| Độ chính xác | 0,766 | [0,738; 0,792] |
| Độ chính xác dương | 0,989 | [0,975; 1,000] |
| Độ bao phủ ảnh giả | 0,538 | [0,493; 0,579] |
| F1 | 0,697 | [0,658; 0,730] |
| Diện tích dưới đường cong ROC | 0,945 | [0,930; 0,957] |
| Độ chính xác trung bình | 0,953 | — |
| Tỷ lệ chấp nhận nhầm ảnh giả | 0,462 | [0,421; 0,507] |
| Tỷ lệ từ chối nhầm ảnh thật | 0,006 | [0,000; 0,014] |
| Sai số cân bằng | 0,234 | [0,214; 0,257] |

Ma trận nhầm lẫn cho thấy 497 trên 500 ảnh thật được nhận diện đúng, trong khi chỉ 269 trên 500 ảnh giả được phát hiện. Mô hình vì vậy có xu hướng bảo thủ đối với ảnh thật nhưng chưa đủ nhạy đối với ảnh giả. Phân tích chẩn đoán trên tập kiểm tra cho thấy sai số cân bằng có thể giảm xuống 0,114 nếu tối ưu điểm quyết định trực tiếp trên tập này; kết quả đó không được dùng làm ước lượng cuối vì gây sai lệch đánh giá.

### B. Độ ổn định của dự đoán

| Biến đổi | Mức 1 | Mức 2 | Mức 3 |
|---|---:|---:|---:|
| Làm mờ | 0,985 | 0,931 | 0,846 |
| Giảm độ sáng | 0,977 | 0,960 | 0,941 |
| Nén JPEG | 0,840 | 0,815 | 0,774 |

Nén JPEG gây ảnh hưởng lớn nhất đến quyết định phân loại. Giảm độ sáng gây ảnh hưởng nhỏ nhất. Khi mức độ biến đổi tăng, độ ổn định giảm ở cả ba nhóm.

### C. Độ ổn định của bản đồ giải thích

| Phương pháp | Tương đồng cosine | Tương quan Spearman | Giao nhau vùng 10% | Giao nhau vùng 20% |
|---|---:|---:|---:|---:|
| Grad-CAM | 0,9119 | 0,8186 | 0,5608 | 0,6236 |
| Integrated Gradients | 0,7098 | 0,5580 | 0,3123 | 0,3770 |

Kết quả được tính trên 2.420 cặp có cùng quyết định phân loại. Grad-CAM đạt giá trị cao hơn Integrated Gradients ở cả bốn thước đo. Ưu thế này vẫn được duy trì khi mức độ biến đổi tăng.

### D. Tính trung thực, sanity và chi phí tính toán

Trong phép loại bỏ, diện tích trung bình của Grad-CAM là 0,606 với ảnh tham chiếu trung bình, so với 0,690 của Integrated Gradients. Trong phép khôi phục, Grad-CAM đạt 0,821, còn Integrated Gradients đạt 0,695. Kết quả thay đổi theo ảnh tham chiếu, do đó không nên diễn giải như bằng chứng nhân quả trực tiếp.

Khi mô hình bị ngẫu nhiên hóa ở mức cao nhất, độ tương đồng của các bản đồ giảm về gần bằng không. Đây là dấu hiệu cho thấy giải thích phụ thuộc vào tham số mô hình, nhưng chưa chứng minh rằng vùng nổi bật là nguyên nhân thực sự của quyết định.

Trên GPU Tesla T4, thời gian trung vị cho một dự đoán là 6,32 mili giây, cho Grad-CAM là 7,62 mili giây và cho Integrated Gradients là 47,98 mili giây. Phép loại bỏ cần 13,47 mili giây, trong khi phép khôi phục cần 95,96 mili giây.

### E. Sai số theo loại giả mạo

Tỷ lệ chấp nhận nhầm cao nhất thuộc về các nhóm PC, mặt nạ phần thân trên, poster và A4. Các nhóm pad, mặt nạ vùng và mặt nạ ba chiều được phát hiện tốt hơn. Sự khác biệt cho thấy độ khó phụ thuộc mạnh vào vật liệu, hình thức trình diễn và khoảng cách giữa miền dữ liệu huấn luyện với miền dữ liệu kiểm tra.

## V. THẢO LUẬN

Mô hình có khả năng xếp hạng ảnh tương đối tốt nhưng chưa đạt một điểm vận hành an toàn. Đây là phân biệt quan trọng trong hệ thống chống giả mạo: diện tích dưới đường cong ROC mô tả năng lực xếp hạng trên nhiều điểm quyết định, trong khi hệ thống thực tế chỉ hoạt động tại một điểm cụ thể. Điểm quyết định cần được lựa chọn theo chi phí của việc chấp nhận ảnh giả và từ chối ảnh thật.

Grad-CAM ổn định hơn Integrated Gradients trong toàn bộ phép thử. Một cách diễn giải hợp lý là Grad-CAM tạo ra tín hiệu không gian cô đọng hơn, trong khi Integrated Gradients nhạy với thay đổi cục bộ của điểm ảnh và ảnh tham chiếu. Đây là giả thuyết diễn giải kết quả, không phải bằng chứng về cơ chế nhân quả.

## VI. GIỚI HẠN NGHIÊN CỨU

Thực nghiệm sử dụng một tập con tương đối nhỏ và một kiến trúc chính. Có khả năng cùng một chủ thể xuất hiện ở nhiều phân chia dữ liệu, làm hạn chế việc đánh giá khả năng tổng quát hóa sang chủ thể mới. Nghiên cứu cũng chỉ có một lần chạy chính thức, chưa đánh giá dữ liệu ngoài tập, điều kiện camera thực tế, công bằng giữa các nhóm người dùng hoặc các hình thức giả mạo chưa quan sát.

Các phép loại bỏ và khôi phục có thể tạo ra ảnh nằm ngoài phân phối tự nhiên. Vì vậy, các thước đo trung thực chỉ nên được hiểu là chỉ báo thực nghiệm. Những kết luận mạnh hơn cần được kiểm tra bằng phân chia dữ liệu theo chủ thể, nhiều lần chạy độc lập và đánh giá trên các tập dữ liệu khác.

## VII. KẾT LUẬN

Nghiên cứu đã đánh giá một mô hình học sâu gọn nhẹ cho phát hiện giả mạo khuôn mặt cùng hai phương pháp giải thích hậu nghiệm. Mô hình đạt kết quả xếp hạng tốt, nhưng tỷ lệ bỏ sót ảnh giả tại điểm vận hành được lựa chọn vẫn cao. Grad-CAM cho thấy độ ổn định vượt trội so với Integrated Gradients dưới các biến đổi chất lượng ảnh và có chi phí tính toán thấp hơn. Tuy nhiên, độ ổn định, tính trung thực và sanity không nên được xem là bằng chứng về tính nhân quả hoặc mức độ an toàn tuyệt đối.

## TÀI LIỆU THAM KHẢO

[1] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf. Comput. Vis.*, 2017, pp. 618–626.

[2] M. Sundararajan, A. Taly, and Q. Yan, “Axiomatic attribution for deep networks,” in *Proc. Int. Conf. Mach. Learn.*, 2017, pp. 3319–3328.

[3] Y. Zhang *et al.*, “CelebA-Spoof: Large-scale face anti-spoofing dataset with rich annotations,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit.*, 2020, pp. 1549–1558.

[4] Kết quả thực nghiệm, manifest dữ liệu và báo cáo môi trường của dự án XAI Face Anti-Spoofing MVP.
