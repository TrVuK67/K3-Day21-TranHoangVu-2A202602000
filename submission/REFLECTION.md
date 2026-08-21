# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**  
Điều làm tôi ngạc nhiên nhất là hiện tượng "Nghịch lý Training Loss" ở NB4: cấu hình `attn_only` (với rank matched $r=283$) đạt training loss thấp nhất trong toàn bộ 4 run (0.0531), nhưng khi đánh giá trên tập kiểm thử thực tế (target accuracy ở NB5 §4), nó lại thua cấu hình chuẩn `correct` tới 8 điểm phần trăm (0.7450 so với 0.8250). Điều này cho thấy rõ ràng việc tối ưu hóa loss trên dữ liệu train không đồng nghĩa với năng lực giải quyết bài toán nghiệp vụ.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**  
Tôi mất nhiều thời gian nhất ở việc hiểu và kiểm chứng cơ chế hoạt động của ChatML template kết hợp với Loss Masking ở NB1 (đặc biệt là việc xử lý các block `<think></think>` của mô hình Qwen3.5). Ban đầu tôi nghĩ thời gian huấn luyện GPU ở NB3 và NB4 sẽ là phần mất nhiều công sức nhất, nhưng thực tế việc đảm bảo dữ liệu đầu vào và nhãn loss được ánh xạ chuẩn xác qua offset mapping mới là bước đòi hỏi sự cẩn trọng và kiểm chứng logic khắt khe nhất.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**  
Trước lab này, tôi từng tin rằng:
1. "Chỉ cần tăng rank $r$ của LoRA càng cao thì mô hình sẽ càng mạnh". Thực tế chứng minh việc phủ adapter lên toàn bộ các lớp linear (`text-linear`) quan trọng hơn rất nhiều so với việc chỉ tăng rank trên lớp $q, v$.
2. "QLoRA 4-bit luôn là lựa chọn mặc định tốt nhất cho mọi bài toán tiết kiệm VRAM". Sau khi đo lường thực tế trên dòng mô hình Qwen3.5, tôi nhận ra sai số lượng tử hóa 4-bit gây tụt điểm đáng kể, và nếu GPU đủ chỗ nạp 16-bit thì không nên đánh đổi chất lượng lấy VRAM.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**  
Tôi sử dụng AI assistant để hỗ trợ phân tích luồng dữ liệu của notebook, kiểm tra các điều kiện trong `scripts/verify.py` và hỗ trợ viết code kiểm thử smoke test. AI có xu hướng mặc định đề xuất ép cờ `bf16=True` cho LoRA theo các bài hướng dẫn phổ biến trên A100, nhưng điều này hoàn toàn sai trên kiến trúc GPU Turing (T4) vốn chỉ hỗ trợ `fp16` kèm gradient scaling. Tôi đã phải can thiệp để giữ nguyên cơ chế tự động nhận diện thiết bị của `labkit/device.py`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**  
Bước đầu tiên tôi làm sẽ là: **Xây dựng một tập dữ liệu kiểm thử (Evaluation Set) chuẩn mực và đóng băng mốc Baseline (b) bằng Prompt Engineering chất lượng cao trước khi bắt đầu huấn luyện.** Tôi cần xác định rõ xem bài toán của khách hàng có thực sự cần fine-tune hay không (nếu prompt chuẩn đã giải quyết tốt thì không cần tốn chi phí hạ tầng train mô hình), và thiết lập cơ chế đo lường 4 nhóm chỉ số (Target, Regression, Format, Latency) để có bằng chứng khách quan chứng minh bản fine-tune thực sự mang lại giá trị vượt trội.
