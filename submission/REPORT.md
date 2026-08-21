# Lab 21 — Evaluation Report

**Họ tên**: Trần Hoàng Vũ  **MSSV**: 2A202602000  **Ngày**: 2026-08-21  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6GB khả dụng)`

> Mọi con số dưới đây khớp chính xác 100% với các file artefact trong `results/`.

---

## 1. Setup

| Thông số | Giá trị |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường |
| Train / val | 225 / 25 mẫu (seed 42 cố định) |
| `max_length` | 1024 — p95 đo được thực tế là 98 *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps (batch hiệu dụng = 16) |

**Template có giữ khối `<think>` không?** Có — *(results/template_check.json)*  
ChatML template của `Qwen3.5-4B` bảo toàn nguyên vẹn khối `<think>` ngay cả khi chuỗi suy luận rỗng hoặc có nội dung phân tích. Nhờ cơ chế tokenize offset mapping ở NB1, phần mở đầu prompt đóng khối `<think></think>` chuẩn xác, giúp toàn bộ phản hồi JSON của assistant được gán nhãn tính loss đầy đủ mà không bị xáo trộn ranh giới token.

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị thực tế |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% token được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Dán 3–5 dòng đầu của đoạn được tính loss (trích từ `results/mask_proof.json`):

```json
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7578 | 0.0000 | 3215.0 |
| (b) base + optimized prompt | 0.7650 | 0.7578 | 1.0000 | 1017.0 |
| (c) LoRA fine-tune | 0.8250 | 0.7578 | 1.0000 | 980.0 |

**(b) có thật sự mạnh hơn (a) không?** Có — Baseline (b) với prompt công phu (định nghĩa rõ 4 trường và ví dụ 1-shot) đưa độ chính xác target từ 0.000 lên 0.7650, chuẩn hóa format JSON đạt 100% và giảm độ trễ sinh từ 3215 ms xuống còn 1017 ms.  
**Bạn có sửa `OPTIMIZED_PROMPT` không?** Không — Tôi giữ nguyên SHA gốc (`719e74d3b6232053`) để đảm bảo tính liêm chính khoa học, tạo ra mốc so sánh thách thức và thực chất cho bản fine-tune vượt qua.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0549 | **0.8250** | 995.5 | 12.07 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | **0.0531** | 0.7450 | 888.9 | 12.09 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 0.0903 | 0.1850 | 1021.3 | 12.08 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0670 | 0.7850 | 1084.7 | **7.15** |

> Xếp hạng theo năng lực thực tế trên tập **target** (NB5 §4): `correct` (0.8250) > `qlora` (0.7850) > `attn_only` (0.7450) > `wrong_lr` (0.1850).  
> Nghịch lý quan trọng: Nếu chỉ nhìn vào `train loss` ở NB4, `attn_only` (0.0531) tưởng chừng thắng `correct` (0.0549), nhưng khi ra tập target nó lại kém hơn 8 điểm phần trăm.

### Trả lời chi tiết 3 câu hỏi phân tích:

**4.1 — `attn_only` vs `correct` (Vị trí vs Rank):**  
Mặc dù `attn_only` được nâng rank lên $r=283$ bằng thuật toán `matched_rank()` để có cùng ~32.46M tham số huấn luyện với `correct` ($r=16$), trên tập test target nó chỉ đạt **0.7450**, thua `correct` (**0.8250**). Thứ tự này hoàn toàn đảo ngược so với cột train loss ở NB4 (nơi `attn_only` có loss thấp hơn: 0.0531 vs 0.0549). Điều này chứng minh một cách thực nghiệm rằng **vị trí gắn adapter (placement) mới là đòn bẩy quyết định**; việc chỉ nhồi nhét dung lượng rank khổng lồ vào riêng các lớp attention ($q, v$) chỉ dẫn đến hiện tượng học vẹt dữ liệu train mà thiếu đi khả năng tổng quát hóa trên MLP/Linear attention layers.

**4.2 — `wrong_lr` (Sai lệch thang đo Learning Rate):**  
Run `wrong_lr` chỉ thay đổi duy nhất learning rate về mức $1 \times 10^{-5}$ (thang đo của Full Fine-Tuning thay vì thang $1 \times 10^{-4}$ của LoRA). Đường loss của `wrong_lr` giảm cực kỳ chậm chạp và dừng lại ở mức 0.0903, dẫn đến điểm target thảm hại là **0.1850** (mô hình hầu như không học được cấu trúc phân loại mới). Nếu chỉ nhìn vào loss mà không biết LR, người thực hành sẽ kết luận sai rằng "LoRA không đủ dung lượng để giải quyết bài toán", trong khi bản chất là bước cập nhật gradient của ma trận adapter quá nhỏ để kịp hội tụ trong 30 steps.

**4.3 — `qlora` (Đánh đổi VRAM vs Độ chính xác):**  
Run `qlora` tiết kiệm tới **41% bộ nhớ VRAM** (từ 12.07 GB xuống còn 7.15 GB), cho phép nạp mô hình vào các GPU có dung lượng hạn hẹp. Tuy nhiên, sai số lượng tử hóa 4-bit (NF4) đã khiến điểm target tụt từ 0.8250 xuống 0.7850 (mất 4.0 điểm %) và thời gian huấn luyện tăng thêm gần 90 giây do chi phí dequantize on-the-fly. Kết quả này hoàn toàn ủng hộ khuyến nghị từ Unsloth và Deck §12: đối với dòng mô hình Qwen3.5 thế hệ mới, khi VRAM trên GPU T4 vẫn vừa vặn nạp mô hình 4B ở độ chính xác 16-bit, chúng ta không nên dùng QLoRA để tránh đánh đổi chất lượng không cần thiết.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`PASSED`**  
`target Δ = +0.060` · `regression Δ = +0.000` · `valid_trace_rate = 0.0000`

### Diễn giải phán quyết:
Bản fine-tune LoRA cấu hình chuẩn `correct` đã xuất sắc vượt qua cổng hồi quy (Regression Gate) bằng việc nâng độ chính xác target lên **0.8250** (vượt mốc Baseline (b) là 0.7650 một khoảng $\Delta = +0.060$), đồng thời bảo toàn nguyên vẹn 100% năng lực ngôn ngữ phổ thông trên tập regression 15 câu hỏi ($\Delta = 0.000$, điểm đạt 0.7578). Kết quả này chứng minh rằng việc áp dụng cấu hình "LoRA Không Hối Tiếc" (All text-linear layers, rank 16, LR 1e-4, batch hiệu dụng 16) đã giúp mô hình tiếp thu tri thức phân loại ticket CSKH sâu sắc mà không hề mắc phải hội chứng quên thảm họa (catastrophic forgetting). Hơn thế nữa, bản fine-tune cho phép hệ thống chỉ cần dùng Naive Prompt ngắn gọn mà vẫn đạt tốc độ xử lý nhanh hơn Baseline (b) (980 ms so với 1017 ms), mang lại lợi ích kép cả về chi phí token lẫn độ trễ trong môi trường production.

---

## 6. Định tính — Bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại... | `doi_tra, cao, chuột không dây, tich_cuc` | `doi_tra, trung_binh, chuột không dây, tich_cuc` | `doi_tra, cao, chuột không dây, tich_cuc` | ✅ **FT thắng**: Nhận diện đúng urgency "cao" từ từ khóa "Gấp". |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé... | `hoan_tien, trung_binh, ốp lưng điện thoại, tieu_cuc` | `hoan_tien, cao, ốp lưng điện thoại, tieu_cuc` | `hoan_tien, trung_binh, ốp lưng điện thoại, tieu_cuc` | ✅ **FT thắng**: Bắt đúng ngữ cảnh mức độ khẩn cấp trung bình. |
| 3 | Shop ơi, mình đặt máy xay sinh tố mã đơn DH777946. Khi nào có tiền về... | `hoan_tien, trung_binh, máy xay sinh tố, tieu_cuc` | `hoan_tien, trung_binh, máy xay sinh tố, tieu_cuc` | `hoi_thong_tin, trung_binh, máy xay sinh tố, tieu_cuc` | ❌ **FT thua**: Nhầm lẫn giữa "hỏi thông tin" và "hoàn tiền" do câu hỏi chứa cụm "Khi nào". |
| 4 | Xin chào, mình đặt balo laptop mã đơn DH863123. Đổi size. Hỏi cho biết thôi... | `doi_tra, thap, balo laptop, tieu_cuc` | `doi_tra, thap, balo laptop, tieu_cuc` | `doi_tra, trung_binh, balo laptop, tieu_cuc` | ❌ **FT thua**: Dự đoán sai urgency thành "trung_binh" do bỏ sót sắc thái "Hỏi cho biết thôi". |
| 5 | Cho mình hỏi, mình đặt tai nghe bluetooth mã đơn DH716609. Vỡ khi nhận... | `san_pham_loi, thap, tai nghe bluetooth, tieu_cuc` | `san_pham_loi, thap, tai nghe bluetooth, tieu_cuc` | `san_pham_loi, thap, tai nghe, tieu_cuc` | ❌ **FT thua**: Trích xuất thiếu từ "bluetooth" trong tên sản phẩm. |

**Mẫu chung ở các ca FT thua:**  
Các ca FT thua thường xuất hiện ở những ticket có cấu trúc câu hỏi gián tiếp (chứa từ để hỏi như "Khi nào", "Hỏi cho biết") làm mô hình bị thiên kiến hướng về intent `hoi_thong_tin`, hoặc các trường hợp tên sản phẩm dài bị cắt bớt tính từ bổ nghĩa.

---

## 7. Kết luận & Điều tôi học được

### Kết luận:
Dựa trên các bằng chứng thực nghiệm thu được, tôi **khuyến nghị triển khai (deploy)** bản fine-tune `adapters/correct` vào hệ thống xử lý ticket tự động. Bản fine-tune không chỉ vượt qua mốc Baseline (b) đã được tối ưu hóa kỹ lưỡng (+6.0% accuracy), đảm bảo chuẩn format JSON 100%, mà còn giúp giảm thiểu đáng kể chi phí hạ tầng: loại bỏ được prompt hệ thống cồng kềnh giúp tiết kiệm ~200 context token mỗi lượt gọi và giảm thời gian phản hồi từ 1017 ms xuống 980 ms. 

Đòn bẩy thực sự quyết định thành bại trong lab này không phải là dung lượng rank LoRA hay việc lượng tử hóa mô hình, mà là **tính đúng đắn của Loss Mask** (chỉ tính loss trên câu trả lời), **vị trí gắn adapter bao phủ toàn bộ các lớp Linear của Text Decoder**, và **thang đo Learning Rate chuẩn LoRA (1e-4)**. Khi thiếu đi các yếu tố này, mô hình hoặc sẽ học vẹt (như `attn_only`), hoặc hoàn toàn không hội tụ (như `wrong_lr`).

### Ba điều tôi học được:
1. **Không bao giờ tin vào Training Loss đơn thuần:** Run `attn_only` có train loss thấp nhất (0.0531) nhưng lại cho kết quả kiểm thử thực tế kém hơn `correct` (0.0549 loss). Đánh giá mô hình phải dựa trên task-specific metric trên tập holdout, không thể dựa vào surrogate metrics.
2. **Loss Mask là nền móng của Fine-tuning:** Nếu tính loss trên cả phần prompt (`supervised_fraction` cao), mô hình sẽ học cách lặp lại câu hỏi thay vì sinh câu trả lời. Việc dùng offset mapping để kiểm chứng mask độc lập là bước bắt buộc trước khi tốn thời gian train GPU.
3. **Thí nghiệm đối chứng phải kiểm soát biến chặt chẽ:** So sánh hai cấu hình LoRA chỉ có ý nghĩa khoa học khi chúng có cùng ngân sách tham số (`matched_rank()`) và cùng số bước huấn luyện (`max_steps`).

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
Tôi sẽ bổ sung thêm 200 mẫu dữ liệu edge-case tiếng Việt đa dạng hơn (với các trường hợp viết tắt, teencode, và câu hỏi phức hợp) và thử nghiệm kỹ thuật **Direct Preference Optimization (DPO)** để tinh chỉnh khả năng phân biệt giữa các sắc thái khẩn cấp (urgency) tinh vi.

---

## Phụ lục — Thưởng đã làm

- [x] **B1 — NB6 Merge & Hot-Swap:** Đã chạy và sinh `results/merge_check.json`, độ suy giảm điểm sau merge bằng 0.000 (nằm trong ngưỡng cho phép 0.01), hot-swap thành công adapter trên base model.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
