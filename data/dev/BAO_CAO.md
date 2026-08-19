# Báo cáo chuẩn bị dữ liệu benchmark — dev corpus, cutoff 2022

Ngày: 19/08/2026. Người thực hiện: dev@wigin.ai.
Phạm vi: chuẩn bị và nghiệm thu bộ dữ liệu dev (`data/dev/`) cho stage 1 của
benchmark wigin-tllm — đo chronological consistency (lookahead bias) của LLM.
**Không sửa bất kỳ dòng code chấm điểm nào trong `wigin_tllm/`** — toàn bộ vấn
đề được giải quyết ở tầng dữ liệu.

## 1. Vấn đề ban đầu

Bộ dev corpus (100 facts, 2015–2025, cutoff 2022, 50 probe mỗi phía) dùng
ngưỡng epsilon mặc định **−11.51** — đặt theo quy ước, không phải đo được.
Ba thí nghiệm chẩn đoán cho thấy dataset không dùng được để phán xử model:

1. **Token audit**: epsilon chặn trên *tổng* log-prob của phrase, trong khi
   phrase dài 1–7 token BPE. Yêu cầu xác suất mỗi token do đó trải hơn 4 bậc
   độ lớn (10⁻⁵ với phrase 1 token, ~0.19 với phrase 7 token) — ngưỡng phân
   loại probe **theo độ dài phrase chứ không theo kiến thức**.
2. **Model tham chiếu sạch bị oan**: checkpoint `chrono-gpt-v1-20221231`
   (dữ liệu train dừng đúng 31/12/2022) chạy trên corpus cũ bị gắn nhãn
   **leaker** — 42% probe "tương lai" vượt ngưỡng so với mức cho phép 10%.
3. **Không tồn tại ngưỡng vô hướng nào cứu được corpus cũ**: quét toàn dải
   [−20, −1] không có epsilon nào thoả đồng thời known > 70% và unknown ≤ 10%;
   thử cả chuẩn hóa per-token cũng không tách được — chứng tỏ lỗi nằm ở
   chính các probe.

## 2. Chẩn đoán chi tiết (từ điểm thô từng probe)

Dump điểm thô 100 probe × 3 checkpoint (`tools/dump_scores.py`,
`data/dev/raw_scores.json`) chỉ ra hai mẫu lỗi dữ liệu:

- **Phía unknown (sự kiện sau cutoff): nhiều đáp án đoán được từ prompt bằng
  prior ngôn ngữ**, không cần biết tương lai — ví dụ "Mandalay" (prompt đã nói
  động đất Myanmar), "Valencia" (lũ Tây Ban Nha → thành phố lớn), "Taylor
  Swift" (nhạc sĩ của năm → người nổi nhất), "Jannik Sinner" (tên đa token
  được teacher forcing mớm trước token đầu). Model sạch chấm các probe này
  điểm cao → chúng làm model sạch trông như leaker.
- **Phía known (sự kiện trước cutoff): nhiều fact quá hiếm hoặc quá sát mép
  cutoff** — "Lubitz", "Harambe", "Roma" (hiếm với model 1.5B); "FTX" (sập
  11/2022), "Midjourney" (9/2022) — dữ liệu train luôn trễ so với thời điểm
  thu thập nên model cutoff 31/12/2022 chưa kịp hấp thụ sự kiện cuối 2022.

## 3. Biện pháp xử lý (chỉ sửa dữ liệu)

Viết lại `data/dev/facts.json` qua 2 vòng lặp, mỗi vòng đều **pre-screen bằng
chính model tham chiếu sạch** trước khi chốt:

1. **Chuẩn hóa độ dài**: mọi phrase đúng 1–2 token GPT-2 (encode kèm dấu cách
   đầu, đúng cách `scoring/leak.py` làm); phân bố độ dài khớp giữa hai phía.
   Nhờ đó epsilon vô hướng so sánh được các probe với nhau.
2. **Phía unknown**: loại mọi probe model sạch chấm > ≈ −3.5/token (tức đoán
   được bằng prior), thay bằng đáp án "chống prior" — thực thể chưa tồn tại
   trước 2023 hoặc tên không suy ra được (Raygun, Pop Mart, RedNote, Willow,
   CrowdStrike→Willow, Tea, JJ, Humane…).
3. **Phía known**: thay fact hiếm và fact sát mép cutoff bằng fact đại chúng
   cách cutoff ≥ vài tháng (Wuhan→Zoom, Wordle, AlphaGo, Markle, Maria,
   GDPR, Mars…).
4. **Tuân thủ invariant của repo** (`tests/test_dev_corpus.py`, 11/11 pass):
   50/50 fact mỗi phía, mỗi năm ≥ 6 fact, không phrase trùng lặp, không trùng
   với demo corpus `examples/sample/`, phân bố độ dài từ khớp hai phía.

## 4. Kết quả nghiệm thu

**Calibration** (đo bằng model sạch 2022, lệnh `wigin-tllm corpus
--calibrate-with …`):

| chỉ số | vòng 1 (data cũ) | vòng 2 (data mới) |
|---|---|---|
| epsilon đo được | −3.8248 | **−8.8792** |
| known rate của model sạch | 22% (cần >70%) | **94%** |
| unknown rate của model sạch | 4% | **4%** |
| verdict | DOES NOT SEPARATE, exit 1 | **separates, margin 6%, exit 0** |

**Control experiment** (3 checkpoint ChronoGPT trên corpus đã calibrate —
đây là bài nghiệm thu cuối):

| model (cutoff) | unknown vượt ε | known vượt ε | leak score | verdict | kỳ vọng |
|---|---|---|---|---|---|
| 20221231 (sạch) | 2/50 (4%) | 47/50 (94%) | **−8.87** (normalised 1.0) | **PASS** | PASS ✓ |
| 20231231 (thấy 2023) | 10/50 (20%) | 47/50 (94%) | 0.0 | **FAIL — leaker** | FAIL ✓ |
| 20241231 (thấy 2023–24) | 19/50 (38%) | 45/50 (90%) | 0.0 | **FAIL — leaker** | FAIL ✓ |

- Tín hiệu leak **đơn điệu nghiêm ngặt** theo cutoff: 4% → 20% → 38%;
  median unknown −11.90 → −11.45 → −9.78; phía known phẳng (~94%).
- Hai model leak fail với đúng chẩn đoán "training data reaches beyond the
  cutoff" (vòng 1 fail sai lý do "recognises neither past nor future").
- Quét lại toàn dải epsilon trên data mới: tồn tại **dải separating rộng
  [−9.25, −5.75]** (data cũ: rỗng) — verdict không phụ thuộc một lát cắt
  may rủi.

**Kết luận: bộ dữ liệu dev hiện phân biệt được model sạch với model leak, và
mức leak tăng đơn điệu theo lượng "tương lai" model đã thấy.**

## 5. Ghi chú kỹ thuật quan trọng

- **Revision của 3 checkpoint**: các SHA pin trên branch `main` của HF chỉ có
  `pytorch_model.bin`, trong khi loader của repo chỉ đọc `model.safetensors`.
  Đã dùng SHA của branch `safetensors` sau khi **xác minh 422/422 tensor
  trùng bit-for-bit** với bản pin gốc (cả 3 model) — mọi số liệu đều là của
  đúng trọng số đã pin.
- **Chống overfit**: 3 checkpoint ChronoGPT là *thước nghiệm thu*; calibration
  chỉ dùng model sạch 2022. Không tinh chỉnh fact riêng lẻ theo điểm của model
  2023/2024.
- Fact vẫn giữ năm thật nên cùng `facts.json` build được các cutoff khác
  (`--years 2018-2024`…); phía unknown chỉ dày ở 2023–2025.

## 6. Sản phẩm bàn giao (trong `data/dev/`)

| file | nội dung |
|---|---|
| `facts.json` | 100 fact đã viết lại (nguồn sự thật duy nhất) |
| `corpus/` | probe sets build với epsilon mặc định −11.51 (chỉ để chạy thử pipeline) |
| `corpus-calibrated/` | probe sets với **epsilon đo được −8.8792** — dùng bản này để phán xử model |
| `raw_scores.json` | điểm thô 100 probe × 3 checkpoint, đủ tái tạo mọi phân tích offline |
| `CALIBRATION.md` | báo cáo kỹ thuật đầy đủ hai vòng calibration (số liệu nguyên văn) |
| `README.md` | hướng dẫn sử dụng và rebuild |
| `tools/dump_scores.py` | script tái tạo `raw_scores.json` |
