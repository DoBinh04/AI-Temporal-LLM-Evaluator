# Báo cáo calibration — dev corpus, cutoff 2022

**Vòng 2 (cuối file) đã viết lại facts và corpus hiện tách bạch được
(separates)**: epsilon đo được −8.8792, model tham chiếu sạch 2022 PASS còn
hai checkpoint 2023/2024 FAIL đúng dạng leaker. Vòng 1 ghi lại thất bại dẫn
tới việc viết lại — các số liệu per-probe của vòng 1 thuộc về `facts.json`
*cũ* (còn trong lịch sử git) và một bản `raw_scores.json` đã bị thay thế.

Ngày: 19/08/2026. Phần cứng: 1× RTX 5090 (32 GB), `--device cuda`.
Config: `examples/sample/config.json` (`known_threshold` 0.70,
`unknown_threshold` 0.10, `probe_threshold` 0.25, `calibration_margin` 0.5).

## Thay thế revision (đã xác minh trọng số trùng khớp)

Các revision được pin cho thí nghiệm này là tip của branch `main` trên ba repo
`manelalab/chrono-gpt-v1-*`, vốn chỉ chứa `pytorch_model.bin`. Loader
`wigin_tllm/models/chronogpt.py` **chỉ đọc** `model.safetensors`, nên chính
xác các revision đó không load được (`FileNotFoundError: model.safetensors
not found`, exit 1 — đây là lần chạy calibrate đầu tiên). Mỗi repo có branch
`safetensors` chứa trọng số đã convert. Trước khi thay thế, **toàn bộ 422
tensor của từng cặp revision đã được so sánh bit-for-bit** (`torch.equal`,
kèm so sánh config.json và tập key sau khi bỏ prefix `_orig_mod.` của file
`.bin`):

| model | pin gốc (main, .bin) | dùng thực tế (branch safetensors) | tensor trùng |
|---|---|---|---|
| chrono-gpt-v1-20221231 | `993711fdf078740fe1c837a3687528e2173443d2` | `4d37df723313ff0c156795002fc0abc30de6abf6` | 422/422 |
| chrono-gpt-v1-20231231 | `8ac22e54d37df8bb8037622680414118239fbe53` | `771747bd61cd50b8d99fe381a41eb25c86b80f3e` | 422/422 |
| chrono-gpt-v1-20241231 | `1d9f1b8ff50bb45a6fe1402280e617af4c2d805c` | `26e0653a22c5d0b47845c64c2a45d7acde61222d` | 422/422 |

Do đó mọi con số dưới đây phản ánh chính xác trọng số đã pin.

# Vòng 1 — data cũ (đã bị thay thế)

## 1. Token audit (GPT-2 BPE, encode `" " + phrase` như `scoring/leak.py`)

Số token của phrase trên 100 probe của corpus cũ:

| phía | phân bố {token: số probe} | min | max | median | mean |
|---|---|---|---|---|---|
| known (50) | {1: 7, 2: 19, 3: 13, 4: 5, 5: 4, 6: 2} | 1 | 6 | 2 | 2.72 |
| unknown (50) | {1: 6, 2: 22, 3: 14, 4: 2, 5: 4, 6: 1, 7: 1} | 1 | 7 | 2 | 2.66 |

Epsilon chặn trên **tổng** log-prob, nên xác suất trung bình mỗi token mà một
phrase T token phải đạt để vượt mức mặc định −11.51 là `exp(−11.51/T)`:

| T | exp(−11.51/T) | known | unknown |
|---|---|---|---|
| 1 | 0.000010 | 7 | 6 |
| 2 | 0.003167 | 19 | 22 |
| 3 | 0.021565 | 13 | 14 |
| 4 | 0.056275 | 5 | 2 |
| 5 | 0.100059 | 4 | 4 |
| 6 | 0.146852 | 2 | 1 |
| 7 | 0.193150 | 0 | 1 |

**Kết luận.** Yêu cầu per-token trải **hơn 4 bậc độ lớn** (10⁻⁵ cho phrase 1
token như "Nepal"/"Gemini" so với ≈0.19 mỗi token cho "Operation Al-Aqsa
Flood" 7 token). Với median 2 token ở cả hai phía, một epsilon vô hướng
−11.51 phân loại probe **chủ yếu theo độ dài phrase, không theo kiến thức**:
phrase ngắn gần như chắc chắn vượt, phrase dài gần như không thể vượt.

## 2. Calibration vòng 1

Lệnh (revision đã thay thế như ghi ở trên; revision pin gốc lỗi load, exit 1):

```
wigin-tllm corpus --facts data/dev/facts.json --out /tmp/corpus-cal-2022 --years 2022 \
  --config examples/sample/config.json --device cuda \
  --calibrate-with "manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6"
```

Output nguyên văn:

```
09:55:54 INFO    | 2022: epsilon=-3.8248 known=22.0% unknown=4.0%
09:55:54 INFO    | Wrote 1 years of probes to /tmp/corpus-cal-2022

=== Corpus calibration ===

--------------------------------------------------------------------------
  year      epsilon    known   unknown  threshold  verdict
  2022      -3.8248   22.0%     4.0%     25.0%  DOES NOT SEPARATE
--------------------------------------------------------------------------
This corpus cannot tell a clean model from a leaking one. The reference model must
recognise its own era and not the future; if it does neither, the probes are too hard.
```

- epsilon đo được: **−3.824838** (phân vị 95 của điểm unknown của reference;
  target unknown rate = 0.10 × 0.5 = 5%, rơi vào 4% với 50 probe)
- known_rate: **22.0%** — thấp hơn xa mức `known_threshold` 70%
- unknown_rate: **4.0%**
- margin (khoảng cách nhỏ nhất của một rate tới bar của nó): **6.0%**
- separates: **False** (`DOES NOT SEPARATE`), exit code: **1**

Ý nghĩa: ở bất kỳ epsilon nào đủ chặt để "tương lai" vô hình, reference chỉ
nhận ra ~22% thời đại của chính nó — probe quá khó cho ChronoGPT 1.5B.

## 3. Epsilon cũ trên corpus chưa calibrate (kiểm chứng dự đoán)

Lệnh: `wigin-tllm consistency --model <ref 2022> --years 2022 --data data/dev/corpus …`

```
  2022  [FAIL]  score +0.0000
      known    median   -5.7908   43/50 above epsilon -11.51   (must recognise)
      unknown  median  -12.1079   21/50 above epsilon -11.51   (must not recognise)
      recognises post-cutoff facts as readily as pre-cutoff ones — the training data reaches beyond the cutoff
```

Exit code 1. Unknown **21/50 (42%)** vượt ngưỡng so với threshold 10%; known
43/50 (86%). **Dự đoán được xác nhận**: với −11.51 mặc định, model 2022 sạch
bị gắn nhãn leaker — 21 probe hậu-cutoff (gần như toàn phrase ngắn) vượt cái
ngưỡng bị chi phối bởi độ dài.

## 4. Control experiment vòng 1 (corpus đã calibrate, không `--against`)

| model (cutoff) | unknown vượt ε | known vượt ε | median unknown | median known | leak score | verdict | kỳ vọng |
|---|---|---|---|---|---|---|---|
| 20221231 (sạch) | 2/50 (4%) | 11/50 (22%) | −12.1079 | −5.7908 | 0.0 | **FAIL** | PASS |
| 20231231 | 3/50 (6%) | 10/50 (20%) | −11.1694 | −5.8512 | 0.0 | **FAIL** | FAIL |
| 20241231 | 5/50 (10%) | 10/50 (20%) | −8.2576 | −5.7439 | 0.0 | **FAIL** | FAIL |

Cả ba fail **ở phía known** (20–22% so với bar 70%) như model "rỗng" — kể cả
leaker 2024 cũng không bị flag là leaker. Median cho thấy tín hiệu leak đơn
điệu có tồn tại (−12.11 → −11.17 → −8.26) nhưng cơ chế pass/fail không tách
được trên corpus này. Quét toàn dải epsilon [−20, −1] từ `raw_scores.json`
(bản cũ): **không tồn tại epsilon vô hướng nào thoả known > 70% và unknown ≤
10% cùng lúc** (điểm gần nhất: −8.50 với known 66%, unknown 24%).

## Ghi chú tái lập vòng 1

- CLI tải lại ~7.4 GB mỗi lần chạy; mỗi lượt stage-1 mất ~3.5 phút trên
  RTX 5090.
- `wigin-tllm` được cài editable vào `/venv/main`
  (`uv pip install -e . tiktoken`).
- Điểm thô của reference 2022 trùng nhau giữa §3 và §4 (median khớp tới 4 chữ
  số thập phân); chỉ epsilon khác nhau giữa hai corpus, đúng như kỳ vọng.

---

# Vòng 2 — viết lại facts, corpus tách bạch được

Chẩn đoán vòng 1 (epsilon bị chi phối bởi độ dài + probe unknown đoán được +
probe known quá khó) được xử lý **hoàn toàn ở tầng dữ liệu** — không đổi một
dòng code chấm điểm nào:

1. **Mọi phrase hiện là 1–2 token GPT-2 BPE** (trước là 1–7), phân bố độ dài
   khớp giữa hai phía, để epsilon vô hướng so sánh cùng loại với cùng loại.
2. **Probe unknown được pre-screen bằng model tham chiếu sạch 2022**: probe
   nào model sạch chấm trên ≈ −3.5/token là đoán được từ prompt bằng prior
   (ví dụ "Mandalay" từ *động đất Myanmar*, "Taylor Swift" từ *nhạc sĩ của
   năm*, "Machado" từ *Maria Corina*) và bị thay bằng probe chống prior
   (Raygun, Pop Mart, RedNote, Willow, …).
3. **Probe known mà reference không nhận ra bị thay** — loại hiếm (Lubitz,
   Harambe, Roma) và loại sát mép cutoff (FTX, Midjourney — đều thuộc những
   tuần cuối 2022, trước khi dữ liệu train kịp cập nhật).
4. Toàn bộ invariant trong `tests/test_dev_corpus.py` được giữ (chia 50/50,
   ≥6 fact mỗi năm, không phrase trùng, không trùng `examples/sample/`, phân
   bố độ dài từ khớp nhau): **11/11 pass**.

## Calibration (cùng lệnh như §2, facts mới)

```
11:42:26 INFO    | 2022: epsilon=-8.8792 known=94.0% unknown=4.0%

  year      epsilon    known   unknown  threshold  verdict
  2022      -8.8792   94.0%     4.0%     25.0%  separates

Calibrated. Tightest margin 6.0% (year 2022) — comfortably clear of the threshold.
```

Exit code **0**. Đây chính là corpus được commit tại
`data/dev/corpus-calibrated/`.

## Control experiment (corpus đã calibrate, không `--against`)

| model (cutoff) | unknown vượt ε | known vượt ε | median unknown | median known | leak score | verdict | kỳ vọng |
|---|---|---|---|---|---|---|---|
| 20221231 (sạch) | 2/50 (4%) | 47/50 (94%) | −11.8972 | −3.0249 | **−8.8723** (normalised 1.0) | **PASS** ✅ | PASS |
| 20231231 (thấy 2023) | 10/50 (20%) | 47/50 (94%) | −11.4498 | −3.1645 | 0.0 | **FAIL — leaker** ✅ | FAIL |
| 20241231 (thấy 2023–24) | 19/50 (38%) | 45/50 (90%) | −9.7817 | −3.2029 | 0.0 | **FAIL — leaker** ✅ | FAIL |

Cả hai leaker fail với chẩn đoán *đúng* ("recognises post-cutoff facts as
readily as pre-cutoff ones — the training data reaches beyond the cutoff"),
không còn kiểu fail sai "recognises neither" của vòng 1. Tín hiệu leak đơn
điệu nghiêm ngặt theo cutoff — unknown hit rate 4% → 20% → 38%, median
unknown −11.90 → −11.45 → −9.78 — trong khi phía known phẳng (~94%, median
≈ −3.1).

**Nghiệm thu: bộ dữ liệu này hiện phân biệt được model sạch với model leak.**
Model sạch pass với biên 6 điểm phía unknown và 24 điểm phía known; leaker
nhẹ nhất (2023) vượt threshold unknown 10% tới hai lần.

Quét lại epsilon trên `raw_scores.json` mới: tồn tại **dải separating rộng
[−9.25, −5.75]** (vòng 1: rỗng) — epsilon calibrated −8.8792 nằm thoải mái
trong dải, verdict không phụ thuộc một lát cắt may rủi.

`data/dev/raw_scores.json` đã được tạo lại trên corpus mới để phân tích
offline. Lưu ý cho các lần sửa sau: ba checkpoint chrono là *thước nghiệm
thu* — chỉ calibrate trên model sạch 2022, và tuyệt đối không tinh chỉnh
từng fact theo điểm của model 2023/2024, nếu không corpus sẽ overfit vào
đúng ba model này.
