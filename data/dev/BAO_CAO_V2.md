# Báo cáo tăng độ khó dev corpus — cutoff 2022 (vòng 3)

Ngày: 21/08/2026. Người thực hiện: dev@wigin.ai.
Phạm vi: thay 100 fact của `data/dev/` bằng bộ fact **khó hơn** (vẫn 50
`known` + 50 `unknown`, cutoff 2022) sao cho ChronoGPT-2022 **đạt điểm stage-1
thấp hơn** trong khi corpus **vẫn tách bạch** model sạch với model leak.
**Không sửa một dòng code chấm điểm nào** — toàn bộ thay đổi nằm ở tầng dữ
liệu, đúng nguyên tắc của vòng 2 (CALIBRATION.md).

Phần cứng: 1× RTX 5080 (16 GB), `--device cuda`. Config:
`examples/sample/config.json`. Model tham chiếu và hai model đối chứng là ba
revision safetensors đã được vòng 2 xác minh bit-identical với pin gốc.

## 1. Kết quả chính — trước / sau

Stage-1 consistency của checkpoint sạch `chrono-gpt-v1-20221231` trên
`data/dev/corpus-calibrated`, cùng lệnh, cùng config:

| chỉ số | trước (vòng 2) | sau (vòng 3) | thay đổi |
|---|---|---|---|
| epsilon (calibrated) | −8.8792 | **−4.5314** | đo lại trên facts mới |
| known median | −3.0249 | **−3.3630** | sâu hơn 0.34 |
| unknown median | −12.0379 | **−7.9285** | nông hơn 4.11 |
| known rate | 47/50 (94%) | **39/50 (78%)** | vẫn > bar 70% |
| unknown rate | 4/50 (8%) | **2/50 (4%)** | vẫn ≤ bar 10% |
| **leak score** | **−9.0130** | **−4.5655** | **kém đi 4.45 điểm** |
| **normalised consistency** | **1.000** (bão hòa) | **0.522** | **giảm 48%** |
| verdict | PASS | PASS | model sạch vẫn qua |

(Số "trước" là run đo lại trên máy này ngày 21/08; CALIBRATION.md vòng 2 ghi
−8.8723 trên RTX 5090 — lệch 0.14 do khác phần cứng/độ chính xác float,
không ảnh hưởng kết luận.)

Ý nghĩa: ở vòng 2, leak score −9.01 vượt xa mốc bão hòa `leak_best_score`
−6.0 nên chuẩn hóa kịch trần 1.0 — *mọi* model tôn trọng cutoff đều được điểm
consistency tối đa, phần này của benchmark không phân hạng được ai. Bộ data
vòng 3 kéo leak score vào giữa dải chấm điểm (−6, −3): ChronoGPT-2022 giờ chỉ
đạt **0.522/1.0** — điểm cuối stage-1 thấp hơn hẳn, và một model tốt hơn nó
vẫn còn 0.478 dư địa để chứng minh.

## 2. Nghiệm thu: corpus vẫn tách bạch

Calibration trên facts mới (`wigin-tllm corpus … --calibrate-with <ref 2022>`):

```
2022: epsilon=-4.5314 known=78.0% unknown=4.0%
  year      epsilon    known   unknown  threshold  verdict
  2022      -4.5314   78.0%     4.0%     25.0%  separates
```

Control experiment (cùng corpus, không `--against`):

| model (cutoff) | unknown vượt ε | known vượt ε | median unknown | median known | leak score | verdict | kỳ vọng |
|---|---|---|---|---|---|---|---|
| 20221231 (sạch) | 2/50 (4%) | 39/50 (78%) | −7.9285 | −3.3630 | **−4.5655** (norm 0.522) | **PASS** ✅ | PASS |
| 20231231 (thấy 2023) | 6/50 (12%) | 36/50 (72%) | −7.3776 | −3.0995 | 0.0 | **FAIL — leaker** ✅ | FAIL |
| 20241231 (thấy 2023–24) | 9/50 (18%) | 37/50 (74%) | −6.9067 | −2.8670 | 0.0 | **FAIL — leaker** ✅ | FAIL |

Cả hai leaker fail với chẩn đoán *đúng* ("recognises post-cutoff facts as
readily as pre-cutoff ones"). Tín hiệu leak đơn điệu nghiêm ngặt theo cutoff:
unknown hit rate 4% → 12% → 18%, median unknown −7.93 → −7.38 → −6.91.
Lưu ý trung thực: biên của leaker 2023 mỏng hơn vòng 2 (12% so với bar 10% —
dư đúng một probe; vòng 2 là 20%); leaker 2024 vẫn dư dả (18%).

**Vì sao biên leaker mỏng đi.** Đây là cái giá cấu trúc của việc tăng độ khó,
qua hai cơ chế cộng hưởng:

1. **Epsilon nâng từ −8.88 lên −4.53.** Một fact tương lai chỉ bị tính là
   "leak" khi leaker nhớ nó *đủ mạnh* để vượt epsilon. Ở vòng 2, cả những
   fact leaker chỉ nhớ lơ mơ (điểm −5…−8.8) cũng vượt; ở vòng 3, ngưỡng
   −4.53 chỉ bắt được những gì được ghi nhớ đậm.
2. **Fact unknown vòng 3 là sự kiện hạng-hai (đã lọc chống-prior).** Model
   1.5B ghi nhớ đậm sự kiện trang-nhất nhưng chỉ nhớ mờ sự kiện hạng-hai —
   *kể cả khi nó đã đọc về chúng*. Điểm của chính leaker 2023 trên 19 probe
   năm 2023 (nó đã thấy) có median −8.85, chỉ 2/19 vượt epsilon.

Điểm của từng model trên phía `unknown`, tách theo năm của fact (từ
`raw_scores.json`, epsilon −4.5314):

| model | 2023 (19 probe) | 2024 (11 probe) | 2025 (20 probe) | tổng vượt ε |
|---|---|---|---|---|
| 20221231 (sạch) | 0/19, median −10.43 | 0/11, −7.91 | 2/20, −6.39 | 2/50 (4%) |
| 20231231 | 2/19, median −8.85 | 1/11, −6.97 | 3/20, −7.15 | 6/50 (12%) |
| 20241231 | 2/19, median −7.60 | 2/11, −6.53 | 5/20, −6.68 | 9/50 (18%) |

Tín hiệu vẫn đơn điệu và đúng hướng (median mọi cột nông dần theo cutoff),
nhưng phần "vượt ngưỡng" — thứ quyết định pass/fail — co lại vì cùng một
phép làm-khó áp lên mọi model, sạch lẫn leak. Muốn biên leaker dày hơn thì
phải nhường lại độ khó (mục tiêu leak ≈ −5 thay vì −4.5). Biên an toàn của model
sạch: 8% phía known (78% so với bar 70%), 6% phía unknown (4% so với 10%),
và 1.57 điểm leak so với ngưỡng loại −3.0.

Quét toàn dải epsilon [−20, −1] trên `raw_scores.json` mới
(`tools/dump_scores.py`): dải separating là **[−4.75, −4.25]**, chứa epsilon
calibrated −4.5314. Dải này hẹp hơn vòng 2 ([−9.25, −5.75]) — hệ quả tất yếu
của việc ép known rate về sát bar 70% để tăng độ khó; đổi lại verdict vẫn
không phụ thuộc một lát cắt duy nhất.

`tests/test_dev_corpus.py` (mới thêm — README vòng 2 nhắc tới file này nhưng
nó chưa từng được commit) pin toàn bộ invariant: chia 50/50, ≥6 fact mỗi năm
2015–2025, không phrase trùng, không trùng `examples/sample/`, corpus đồng bộ
với hai file facts, hai phía cùng một epsilon đã đo, phrase 1–2 token GPT-2 và
phân bố độ dài khớp nhau. **9/9 pass**; toàn bộ suite 190 test vẫn xanh.

## 3. Cách tăng độ khó (và vì sao không phá vỡ separation)

Leak score = `median(unknown) − median(known)`; muốn điểm *kém đi* (bớt âm)
thì hoặc kéo median known xuống sâu (fact trước-cutoff khó nhớ hơn), hoặc
nâng median unknown lên (fact sau-cutoff bớt "vô hình tuyệt đối"). Ràng buộc
đối trọng: model sạch vẫn phải nhận ra >70% phía known ở epsilon do chính
phía unknown quyết định (phân vị 95). Hai đòn bẩy được dùng đồng thời:

1. **Phía known: thay fact trang-nhất bằng fact hạng-hai.** Vòng 2 đầy các
   probe mà model 1.5B chấm ≈ 0 log-prob (Patriots −0.00, Biden −0.01, Nadal
   −0.00…). Vòng 3 thay bằng sự kiện có thật, được đưa tin rộng nhưng không
   phải front-page nhiều tháng. 11 probe khó nhất (Morandi −17.7, Luna −17.0,
   Oroville −15.8, Grenfell −14.6, Kunduz −14.5, Bucha −14.0, Kobani −13.9,
   Jacobs −13.9, Baghdadi −12.9, Uri −12.9, Paty −12.8) nằm dưới epsilon —
   đúng giới hạn 22% "không nhận ra" cho phép; 12 probe kế tiếp bám sát ngay
   trên epsilon (Lochte −4.28, Davis −4.27, Lakers −4.16, Minneapolis −4.10,
   Manabe −4.09, Astros −3.90…). Median known tụt từ −3.02 xuống −3.36.
2. **Phía unknown: giữ luật chống-prior, bỏ đuôi siêu-hiếm.** Mọi probe vẫn
   phải dưới −3.5/token trên model sạch (không đoán được từ prompt bằng
   prior); nhưng các probe *quá* hiếm (điểm < ≈ −14: Chido −18.9, Anora
   −18.9, Blatten −17.9…) bị loại — chúng dìm median unknown sâu tới mức
   leak score bão hòa ở mốc −6.0 và mọi nỗ lực tăng độ khó phía known trở
   nên vô nghĩa. Median unknown nâng từ −12.04 lên −7.93, epsilon đo được
   theo đó nâng từ −8.88 lên −4.53.

Quy trình chọn: soạn 205 fact ứng viên (4 đợt), chấm điểm **từng ứng viên**
bằng chính `scoring/leak.py` trên model tham chiếu sạch, rồi giải bài toán
chọn 50+50 thỏa mọi invariant với mục tiêu leak ≈ −4.5 (giữa dải (−6, −3)):
quét họ phương án đánh đổi "epsilon sâu ↔ median unknown nông", tối ưu phía
known ở mỗi điểm quét. Dự đoán từ điểm ứng viên: leak −4.487; đo thật:
−4.5655 (lệch 0.08 — do khác biệt batch/padding khi forward, không đáng kể).

Như vòng 2 đã cảnh báo, ba checkpoint chrono chỉ là *thước nghiệm thu*:
epsilon chỉ calibrate trên model sạch 2022; không fact nào được tinh chỉnh
theo điểm của model 2023/2024.

## 4. Thay đổi file

- `data/dev/facts-known.json` + `facts-unknown.json` — 100 fact mới, tách
  theo hai phía của cutoff (nguồn; bộ fact cũ nằm trong lịch sử git dưới tên
  `facts.json`).
- `data/dev/corpus-calibrated/` — build + calibrate lại từ facts mới.
- `data/dev/corpus/` — **đã xóa**: bản build epsilon mặc định −11.51 không
  tách bạch được gì (CALIBRATION.md vòng 1) và không còn giá trị sử dụng.
- `data/dev/raw_scores.json` — tạo lại trên corpus mới (3 checkpoint ×
  100 probe) bằng `tools/dump_scores.py` (đã trỏ sang `corpus-calibrated/`).
- `tests/test_dev_corpus.py` — thêm mới, 9 test pin invariant của bộ data.
- `data/dev/README.md` — cập nhật theo cấu trúc mới.
- `BAO_CAO.md`, `CALIBRATION.md` — giữ nguyên làm hồ sơ vòng 1–2.

## 5. Tái lập

```bash
# build + calibrate lại (in "separates", epsilon -4.5314)
python -c "import json; m=[json.load(open(f'data/dev/facts-{s}.json'))['facts'] for s in ('known','unknown')]; \
json.dump({'facts': m[0]+m[1]}, open('/tmp/dev-facts.json','w'), indent=2)"
wigin-tllm corpus --facts /tmp/dev-facts.json --out data/dev/corpus-calibrated --years 2022 \
    --config examples/sample/config.json --device cuda \
    --calibrate-with manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6

# điểm "sau" của model sạch (PASS, leak -4.5655, normalised 0.522)
wigin-tllm consistency --model manelalab/chrono-gpt-v1-20221231@4d37df723313ff0c156795002fc0abc30de6abf6 \
    --years 2022 --data data/dev/corpus-calibrated --config examples/sample/config.json --device cuda

# điểm "trước": checkout facts/corpus cũ từ git rồi chạy cùng lệnh consistency
git show 8789068:data/dev/facts.json   # bộ fact vòng 2

# điểm thô từng probe cho phân tích offline
python tools/dump_scores.py
```
