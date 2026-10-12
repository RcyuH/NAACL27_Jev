# Các phân tích bổ sung cho paper — chạy offline trên predictions có sẵn

Không cần GPU, model server, internet, inference mới, `prepare`, `analyze`, hoặc
`robustness` cũ. Không cần cài thêm thư viện nếu môi trường đã chạy được pipeline
cũ. Phần mới chỉ dùng NumPy, SciPy, pandas, PyArrow, matplotlib, PyYAML và tqdm.

**Không sử dụng lại confidence intervals của continuous spline cũ.**
Đây là follow-up exploratory sau khi đã xem kết quả, không phải preregistration.

## 1. Cập nhật code, giữ nguyên đường dẫn trên server

Copy code mới vào đúng checkout trên server. Không ghi đè config đường dẫn đang
dùng bằng config mặc định trong ZIP nếu bạn đã sửa nó. Phần bổ sung gồm:

- `src/jevcal/strengthen.py`, `strengthen_stats.py`, `strengthen_reporting.py`;
- `src/jevcal/cli.py` cập nhật;
- `configs/strengthen.yaml` (settings riêng, không chứa đường dẫn);
- `scripts/run_strengthening_full.sh`;
- tests và tài liệu này.

Trong **config reproduce.yaml đang dùng trên server**, kiểm tra
`robustness.prediction_sets`. Mỗi `path` phải trỏ đến file **per-example
predictions**, không phải CSV bảng tổng hợp trong `results/`:

```yaml
robustness:
  prediction_sets:
    - name: openjev_criteria_rich
      model: OpenJev-27B
      prompt_set: criteria_rich
      path: artifacts/predictions/civil_comments.parquet
    - name: openjev_minimal
      model: OpenJev-27B
      prompt_set: minimal
      path: artifacts/predictions/runs/openjev_minimal/civil_comments.parquet
    - name: openjev_dataset_aligned
      model: OpenJev-27B
      prompt_set: dataset_aligned
      path: artifacts/predictions/runs/openjev_dataset_aligned/civil_comments.parquet
    - name: onejev_criteria_rich
      model: OneJev-27B
      prompt_set: criteria_rich
      path: artifacts/predictions/runs/onejev_criteria_rich/civil_comments.parquet
```

Có thể dùng đường dẫn tuyệt đối. Đường dẫn tương đối tính từ thư mục chạy lệnh.
Code kiểm tra đủ IDs, nhãn và text units giữa các run; không tự lấy intersection,
không tự bỏ run thiếu. Nếu chỉ có bảng CSV tổng hợp, chưa thể chạy các phép kiểm
tra mới; cần lấy lại các `.parquet` predictions đã tạo khi inference.

## 2. Lệnh FULL cho tất cả OpenJev/OneJev runs

Trong môi trường `.venv_jev` đang có đầy đủ dependencies:

```bash
cd /workspace/storage-shared/nlp/huypq51/projects/Jev_brainrot

bash scripts/run_strengthening_full.sh \
  configs/reproduce.yaml \
  results/strengthening_v1 \
  8
```

Nếu checkout của bạn tên `Jev_brainrot-main`, thay dòng `cd` tương ứng.
`8` là **CPU worker processes**, không phải GPUs. Có thể đổi thành 4 hoặc 16
theo số CPU/RAM được cấp. Script giới hạn mỗi worker ở một BLAS thread để tránh
oversubscription; không cần dừng vLLM trừ khi muốn giải phóng RAM hệ thống.

Script dùng `python -m jevcal.cli` với `PYTHONPATH` của **checkout hiện tại**,
tránh lỗi editable install còn trỏ vào checkout `Jev_brainrot-main` cũ.
Không cần `pip install -e .` lại. `PYTHON_BIN` mặc định là `python` của môi
trường đang activate; có thể đặt biến này thành đường dẫn Python mong muốn.

Đây là chạy đầy đủ: **2.000 bootstrap replicates và 2.000 null simulations**,
ba cạnh chính/đối chứng tại bốn routing rates, đủ 12 directed edges tại hai rates,
trên cả bốn prediction sets. Có 60 jobs mỗi run, tổng cộng 240 jobs theo cấu hình
mặc định. Không có `--quick` trong lệnh trên.

Mỗi job hoàn tất sẽ có checkpoint. Chạy lại **đúng lệnh** để tiếp tục, các jobs
đã xong và đúng checksum được bỏ qua. Nếu bị kill -9/mất điện, có thể còn
`RUNNING.lock`: đọc PID/host trong file, xác nhận process cũ đã dừng rồi mới xóa
riêng lock đó. Không xóa checkpoints. Khi đổi code, input, seed hoặc settings,
dùng output mới như `results/strengthening_v2`; code từ chối trộn hai phiên bản.

## 3. Nếu muốn chia thành từng giai đoạn (vẫn FULL)

Không cần chạy phần này nếu đã dùng script full ở trên. Các lệnh sau dùng cùng
output và có thể resume. Dùng nguyên toàn bộ prediction-set list để fingerprint
nhất quán. Đừng chạy hai lệnh đồng thời vào cùng output.

```bash
cd /workspace/storage-shared/nlp/huypq51/projects/Jev_brainrot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

python -m jevcal.cli --config configs/reproduce.yaml strengthen \
  --settings configs/strengthen.yaml --output-dir results/strengthening_v1 \
  --workers 8 --stage local

python -m jevcal.cli --config configs/reproduce.yaml strengthen \
  --settings configs/strengthen.yaml --output-dir results/strengthening_v1 \
  --workers 8 --stage null

python -m jevcal.cli --config configs/reproduce.yaml strengthen \
  --settings configs/strengthen.yaml --output-dir results/strengthening_v1 \
  --workers 8 --stage recalibration

python -m jevcal.cli --config configs/reproduce.yaml strengthen \
  --settings configs/strengthen.yaml --output-dir results/strengthening_v1 \
  --workers 8 --stage matrix
```

Nếu chủ động muốn chỉ chạy một run, thêm `--prediction-set openjev_criteria_rich`
và dùng **output riêng**, chẳng hạn `results/strengthening_openjev_only`.
Không cần làm vậy cho kế hoạch full.

## 4. Chính xác từng analysis đang kiểm tra gì?

### A. Local score balancing (`local`)

Gaussian local standardization trên downstream probability, tại bandwidth
`0.005 / 0.01 / 0.02`. Đây là độ rộng theo đơn vị xác suất, không phải quantile.
Score được đưa về grid 1.001 điểm để tính nhanh; rounding tối đa 0.0005.
Kernel cắt ở 4 bandwidths. Trong từng group:

`ESS(p) = (sum_i K_h(P_i-p))² / sum_i K_h(P_i-p)²`.

Chỉ dùng grid points có ESS >= 50 cho cả routed và non-routed. Tính local
mean score và mean label ở hai nhóm, rồi standardize cả hai về cùng phân bố
score của các routed examples được giữ lại.

- `native`: vùng support riêng mỗi bandwidth.
- `common_grid`: giao support của cả ba bandwidths; cùng routed target
  population giữa ba estimates trong một sample.

Xuất signed-error gap, mean signed/absolute local score imbalance, prevalence,
support và positive counts. Bootstrap lại text clusters, routing, support và
local regressions; shared support cũng được dựng lại trong mỗi replicate.
Không dùng nearest-neighbor bootstrap hay covariance spline cũ.

**Giới hạn:** kernel smoothing và grid vẫn là approximation; balance tốt hơn
không tự chứng minh exact conditional independence/dependence. ESS đếm copies
trong bootstrap, không phải số independent text clusters. CI được gắn nhãn
unsupported nếu dưới 95% bootstrap draws hợp lệ; không âm thầm bỏ các draw lỗi.

### B. Fitted invariant-score null (`null`)

Chia 50/50 theo exact-text groups, cùng split giữa các models/edges. Routing
threshold và tie cutoff được fit trên calibration half, rồi giữ cố định khi
đánh giá evaluation half. Tie key là hash ID với seed, không dùng nhãn.

Fit hai pooled outcome models `m(P)=Pr(Y=1|P)`: logistic-on-logit (Platt) và
penalized cubic B-spline-on-logit. Không có upstream score/route trong m.
Chạy 5/10/20/40/80 bins. Mỗi simulation:

1. Bootstrap text clusters của calibration half và refit m.
2. Giữ nguyên evaluation scores, routing và bins.
3. Sinh nhãn độc lập `Y* ~ Bernoulli(m(P))`.
4. Tính lại signed-error gaps trên các nhãn giả.

Xuất gap quan sát, fitted expected null gap, observed-minus-fitted-null và
predictive bands. Expected null gap tính chính xác từ coefficient vector và
`P-m(P)`, không phải lấy trung bình Monte Carlo.

**Không gọi predictive band là CI của hiệu ứng; không gọi tail fraction là
p-value exact và không áp BH cho nó.** Đây là model-based diagnostic phụ thuộc
chất lượng m và giả định independent Bernoulli labels. Nuisance bootstrap không
giải quyết mọi misspecification hoặc phụ thuộc nhãn giữa texts. Kết quả phải
được đọc cùng `null_fit_diagnostics.csv` và so sánh cả hai null models.

### C. Held-out recalibration (`recalibration`)

Chỉ fit trên calibration half; evaluation labels không được dùng để chọn
calibrator, threshold, knots hoặc hyperparameters. Dùng bốn methods:

- `raw`: xác suất nguyên bản.
- `platt`: logistic regression trên clipped logit(P).
- `spline`: pooled penalized cubic B-spline trên clipped logit(P).
- `spline_route`: cùng spline + route intercept và route × logit(P).

Spline knots cố định cách đều trên scaled logit [-2,2] (logit [-12,12]), degree
3; second-difference penalty có ridge=1. Likelihood được aggregate trên logit
grid 2.049 điểm, nhưng prediction dùng score thật. Platt có slope penalty yếu,
và mọi fit có ridge rất nhỏ để tránh singularity. Đây là lựa chọn đã freeze,
không tối ưu hyperparameters trên evaluation set.

Báo prevalence, mean prediction, signed error, Brier, log loss và fixed-width
15-bin ECE cho all/routed/non-routed. Xuất paired bootstrap differences giữa
platt-raw, spline-raw và spline_route-spline. Delta Brier/log loss âm có lợi
cho candidate; signed-error delta âm không tự động có nghĩa calibration tốt hơn.

Bootstrap hai phía calibration/evaluation độc lập theo text clusters; mỗi lần
refit threshold và toàn bộ calibrators. CI bao gồm fitting uncertainty nhưng
**conditional on split membership đã cố định**, không phải variation qua nhiều
random splits. Không gọi ECE là kiểm định calibration chính xác. Reliability
plots dùng shared raw-score quantile strata học từ calibration half; chỉ là
descriptive points, không vẽ confidence bands giả cho curves.

Fit failures và số positive examples được xuất riêng. Nếu bootstrap có <95%
draws hợp lệ, CI bị bỏ và đánh cờ; không lấy những fit thành công làm bằng chứng
thay cho một phân tích đã hỏng.

### D. All-pairs + mechanism diagnostics (`matrix`)

Đủ 12 directed pairs của toxicity/identity attack/obscene/insult tại 10% và 5%,
không lựa chọn cạnh dựa vào kết quả. Dùng estimator 10 pooled quantile bins của
draft, nhưng bootstrap theo text clusters. Mỗi run và mỗi metric có BH family
24 cells; unsupported cells giữ p=1. Không công bố q khi family còn thiếu jobs.

`edge_bin_diagnostics.csv` có n/positive_n, mean score, prevalence, residual,
score min/max cho từng group/bin. Kiểm tra decomposition:

`signed-error gap = mean-score gap - outcome-rate gap`.

`qualitative_sample_ids.csv` lấy ngẫu nhiên tối đa ba IDs mỗi supported
bin/group, không dựa trên việc ví dụ có ủng hộ câu chuyện hay không. Không xuất
comment text để tránh vô tình đóng gói dữ liệu nhạy cảm. Dùng ID join với parquet
local để xem text; xem xét thủ công trước khi đưa ví dụ vào paper.

## 5. Kết quả và lệnh vẽ lại

```text
results/strengthening_v1/
  manifest.json                       # status, protocol, input/code hashes, versions
  REPORT.md                           # cách đọc + quality checklist
  splits/*.parquet                    # IDs và split assignments
  tables/*.csv                        # kết quả đầy đủ
  tables/*.parquet                    # bản binary cùng nội dung
  figures/<prediction_set>/*.pdf       # vector figures
  figures/<prediction_set>/*.png       # previews
  publication_tables/*.tex            # appendix longtables có màu
  checkpoints/<set>/<job>/draws.npz    # bootstrap/predictive draws; thứ tự theo table
```

`status: complete` nghĩa đủ jobs của protocol đã chạy, **không có nghĩa mọi
comparison đều supported/significant**. Xem cờ CI và diagnostic tables.
`quick_mode` phải là `false` cho lần chạy dùng trong paper.

Vẽ lại bảng/hình từ final results, không chạy thống kê lại:

```bash
cd /workspace/storage-shared/nlp/huypq51/projects/Jev_brainrot
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m jevcal.cli --config configs/reproduce.yaml strengthen-report \
  --output-dir results/strengthening_v1
```

Các `.tex` là appendix longtables; dùng `booktabs`, `longtable`, `xcolor[table]`.
Trong tài liệu ACL hai cột, đưa longtable vào appendix `\onecolumn`, hoặc chọn
những hàng chính để chuyển thành `table`/`table*`. Không input nguyên longtable
vào main hai cột rồi ép thu nhỏ font.

## 6. Gửi gì lại để cùng phân tích?

Sau khi thấy manifest complete, đóng gói các aggregate results + figures + split
IDs. Không cần gửi model, raw text hay predictions hàng GB:

```bash
cd /workspace/storage-shared/nlp/huypq51/projects/Jev_brainrot
tar -czf results/strengthening_v1_review.tar.gz \
  -C results/strengthening_v1 \
  manifest.json REPORT.md tables figures publication_tables splits
```

Giữ `checkpoints/` trên server để resume và truy xuất draws khi cần; không bắt
buộc gửi lần đầu. Chúng ta sẽ kiểm tra support, fit failures, imbalance, null
sensitivity và held-out losses trước khi quyết định claim mới cho paper.

## 7. Hạn chế đã biết / không làm trong vòng này

- Không thêm model/dataset, không đo causal routing effect hoặc deployment costs.
- Không đổi primary results cũ hoặc sửa paper trước khi xem follow-up results.
- Không coi nhiều rates/prompts trên cùng dữ liệu là independent replications.
- Không tự động chọn bandwidth/calibrator theo p-value hoặc magnitude.
- Không giả định post-hoc exploratory p-values là confirmatory bằng chứng.
- Không có exact distribution-free conditional-independence test.
- Không chạy spline uncertainty cũ bằng lệnh full mới.
