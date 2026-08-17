# Legal cross-attention MTL

Pipeline dự đoán đa nhiệm từ dữ liệu phán quyết pháp lý Nhật Bản:

- mức chấp nhận claim của nguyên đơn;
- mức chấp nhận claim của bị đơn;
- kết quả tort ở cấp hồ sơ.

Model dùng một ModernBERT encoder dùng chung, claim-to-fact cross-attention và
hai chiều tương tác nguyên đơn/bị đơn. Gold rationale label không được truyền
vào tort head.

## Cài đặt và chạy

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

.venv/bin/python -m src.main smoke-test
.venv/bin/python -m src.main train --config configs/base.yaml
.venv/bin/python -m src.main train --config configs/base.yaml \
  --resume outputs/modernbert_legal_mtl/last
.venv/bin/python -m src.main evaluate \
  --checkpoint outputs/modernbert_legal_mtl/best --split test
```

Lệnh `train` tự động đánh giá tập dev sau mỗi epoch và, sau khi huấn luyện
xong, đánh giá tập test bằng checkpoint `best`. Kết quả test được lưu tại
`best/test_metrics.json` và `best/test_predictions.jsonl`. Lệnh `evaluate` vẫn
có thể dùng để đánh giá lại một checkpoint hoặc một split cụ thể.

Các lệnh trên chạy từ thư mục `mtl`. Mặc định W&B bị tắt. Để bật, đặt
`logging.wandb.enabled: true`; API key được đọc từ environment hoặc `.env` và
không được ghi vào output. Mỗi run lưu config đã resolve, history, best/last
checkpoint, metrics và prediction JSONL.

## Cấu hình

`configs/base.yaml` là nguồn sự thật cho toàn bộ hyperparameter model, dữ liệu,
loss, optimizer, scheduler, evaluation, logging và runtime. `encoder_hidden_size`
được đọc từ pretrained encoder rồi ghi vào `resolved_config.yaml`.

## Kiểm thử

```bash
PYTHONPATH="$PWD" .venv/bin/pytest -q
```
