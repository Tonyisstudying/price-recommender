Place incoming marketplace files here.

Suggested layout:

```text
external/
├── lazada/
├── shopee/
├── tiktok/
├── tokopedia/
└── generic/
```

Examples:

```bash
python update_data.py --input data/external/lazada/pages-new.csv --source lazada --snapshot-date 2026-09-16
python update_data.py --input data/external/shopee/file.csv --source shopee --snapshot-date 2026-09-16
python update_data.py --input data/external/generic/marketplace_data.csv --source generic --snapshot-date 2026-09-16
```
