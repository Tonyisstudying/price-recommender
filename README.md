# SEA E-commerce Intelligent Pricing Engine

End-to-end pricing system combining:
- marketplace data engineering
- comparable-product retrieval
- competitor price distributions
- ML market-price estimation
- demand estimation
- gross-margin constraints
- profit-aware price optimization
- FastAPI
- Streamlit
- Docker
- tests

> The included marketplace data is **synthetic/mock data** for development. Replace it with a licensed marketplace export before presenting results as real-world findings.

## Architecture

```text
Marketplace Data
      |
      v
Data Validation + Cleaning
      |
      +--------------------------+
      |                          |
      v                          v
Comparable Product          Feature Engineering
Retrieval (TF-IDF)                |
      |                           v
      v                     Market Price Model
Competitor Distribution           |
      |                           v
      +---------------------- Demand Model
                              |
                              v
                         Price Optimizer
                              |
                    +---------+---------+
                    |                   |
              Margin constraint   Market constraint
                    |                   |
                    +---------+---------+
                              |
                              v
                     Recommended Price
                              |
                  +-----------+-----------+
                  |                       |
                FastAPI              Streamlit
```

## Quick start

### 1. Create environment

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\\Scripts\\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Generate development data

```bash
python generate_mock_data.py
```

### 4. Train

```bash
python train.py
```

### 5. Start API

```bash
uvicorn api.main:app --reload
```

Docs: http://127.0.0.1:8000/docs

### 6. Start dashboard

```bash
streamlit run app/streamlit_app.py
```

Dashboard: http://127.0.0.1:8501

### 7. Tests

```bash
pytest -q
```

## Example API request

```json
{
  "country": "Vietnam",
  "marketplace": "Shopee",
  "category": "Accessories",
  "product_name": "Logitech M331 Wireless Mouse",
  "brand": "Logitech",
  "cost": 8.0,
  "target_margin": 0.30,
  "strategy": "balanced"
}
```

## Pricing logic

Minimum gross-margin price:

```text
P_min = cost / (1 - target_margin)
```

Strategy anchor:
- competitive -> competitor P25
- balanced -> competitor median
- premium -> competitor P75

Baseline recommendation:

```text
market_anchor = strategy percentile
blended_market = 0.60 * market_anchor + 0.40 * ML_market_price
recommended = max(minimum_profitable_price, blended_market)
```

Then a demand-aware optimizer evaluates candidate prices with:

```text
Expected Profit(p) = (p - cost) * predicted_demand(p)
```

The target margin is a hard floor.

## Real marketplace data

The raw adapter expects at least:

```text
sku_name
brand
category
price_usd
month
country
marketplace
```

Preferably also:

```text
merchant
discount
sold
gmv_usd
rating
reviews
```

Do not collect or scrape marketplace data in a way that violates the provider's terms. Use a licensed/exported dataset.

## Limitations

- Listed price is not necessarily transaction price.
- Product variants can differ despite similar names.
- Observational data does not automatically identify causal price elasticity.
- Demand is affected by advertising, rank, seller reputation, logistics and promotions.
- Production systems should include human override and monitoring.


### Dependency fallback

Parquet and CatBoost are optional accelerators. If `pyarrow` is unavailable, the pipeline uses CSV automatically. If CatBoost is unavailable, the market-price model falls back to RandomForestRegressor.
