I want to build a AI/ML model that recommend you with a reasonable price for accessories, groceries. The dataset should be around 3 to 6 months ago (e.g Lazada and Shoppe dataset) and the overall workflow should be: input the user's price ->Your system calculates the competitive price range and combines it with your required margin.-> output the recommended selling price This is important because I want pricing based on **cost + target gross margin + competitor price distribution**. The dataset should cover the wide rage for South Asea countries.  Go to code with explanations. Project architecture

raw data: src/raw/pages-new.csv
I recommend:

smart-price-recommender/
│
├── data/
│   ├── raw/
│   │   └── marketplace_data.csv
│   │
│   ├── processed/
│   │   ├── cleaned_products.parquet
│   │   └── competitor_features.parquet
│   │
│   └── external/
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_competitor_analysis.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── competitor.py
│   ├── features.py
│   ├── train.py
│   ├── pricing.py
│   └── utils.py
│
├── models/
│   └── price_model.joblib
│
├── app/
│   └── streamlit_app.py
│
├── requirements.txt
├── config.yaml
└── README.md