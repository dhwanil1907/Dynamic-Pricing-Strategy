# 🏷️ **Dynamic Pricing Strategy**

A complete machine-learning pipeline for predicting and optimizing product prices. This project includes data preparation, feature engineering, baseline & advanced models, evaluation, and a Streamlit app that allows interactive exploration of pricing predictions.

---

## 🚀 **Features**

* **Data Cleaning & Processing**
  Structured workflow for preparing raw product and seller data.

* **Feature Engineering**
  Category-level stats, seller metrics, price ratios, time-based features, and more.

* **Modeling Pipeline**

  * Linear Regression
  * Ridge Regression
  * Random Forest
  * XGBoost
    Each model is evaluated using RMSE, MAE, and cross-model comparison.

* **Streamlit App (`app.py`)**

  * Upload or preview processed data
  * Generate predictions using the trained model
  * Visualize price distributions and feature relationships

* **Exploratory Notebooks**

  * `pricing_eda.ipynb` — Exploratory data analysis
  * `pricing_model.ipynb` — Model training & evaluation
  * `pricing_simulation.ipynb` — Price sensitivity simulation

---

## 📁 **Project Structure**

```
Dynamic-Pricing-Strategy/
│
├── app.py                   # Streamlit interface
├── README.md                # Project documentation
├── requirements.txt         # Dependencies
│
├── data/
│   ├── raw/                 # Optional: raw datasets
│   └── processed/           # Cleaned datasets for modeling
│
├── notebooks/
│   ├── pricing_eda.ipynb
│   ├── pricing_model.ipynb
│   └── pricing_simulation.ipynb
│
└── models/
    └── final_model.pkl      # Saved trained model (optional)
```

---

## 🛠️ **Installation**

Clone the repo:

```bash
git clone https://github.com/dhwanil1907/Dynamic-Pricing-Strategy.git
cd Dynamic-Pricing-Strategy
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ **Run the Streamlit App**

```bash
streamlit run app.py
```

This opens a browser UI where you can explore the data, run predictions, and visualize model outputs.

---

## 📊 **Modeling Pipeline Overview**

1. **Load cleaned data**
2. **Engineer features**
3. **Build dataset (X, y)**
4. **Train/test split**
5. **Train baseline + tree-based models**
6. **Evaluate using RMSE, MAE**
7. **Select best model**
8. **Export model to `final_model.pkl`**

---

## 🤝 Contributing

Pull requests are welcome!
For major changes, please open an issue to discuss the proposal first.

---

## 📜 License

This project is for educational and research purposes.
