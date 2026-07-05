# 🛒 FMCG Multi-Country Sales Analysis & Demand Forecasting

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Latest-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-02C39A?style=for-the-badge)

**Data Science Cohort 5 Capstone Project — Everything Data**

*By Ian Waweru Kariuki | BSc. Data Science & Analytics | The Cooperative University of Kenya*

---

| 📊 1,100,000+ Records | 🌍 7 Countries | 🏷️ 102 SKUs | 📈 Best R² Score: 0.956 |
|:---:|:---:|:---:|:---:|

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Key Findings](#-key-findings)
- [Forecasting Results](#-forecasting-results)
- [Recommendations](#-recommendations)
- [Tech Stack](#-tech-stack)
- [How To Run](#-how-to-run)
- [Project Deliverables](#-project-deliverables)
- [Author](#-author)

---

## 🔍 Project Overview

This project is an end-to-end data science solution for a **multinational FMCG (Fast Moving Consumer Goods) manufacturer** operating across 7 European countries. Using over **1.1 million rows of daily sales transaction data** spanning 3 full years (2021–2023), the project delivers:

- 📊 **Deep market intelligence** — country-level sales performance, product segmentation and demand driver analysis
- 🔮 **XGBoost demand forecasting** — 6-month forward forecasts for 10 strategically selected SKUs
- 💡 **6 actionable business recommendations** — directly tied to data-driven insights

> *"This analysis transforms 1.1 million rows of raw sales data into production planning intelligence for a multinational manufacturer."*

---

## 💼 Business Problem

A multinational FMCG company faces complex operational challenges across multiple regions:

| Challenge | Impact |
|---|---|
| High demand volatility | Unpredictable production requirements |
| Frequent promotions impacting baselines | Distorted demand signals |
| Low forecast accuracy | Excess inventory or stockouts |
| Non-optimised safety stock levels | Working capital inefficiency |
| Stockouts on A-products, overstocks on C-products | Lost revenue and waste |
| Large discrepancies in country-level performance | Misallocated resources |

### Problem Statement

> *Using the FMCG Multi-Country Sales Dataset, I want to understand total sales by country, the top SKUs by revenue, demand volatility, promotional effects and seasonality patterns — and forecast 10 SKUs with different demand profiles using XGBoost — in order to help the manufacturer make better decisions about production planning, inventory allocation and promotional investment across countries.*

---

## 📁 Dataset

**Source:** FMCG Multi-Country Sales Dataset (provided by Everything Data)

| Property | Value |
|---|---|
| Total Records | 1,100,000 rows |
| Features | 33 columns |
| Date Range | January 2021 — December 2023 |
| Countries | 7 (Italy, Spain, Germany, Austria, France, Poland, Netherlands) |
| Cities | 60+ with real lat/lon coordinates |
| SKUs | 102 unique products |
| Sales Channels | Modern Trade, Traditional Trade, Wholesale, E-commerce |

### Feature Categories

```
📅 Time        → date, year, month, day, weekofyear, weekday, is_weekend, is_holiday
🌍 Geography   → country, city, latitude, longitude
🌦️ Weather     → temperature, rain_mm
🏷️ Product     → sku_id, sku_name, category, subcategory, brand
💰 Commercial  → list_price, discount_pct, promo_flag, gross_sales, net_sales
📦 Supply      → stock_on_hand, stock_out_flag, lead_time_days, supplier_id, purchase_cost, margin_pct
```

---

## 📂 Project Structure

```
fmcg-sales-analysis-capstone/
│
├── 📓 FMCG_Capstone_Project.ipynb       # Main Jupyter Notebook (all code)
├── 📊 FMCG_Capstone_Presentation.pptx   # 12-slide presentation
├── 📄 FMCG_Project_Narrative.docx       # Full project narrative document
└── 📋 README.md                         # This file
```

---

## 🔬 Methodology

The project followed the standard Data Science lifecycle across two main assignments:

### Assignment 1 — Market Dynamics & Demand Analysis

```
Data Loading → Cleaning → EDA → ABC/XYZ Segmentation → Correlation → Seasonality
```

| Step | Technique | Tool |
|---|---|---|
| Data Quality Check | Null detection, duplicate removal | Pandas |
| Sales Analysis | GroupBy aggregation, ranking | Pandas |
| ABC Classification | Cumulative revenue distribution | Pandas + custom function |
| XYZ Classification | Coefficient of Variation (CV) | Pandas + NumPy |
| Correlation Analysis | Pearson correlation matrix | Pandas |
| Seasonality | Monthly & weekly aggregation | Seaborn, Matplotlib |

### Assignment 2 — Demand Forecasting

```
SKU Selection → Feature Engineering → Model Iteration → Evaluation → Forecast
```

The forecasting journey involved **4 progressive model iterations:**

```
Linear Regression → Random Forest → Tuned Random Forest → XGBoost ✅
     R²: 0.628        R²: 0.817          R²: 0.809          R²: 0.956
```

### Feature Engineering (Key Innovation)

```python
# Lag features — teaches model what sold last month
sku_enhanced['lag_1'] = sku_enhanced.groupby('sku_name')['units_sold'].shift(1)
sku_enhanced['lag_2'] = sku_enhanced.groupby('sku_name')['units_sold'].shift(2)
sku_enhanced['lag_3'] = sku_enhanced.groupby('sku_name')['units_sold'].shift(3)

# Rolling average — captures medium-term trends
sku_enhanced['rolling_mean_3'] = sku_enhanced.groupby('sku_name')['units_sold']\
                                  .transform(lambda x: x.shift(1).rolling(3).mean())

# Cyclical month encoding — captures seasonal cycles properly
sku_enhanced['month_sin'] = np.sin(2 * np.pi * sku_enhanced['month'] / 12)
sku_enhanced['month_cos'] = np.cos(2 * np.pi * sku_enhanced['month'] / 12)
```

### Model Validation Strategy

A **time-based train/test split** was used — more appropriate than random splitting for time series data:
- **Training set:** January 2021 — December 2022 (24 months)
- **Test set:** January 2023 — December 2023 (12 months)

---

## 📊 Key Findings

### 1️⃣ Sales by Country

| Rank | Country | Gross Sales | Share |
|---|---|---|---|
| 1 | 🇮🇹 Italy | 140,628,450 | Dominant leader |
| 2 | 🇪🇸 Spain | 109,279,741 | Strong 2nd |
| 3 | 🇩🇪 Germany | 90,766,602 | Significant 3rd |
| 4 | 🇦🇹 Austria | 44,018,768 | Mid-tier |
| 5 | 🇫🇷 France | 43,631,135 | Mid-tier |
| 6 | 🇵🇱 Poland | 43,535,481 | Mid-tier |
| 7 | 🇳🇱 Netherlands | 12,888,497 | Smallest market |

### 2️⃣ ABC/XYZ Segmentation

**ABC Classification (Revenue Segmentation):**

```
Class A — 49 SKUs (48%) → Top revenue generators → Zero stockout priority
Class B — 29 SKUs (28%) → Mid-tier performers   → Maintain buffer stock
Class C — 24 SKUs (24%) → Low revenue           → Review for rationalization
```

**XYZ Classification (Demand Volatility):**

```
Class X — 102 SKUs (100%) → CV range: 0.03 - 0.24 → ALL SKUs have stable demand
```
> ✅ All 102 SKUs exhibit highly stable, predictable demand — making the entire portfolio suitable for ML forecasting.

### 3️⃣ Demand Driver Correlation

| Variable | Correlation with Units Sold | Interpretation |
|---|---|---|
| Discount % | +0.29 | Weak positive — discounts have limited impact |
| Promo Flag | +0.29 | Weak positive — promotions have limited impact |
| List Price | -0.06 | Almost no relationship — low price sensitivity |

> 💡 **Key Insight:** Geography and seasonality are stronger demand drivers than commercial levers.

### 4️⃣ Seasonality Patterns

**Monthly:** Consistent bi-modal pattern across all 3 years:
- 📈 **Peak months:** July (summer) and December (holiday season)
- 📉 **Trough months:** February (post-holiday) and September (transition)

**Weekly:** Clear weekend effect:
- Weekdays (Mon–Fri): ~420 average gross sales
- Weekends (Sat–Sun): ~505 average gross sales (**+20% uplift**)

---

## 🔮 Forecasting Results

### XGBoost Model Performance (Test Set: 2023)

| SKU | Category | R² Score | Performance |
|---|---|---|---|
| BrandB Soda | Beverages | **0.956** | 🟢 Excellent |
| BrandF Water | Beverages | **0.952** | 🟢 Excellent |
| BrandE Soda | Beverages | **0.900** | 🟢 Excellent |
| BrandC Chips | Snacks | **0.821** | 🟢 Good |
| BrandF Nuts | Snacks | **0.800** | 🟢 Good |
| BrandD Detergent | Home Care | 0.053 | 🟡 Poor |
| BrandA Toothpaste | Personal Care | -0.203 | 🔴 Failed |
| BrandD Cheese | Dairy | -0.428 | 🔴 Failed |
| BrandF Milk | Dairy | -0.638 | 🔴 Failed |
| BrandB Soap | Personal Care | -1.011 | 🔴 Failed |

> ⚠️ **Why did Dairy & Personal Care fail?** Their demand is driven by supply chain availability, consumer habit and competitor pricing — variables not present in the dataset. This is a data gap finding, not a modelling failure.

### 2024 Demand Forecast (Jan–Jun)

| SKU | January | June | Change |
|---|---|---|---|
| BrandF Water | 28,802 units | 41,803 units | ⬆️ +45% summer spike |
| BrandE Soda | 24,135 units | 38,258 units | ⬆️ +58% summer spike |
| BrandB Soda | 19,156 units | 28,032 units | ⬆️ +46% summer spike |
| BrandC Chips | 31,173 units | 31,204 units | ➡️ Stable |
| BrandD Detergent | 29,912 units | 30,310 units | ➡️ Stable |

---

## 💡 Recommendations

| # | Recommendation | Priority |
|---|---|---|
| 1 | **Prioritise Italy & Spain supply chain** — both markets account for the majority of revenue. Zero stockouts on A-class SKUs is non-negotiable. | 🔴 Critical |
| 2 | **Plan production around bi-modal seasonality** — begin build-up 6-8 weeks before July and December peaks. Beverage production must increase by up to 58% for June. | 🔴 Critical |
| 3 | **Review Germany's C-class SKU portfolio** — 26 underperforming SKUs need rationalization, repackaging or promotional support. | 🟡 High |
| 4 | **Optimise retail replenishment for weekends** — schedule deliveries on Thursdays/Fridays to ensure full shelf availability for the 20% weekend sales uplift. | 🟡 High |
| 5 | **Deploy XGBoost for beverage S&OP planning** — integrate monthly forecasts into the Sales & Operations Planning process for production scheduling. | 🟢 Medium |
| 6 | **Invest in additional data for Dairy & Personal Care** — integrate competitor pricing, consumer panel data and distributor sell-through data to unlock forecasting capability for these categories. | 🟢 Medium |

---

## 🛠️ Tech Stack

```python
# Core
Python 3.12
Google Colab (Jupyter Notebook environment)

# Data Manipulation
pandas
numpy

# Visualisation
matplotlib
seaborn

# Machine Learning
scikit-learn    # Linear Regression, Random Forest, cross-validation, GridSearch
xgboost         # Final forecasting model
```

---

## ▶️ How To Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/fmcg-sales-analysis-capstone.git
cd fmcg-sales-analysis-capstone
```

### 2. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 3. Download the dataset
The dataset is available at: [FMCG Multi-Country Sales Dataset](https://drive.google.com/file/d/1S7f1SfS3jibw4rJAop0iPE6KxT-w4uTX/view)

### 4. Open the notebook
```bash
jupyter notebook FMCG_Capstone_Project.ipynb
```
Or open directly in **Google Colab** by uploading the `.ipynb` file.

### 5. Mount Google Drive (if using Colab)
```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/your_folder/fmcg_sales_3years_1M_rows.csv')
```

---

## 📦 Project Deliverables

| Deliverable | Description |
|---|---|
| 📓 `FMCG_Capstone_Project.ipynb` | Complete Jupyter Notebook with all EDA and forecasting code |
| 📊 `FMCG_Capstone_Presentation.pptx` | 12-slide professional presentation |
| 📄 `FMCG_Project_Narrative.docx` | Full project narrative document (9 sections) |
| 📋 `README.md` | This file |

---

## 👨‍💻 Author

<div align="center">

### Ian Waweru Kariuki

**BSc. Data Science & Analytics**
The Cooperative University of Kenya

**Data Science Cohort 5 — Everything Data (2026)**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](www.linkedin.com/in/ian-kariuki-77a8672a2)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Iannoh-png/Everything-Data-Official-Project)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:iankariuki450@gmail.com)

</div>

---

<div align="center">

*Built with ❤️ as part of Data Science Cohort 5 Capstone Project*

⭐ **If you found this project useful, please consider giving it a star!** ⭐

</div>
