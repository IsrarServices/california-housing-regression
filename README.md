# California Housing Regression

Predicting median house values across California census districts using classic regression models, with a full data preprocessing pipeline and model comparison.

## Dataset

The dataset used is the **California Housing dataset** (1990 census), containing 20,640 rows. Each row represents a census block group, with features describing location, housing age, room/bedroom counts, population, households, and median income. The target variable is `median_house_value`.

| Feature | Description |
|---|---|
| `longitude`, `latitude` | Geographic location |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms`, `total_bedrooms` | Total rooms/bedrooms in the block |
| `population`, `households` | Population and household counts |
| `median_income` | Median income of households (in tens of thousands) |
| `ocean_proximity` | Categorical: distance category from the ocean |
| `median_house_value` | **Target** — median house value (USD) |

## Project Structure

```
.
├── housing_regression.py     # Full pipeline: preprocessing + model training + comparison
├── eda_plots.png              # Exploratory data analysis visualizations
├── model_comparison.png       # Bar chart comparing model R² scores
└── README.md
```

## Steps Performed

### 1. Data Preprocessing
- **Data Cleaning**: standardized categorical text formatting, checked for invalid/impossible values (e.g. negative counts).
- **Missing Value Handling**: `total_bedrooms` had 207 missing values, imputed using the column median.
- **Duplicate Removal**: checked for and removed fully duplicate rows.
- **Feature Selection**: kept all original numeric and categorical features; engineered three additional ratio features — `rooms_per_household`, `bedrooms_per_room`, and `population_per_household` — and validated their relevance via correlation analysis.
- **Data Visualization**: distribution of target variable, feature correlation heatmap, income-vs-price scatter plot, and a geographic scatter plot of prices by location.
- **Train-Test Split**: 80% train / 20% test.

### 2. Model Training
Four regression models were trained and evaluated:

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

All models were trained inside a single `scikit-learn` pipeline that handles scaling of numeric features and one-hot encoding of the categorical feature, so preprocessing is applied consistently and without leakage between train and test sets.

### 3. Model Comparison

| Model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost Regressor | 29,886.94 | 45,683.32 | **0.8407** |
| Random Forest Regressor | 32,805.67 | 50,665.84 | 0.8041 |
| Decision Tree Regressor | 40,927.36 | 63,951.93 | 0.6879 |
| Linear Regression | 50,888.66 | 72,668.54 | 0.5970 |

**Best performing model: XGBoost Regressor**, with the highest R² and lowest error. Tree-based ensemble methods (XGBoost, Random Forest) substantially outperform Linear Regression here, since house prices depend on non-linear relationships and interactions between features — particularly location and income — that a linear model can't capture.

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
python housing_regression.py
```

The script will print preprocessing and evaluation results to the console and save `eda_plots.png` and `model_comparison.png` to the working directory.

## Key Observations

- `median_income` is the strongest single predictor of house value (correlation ≈ 0.69).
- House prices show clear geographic clustering — coastal areas (Bay Area, LA) are notably more expensive than inland regions.
- `bedrooms_per_room` is negatively correlated with price, suggesting blocks with a higher *proportion* of bedrooms (relative to total rooms) tend to have lower-valued homes.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn`
