import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set seed for reproducibility
seed = 42

# 1. LOAD DATA
print("--- Loading California Housing Dataset ---")
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print(f"Data shape: {df.shape}")
print(df.head())

# 2. DATA PREPROCESSING & CLEANING
print("\n--- Data Preprocessing & Cleaning ---")

# Standardize categorical values in ocean_proximity
df["ocean_proximity"] = df["ocean_proximity"].str.strip().str.upper()

# Check for negative values
numeric_cols = df.select_dtypes(include=[np.number]).columns
print("Negative values count:")
print((df[numeric_cols] < 0).sum())

# Handle missing values in total_bedrooms (impute with median)
print("\nMissing values before imputation:")
print(df.isnull().sum())

median_val = df["total_bedrooms"].median()
df["total_bedrooms"] = df["total_bedrooms"].fillna(median_val)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# Remove duplicate rows
duplicates = df.duplicated().sum()
df = df.drop_duplicates()
print(f"\nRemoved {duplicates} duplicate rows. New shape: {df.shape}")

# 3. FEATURE ENGINEERING
print("\n--- Feature Engineering ---")
# Create new ratio features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# Check correlations with target variable
correlations = df.select_dtypes(include=[np.number]).corr()["median_house_value"].sort_values(ascending=False)
print("Correlation with median_house_value:")
print(correlations)

# Prepare features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

num_features = X.select_dtypes(include=[np.number]).columns.tolist()
cat_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

# 4. DATA VISUALIZATION
print("\n--- Visualizing Data ---")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target distribution
axes[0, 0].hist(df["median_house_value"], bins=50, color="steelblue", edgecolor="black")
axes[0, 0].set_title("Distribution of Median House Value")
axes[0, 0].set_xlabel("House Value ($)")
axes[0, 0].set_ylabel("Count")

# Heatmap
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap="coolwarm", ax=axes[0, 1])
axes[0, 1].set_title("Correlation Heatmap")

# Scatter: Income vs House Value
axes[1, 0].scatter(df["median_income"], df["median_house_value"], alpha=0.2, s=8, color="darkorange")
axes[1, 0].set_title("Median Income vs House Value")
axes[1, 0].set_xlabel("Median Income")
axes[1, 0].set_ylabel("House Value ($)")

# Scatter: Geo location
sc = axes[1, 1].scatter(df["longitude"], df["latitude"], c=df["median_house_value"],
                         cmap="viridis", alpha=0.4, s=8)
axes[1, 1].set_title("Geographic Distribution of Prices")
axes[1, 1].set_xlabel("Longitude")
axes[1, 1].set_ylabel("Latitude")
plt.colorbar(sc, ax=axes[1, 1], label="House Value ($)")

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=120)
print("Plots saved as eda_plots.png")
plt.close()

# 5. SPLIT AND PREPROCESS
print("\n--- Split and Preprocessing Pipeline ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Define preprocessing pipeline
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 6. MODEL TRAINING & EVALUATION
print("\n--- Training Models ---")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=seed, max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1, max_depth=15),
    "XGBoost": XGBRegressor(n_estimators=200, random_state=seed, max_depth=6, learning_rate=0.1)
}

results = []

for name, model in models.items():
    # Pipeline makes it clean to preprocess and run
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})
    print(f"\n{name}:")
    print(f"  MAE  : {mae:,.2f}")
    print(f"  RMSE : {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")

# Compare results
results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

best_model = results_df.iloc[0]["Model"]
print(f"\nBest Model: {best_model}")

# Plot comparison
plt.figure(figsize=(8, 5))
plt.bar(results_df["Model"], results_df["R2"], color="seagreen")
plt.ylabel("R2 Score")
plt.title("Model Comparison (R2 Score)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
print("Comparison chart saved as model_comparison.png")
plt.close()
