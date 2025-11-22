"""Movie Revenue Regression with H2O AutoML
------------------------------------------------
This script is designed to be run in VS Code (or any local IDE) using Python 3.10 or 3.11.

IMPORTANT:
- You must have Java installed (JDK or JRE) and on your system PATH.
- You must install the H2O Python package:

    pip install h2o seaborn matplotlib pandas numpy

Then place this script and `Movie_regression (1).csv` in the same folder and run.
"""


# --- Block 1: Imports -------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import h2o
from h2o.automl import H2OAutoML


# --- Block 2: Load data with pandas ----------------------------------------

# Adjust the path if your CSV lives somewhere else
CSV_PATH = "/Users/connorross/Downloads/Movie_regression (1).csv"

df = pd.read_csv(CSV_PATH)

print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# --- Block 3: Basic data overview (for report 'Data' section) --------------

print("\nColumn data types:")
print(df.dtypes)

print("\nMissing values per column:")
print(df.isnull().sum())

print("\nNumeric summary statistics:")
print(df.describe())


# --- Block 4: Univariate exploration (target + key drivers) -----------------

target_col = "Collection"

plt.figure()
sns.histplot(df[target_col], kde=True)
plt.title("Distribution of Movie Revenue (Collection)")
plt.xlabel("Collection")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(df["Marketing expense"], kde=True)
plt.title("Distribution of Marketing Expense")
plt.xlabel("Marketing expense")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(df["Trailer_views"], kde=True)
plt.title("Distribution of Trailer Views")
plt.xlabel("Trailer_views")
plt.ylabel("Count")
plt.tight_layout()
plt.show()


# --- Block 5: Correlation matrix and heatmap --------------------------------

numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.show()

print("\nCorrelation with Collection:")
print(corr_matrix[target_col].sort_values(ascending=False))


# --- Block 6: Bivariate plots (numeric and categorical vs target) ----------

plt.figure()
sns.scatterplot(data=df, x="Marketing expense", y="Collection")
plt.title("Marketing Expense vs Collection")
plt.tight_layout()
plt.show()

plt.figure()
sns.scatterplot(data=df, x="Trailer_views", y="Collection")
plt.title("Trailer Views vs Collection")
plt.tight_layout()
plt.show()

plt.figure()
sns.scatterplot(data=df, x="Budget", y="Collection")
plt.title("Budget vs Collection")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data=df, x="3D_available", y="Collection")
plt.title("Collection by 3D Availability")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(data=df, x="Genre", y="Collection")
plt.title("Collection by Genre")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# --- Block 7: Initialize H2O and convert pandas DataFrame -------------------

print("\nInitializing H2O cluster...")
h2o.init()

# Convert pandas DataFrame to H2OFrame
hf = h2o.H2OFrame(df)

# Ensure categoricals are factors
for cat_col in ["3D_available", "Genre"]:
    if cat_col in hf.columns:
        hf[cat_col] = hf[cat_col].asfactor()

# Quick H2O summary
print("\nH2O frame summary:")
print(hf.describe())


# --- Block 8: Train/validation/test split ----------------------------------

# We will split into: 60% train, 20% valid, 20% test
train, valid, test = hf.split_frame(ratios=[0.6, 0.2], seed=123)

y = "Collection"
x = [col for col in hf.columns if col != y]

print("\nTraining rows:", train.nrows)
print("Validation rows:", valid.nrows)
print("Test rows:", test.nrows)


# --- Block 9: Run H2O AutoML for regression --------------------------------

# Since 'Collection' is numeric, H2O will treat this as a regression task.
aml = H2OAutoML(
    max_models=20,          # can adjust based on time
    max_runtime_secs=600,   # total time budget in seconds
    seed=1
)

print("\nStarting H2O AutoML...")
aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

print("\nAutoML training complete.")


# --- Block 10: Leaderboard --------------------------------------------------

print("\nAutoML Leaderboard:")
lb = aml.leaderboard
# Show all models in the leaderboard
print(lb.head(rows=lb.nrows))


# --- Block 11: Evaluate best model on test data -----------------------------

best_model = aml.leader

print("\nBest model details:")
print(best_model)

print("\nPerformance of best model on TEST data:")
perf = best_model.model_performance(test_data=test)
print(perf)

# Key regression metrics
try:
    print("RMSE:", perf.rmse())
    print("MAE :", perf.mae())
    print("R2  :", perf.r2())
except Exception as e:
    print("Could not extract standard regression metrics:", e)


# --- Block 12: Global model explanation (optional, but good for report) -----

# This may open multiple plots depending on your environment.
# Comment out if it causes issues.
try:
    print("\nGenerating AutoML explainability plots...")
    aml.explain(frame=test, figsize=(8, 6))
except Exception as e:
    print("Explainability failed or is not supported in this environment:", e)


# --- Block 13: Predict revenue for new movies -------------------------------

# Example new movie(s) – adjust these values as desired
new_movies = pd.DataFrame([
    {
        "Marketing expense": 500.0,
        "Production expense": 250.0,
        "Multiplex coverage": 7.5,
        "Budget": 40000.0,
        "Movie_length": 120.0,
        "Lead_ Actor_Rating": 7.8,
        "Lead_Actress_rating": 7.5,
        "Director_rating": 8.2,
        "Producer_rating": 7.9,
        "Critic_rating": 8.0,
        "Trailer_views": 550000,
        "3D_available": "YES",
        "Time_taken": 130.0,
        "Twitter_hastags": 300.0,
        "Genre": "Thriller",
        "Avg_age_actors": 35,
        "Num_multiplex": 480
    }
])

# Convert to H2OFrame and predict
new_movies_hf = h2o.H2OFrame(new_movies)
new_movies_hf["3D_available"] = new_movies_hf["3D_available"].asfactor()
new_movies_hf["Genre"] = new_movies_hf["Genre"].asfactor()

new_predictions = best_model.predict(new_movies_hf)
new_predictions_pd = new_predictions.as_data_frame()

print("\nPredicted Collection for new movies:")
print(pd.concat([new_movies, new_predictions_pd], axis=1))


# --- Block 14: Save best model to disk --------------------------------------

# This will save the model in the current working directory by default.
model_path = h2o.save_model(model=best_model, path=".", force=True)

print("\nBest model saved to:", model_path)


# --- Block 15: Optional – Shutdown H2O cluster ------------------------------

# Uncomment if you want to explicitly shut down H2O at the end of the script.
# h2o.shutdown(prompt=False)
