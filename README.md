Movie Revenue Prediction – H2O AutoML Project

<img width="400" height="300" alt="Variable Importance" src="https://github.com/user-attachments/assets/bc29eaee-3e95-448e-8804-63df585214d9" />

This project builds a full predictive modeling pipeline to estimate movie box-office revenue using H2O AutoML. Using a dataset of 506 films, the workflow includes data preparation, exploratory analysis, model training, ensemble selection, deployment, and business-focused interpretation. The final Stacked Ensemble model explains ~90% of revenue variance and provides actionable insights for forecasting future film performance.

Movie-Revenue-AutoML/
│
├── movie_regression_h2o_pipeline.py
├── Movie_regression (1).csv
├── Report/
│ └── DS_Report.pdf
├── Screenshots/
│ ├── Budget vs Collection.png
│ ├── Distribution of Movie Rev.png
│ ├── Trailer Views vs Collection.png
│ ├── Residual Analysis.png
│ ├── Variable Importance.png
│ └── Variable Importance Heatmap.png
└── README.md

Project Objectives

Identify which features most strongly influence movie revenue.

Build an automated machine learning pipeline using H2O AutoML.

Evaluate multiple models (GBM, DRF, Deep Learning, Ensembles).

Deploy the top-performing model for scoring new films.

Deliver a clear, business-ready report for stakeholders.

Key Results

Top Model: Stacked Ensemble (Best of Family)

Test R²: ~0.898

Test RMSE: ~$6.24K

Top Predictors: Budget, Trailer Views, Multiplex Coverage

Use Case: Pre-release forecasting, marketing allocation, scheduling decisions

How to Run the Pipeline

Install dependencies:

pip install h2o pandas numpy seaborn matplotlib


Ensure Java (JDK 17+) is installed for H2O.

Run the script:

python movie_regression_h2o_pipeline.py


The model will train, evaluate, output diagnostics, and save the best model.

Files Included

Dataset: Movie_regression (1).csv

Python Pipeline: movie_regression_h2o_pipeline.py

Screenshots: Learning curve, residuals, variable importance, etc.

Final Report: Data science analysis and recommendations
