Movie Revenue Prediction – H2O AutoML Project

<img width="400" height="300" alt="Variable Importance" src="https://github.com/user-attachments/assets/bc29eaee-3e95-448e-8804-63df585214d9" />

This project delivers a full end-to-end machine learning pipeline using Python and H2O AutoML to forecast movie box-office revenue based on production budgets, marketing spend, talent ratings, engagement metrics, and distribution attributes. It demonstrates my ability to perform exploratory analysis, engineer features, train and evaluate multiple machine learning models, deploy an ensemble model, and translate technical results into clear business recommendations.

Using pandas, NumPy, Seaborn, Matplotlib, and H2O, the pipeline automates model training across GBM, DRF, Deep Learning, and Stacked Ensemble architectures. H2O AutoML selected a Stacked Ensemble as the top-performing model, achieving strong predictive accuracy on unseen data (R² ≈ 0.898, RMSE ≈ $6.24K). Feature interpretation revealed that Budget, Trailer_views, and Multiplex coverage were the most influential drivers of revenue across models.

To demonstrate practical application, the deployed model was used to forecast revenue for a hypothetical new film with moderate marketing activity, strong trailer engagement, and wide multiplex reach. The model estimated an expected box-office revenue of approximately $91,509. This capability allows users to simulate the financial impact of promotional strategies, test alternative release plans, and support data-driven decision-making before a film enters the market.

The repository includes the full Python pipeline, dataset, exploratory visualizations, AutoML outputs, and a complete written report. Overall, the project showcases my proficiency in modern machine learning tooling, automated modeling workflows, data visualization, and delivering insights that bridge technical analysis and business value.
