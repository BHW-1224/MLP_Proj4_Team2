# MLP_Proj4_Team2

By following the steps below, the results of each step can be reproduced.

**IMPORTANT: Kaggle Environment Setup**
For all steps below, you **MUST Disable Internet Access** in the Kaggle Notebook settings.
1.  Go to the **Settings** menu in the right panel.
2.  Set **Internet** to **"Off"**.
3.  This is strictly required for the competition's inference server to function correctly.

## Step-1
To reproduce the Linear Regression baseline results, follow these steps:
1.  Open a new Kaggle Notebook.
2.  Upload the `step1.ipynb` file to the Kaggle environment.
3.  Click "Add input" on the right panel, search for **"Hull Tactical Market Prediction"**, and add the dataset.
4.  **Ensure Internet Access is turned OFF.**
5.  Make sure standard libraries (pandas, numpy, scikit-learn, polars) are available — these are preinstalled in Kaggle by default.
6.  Run all cells without modification.

## Step-2
To reproduce the LightGBM-based baseline results, follow these steps:
1.  Open a new Kaggle Notebook.
2.  Upload the `step2.ipynb` file to the Kaggle environment.
3.  Click "Add input" on the right panel, search for **"Hull Tactical Market Prediction"**, and add the dataset.
4.  **Ensure Internet Access is turned OFF.**
5.  Make sure `lightgbm` is available — this is preinstalled in Kaggle by default.
6.  Run all cells to observe the validation RMSE.

## Step-3
To reproduce the Feature Engineering process (Finding the best feature subset), follow these steps:
1.  Open a new Kaggle Notebook.
2.  Upload the `step3.ipynb` file to the Kaggle environment.
3.  Click "Add input" on the right panel, search for **"Hull Tactical Market Prediction"**, and add the dataset.
4.  **Ensure Internet Access is turned OFF.**
5.  Run all cells. This step evaluates feature group combinations and identifies the optimal subset (Interest + Valuation columns).

## Step-4
To reproduce the Backtest results and visualization, follow these steps:
1.  Open a new Kaggle Notebook.
2.  Upload the `step4.ipynb` file to the Kaggle environment.
3.  Click "Add input" on the right panel, search for **"Hull Tactical Market Prediction"**, and add the dataset.
4.  **Ensure Internet Access is turned OFF.**
5.  Run all cells. This generates cumulative return graphs and calculates metrics like Sharpe ratio and Max Drawdown.

## Step-5
To reproduce the Final Submission (Training + Inference Server), follow these steps:
1.  Open a new Kaggle Notebook.
2.  Upload the `step5.ipynb` file to the Kaggle environment.
3.  Click "Add input" on the right panel, search for **"Hull Tactical Market Prediction"**, and add the dataset.
4.  **Ensure Internet Access is turned OFF.** (The submission will fail if the internet is on).
5.  Run all cells. This script trains the LightGBM model using the 22 selected features and starts the `kaggle_evaluation` inference server.

## Appendix
1. launch bonus.py
2. You can see top 10 Feature Importance, KOSPI Prediction, and results.
