import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# 경고 메시지 억제
warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Data Collection
# ---------------------------------------------------------
def fetch_market_data(start_date='2010-01-01', end_date='2025-11-20'):
    """
    [Report Section: Data Collection]
    - Source: Naver Finance (KOSPI, USD/KRW), Investing.com (S&P500)
    - Method: FinanceDataReader Library
    """
    print("=== [STEP 1] Data Collection via FinanceDataReader ===")
    
    # 1. KOSPI (Target Market)
    kospi = fdr.DataReader('KS11', start_date, end_date)[['Close']].rename(columns={'Close': 'KOSPI'})
    
    # 2. S&P 500 (Leading Indicator)
    sp500 = fdr.DataReader('US500', start_date, end_date)[['Close']].rename(columns={'Close': 'SP500'})
    
    # 3. USD/KRW (Macro Indicator)
    usd_krw = fdr.DataReader('USD/KRW', start_date, end_date)[['Close']].rename(columns={'Close': 'USD_KRW'})
    
    # Merge and Drop NaN
    df = pd.concat([kospi, sp500, usd_krw], axis=1).dropna()
    
    print(f"Data Loaded: {df.shape}, Period: {df.index[0].date()} ~ {df.index[-1].date()}")
    return df

# ---------------------------------------------------------
# 2. Feature Engineering
# ---------------------------------------------------------
def generate_features(df):
    """
    [Report Section: Feature Engineering]
    - Generates Technical Indicators (RSI, MACD, Bollinger Bands)
    - Applies Time-Lagging (t-1) to prevent Look-ahead Bias
    """
    print("=== [STEP 2] Feature Engineering (Technical Indicators & Lagging) ===")
    data = df.copy()
    
    # 1. Basic Returns
    data['KOSPI_Ret'] = data['KOSPI'].pct_change()
    data['SP500_Ret'] = data['SP500'].pct_change()
    data['USD_KRW_Ret'] = data['USD_KRW'].pct_change()
    
    # 2. Technical Indicators
    # (1) Moving Averages & Disparity
    for window in [5, 20, 60]:
        data[f'MA_{window}'] = data['KOSPI'].rolling(window=window).mean()
        data[f'Disparity_{window}'] = data['KOSPI'] / data[f'MA_{window}']
    
    # (2) Volatility (Risk Measure - Critical for Bonus Constraint)
    data['Vol_20'] = data['KOSPI_Ret'].rolling(window=20).std()
    
    # (3) RSI (14-day)
    delta = data['KOSPI'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # (4) MACD
    exp12 = data['KOSPI'].ewm(span=12, adjust=False).mean()
    exp26 = data['KOSPI'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp12 - exp26
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # (5) Bollinger Bands
    data['BB_Mid'] = data['KOSPI'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Mid'] + 2 * data['KOSPI'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Mid'] - 2 * data['KOSPI'].rolling(window=20).std()
    data['BB_Pos'] = (data['KOSPI'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
    
    # 3. Lagging (Shift t-1) - Essential for Prediction
    # Target(KOSPI_Ret) is T (Today), Features must be T-1 (Yesterday)
    feature_candidates = [
        'SP500_Ret', 'USD_KRW_Ret', 'KOSPI_Ret',
        'Disparity_5', 'Disparity_20', 'Disparity_60',
        'Vol_20', 'RSI_14', 'MACD', 'MACD_Signal', 'BB_Pos'
    ]
    
    shifted_cols = []
    for col in feature_candidates:
        new_col = f"{col}_Lag1"
        data[new_col] = data[col].shift(1)
        shifted_cols.append(new_col)
    
    # Add Extra Momentum
    data['KOSPI_Ret_Lag2'] = data['KOSPI_Ret'].shift(2)
    data['KOSPI_Ret_Lag3'] = data['KOSPI_Ret'].shift(3)
    shifted_cols.extend(['KOSPI_Ret_Lag2', 'KOSPI_Ret_Lag3'])
    
    # Clean up
    data_clean = data.dropna().copy()
    
    print(f"Features Generated: {len(shifted_cols)} features ready.")
    return data_clean, shifted_cols

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
def train_lightgbm(train_df, features, target='KOSPI_Ret'):
    """
    [Report Section: Model Development]
    - Algorithm: LightGBM Regressor
    - Setup: Time-Series Split (Train on past, Predict future)
    """
    print("=== [STEP 3] Model Training (LightGBM) ===")
    
    X_train = train_df[features].values
    y_train = train_df[target].values
    
    # Hyperparameters (Optimized for Stability)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 31,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "random_state": 42
    }
    
    # Train
    train_data = lgb.Dataset(X_train, label=y_train)
    model = lgb.train(params, train_data, num_boost_round=1000)
    
    # Feature Importance Visualization
    importance = model.feature_importance()
    imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    imp_df = imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(10), palette='viridis')
    plt.title('Top 10 Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.show() # Capture this for the report
    
    return model, imp_df

# ---------------------------------------------------------
# 4. Strategy & Evaluation Logic
# ---------------------------------------------------------
def pred_to_allocation(y_pred, train_mean, train_std, current_vol, bench_vol, alpha=1.0):
    """
    [Report Section: Strategy Logic & Risk Control]
    - Converts Regression Output -> Allocation (0.0 ~ 2.0)
    - KEY: Applies Volatility Constraint (Inverse Volatility Scaling)
    """
    # 1. Z-score Scaling (Signal Strength)
    z = (y_pred - train_mean) / (train_std + 1e-9)
    base_alloc = 1.0 + alpha * z
    
    # 2. Volatility Constraint Logic (< 1.2x Benchmark)
    # If current market risk (current_vol) is high, reduce weight.
    target_vol = bench_vol * 1.2
    
    # Avoid division by zero
    safe_vol = np.where(current_vol <= 0, bench_vol, current_vol)
    vol_scale = target_vol / safe_vol
    
    # Apply penalty only (do not leverage up based on low vol to keep it safe)
    vol_multiplier = np.clip(vol_scale, 0.0, 1.0)
    
    final_alloc = base_alloc * vol_multiplier
    
    # Clip to allowed range [0, 2]
    return np.clip(final_alloc, 0.0, 2.0)

def calculate_metrics(returns, allocations):
    """
    [Report Section: Metrics]
    - Calculates Annualized Return, Volatility, Sharpe Ratio, Volatility Ratio
    """
    strategy_ret = allocations * returns
    excess_ret = strategy_ret - (0.035/252) # Risk-free rate approx
    
    # Annualization
    ann_ret = np.mean(excess_ret) * 252
    ann_vol = np.std(excess_ret) * np.sqrt(252)
    sharpe = ann_ret / (ann_vol + 1e-9)
    
    # Benchmark Volatility
    bench_vol = np.std(returns) * np.sqrt(252)
    vol_ratio = ann_vol / (bench_vol + 1e-9)
    
    return sharpe, vol_ratio, ann_ret, ann_vol

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
def run_bonus_project():
    print("=== STARTING KOSPI BONUS PROJECT PIPELINE ===")
    
    # 1. Load Data
    raw_df = fetch_market_data()
    
    # 2. Feature Engineering
    df, features = generate_features(raw_df)
    
    # 3. Train/Test Split (Backtest on last 365 days)
    test_days = 365
    train_df = df.iloc[:-test_days]
    test_df = df.iloc[-test_days:].copy()
    
    # 4. Model Training
    model, _ = train_lightgbm(train_df, features)
    
    # 5. Prediction & Strategy Application
    print("\n=== [STEP 4] Backtesting Strategy ===")
    
    # Statistics for normalization
    train_y_mean = np.mean(train_df['KOSPI_Ret'])
    train_y_std = np.std(train_df['KOSPI_Ret'])
    bench_vol = train_df['Vol_20'].mean() # Benchmark Volatility Baseline
    
    # Predict
    y_pred = model.predict(test_df[features].values)
    
    # Calculate Allocation (With Volatility Control)
    test_df['Allocation'] = pred_to_allocation(
        y_pred, train_y_mean, train_y_std, 
        current_vol=test_df['Vol_20_Lag1'].values, # Key: Use Yesterday's Volatility info
        bench_vol=bench_vol
    )
    
    # Calculate Strategy Returns
    test_df['Strategy_Ret'] = test_df['Allocation'] * test_df['KOSPI_Ret']
    
    # 6. Evaluation (Model Strategy)
    sharpe, vol_ratio, ann_ret, ann_vol = calculate_metrics(
        test_df['KOSPI_Ret'].values, 
        test_df['Allocation'].values
    )
    
    print("-" * 50)
    print(f">>> FINAL RESULTS (Report Summary) <<<")
    print(f"Sharpe Ratio      : {sharpe:.4f}")
    print(f"Annualized Return : {ann_ret:.2%}")
    print(f"Volatility Ratio  : {vol_ratio:.4f} (Constraint: < 1.2)")
    print("-" * 50)
    
    # 7. Evaluation (Benchmark Buy & Hold)
    # 비중을 모두 1.0으로 설정하여 벤치마크 성과 계산
    ones_allocation = np.ones(len(test_df))
    bench_sharpe, bench_vol_ratio, bench_ann_ret, bench_ann_vol = calculate_metrics(
        test_df['KOSPI_Ret'].values,
        ones_allocation
    )

    print(f"=== KOSPI Benchmark Metrics ===")
    print(f"Annualized Return : {bench_ann_ret:.2%}")
    print(f"Sharpe Ratio      : {bench_sharpe:.4f}")
    print(f"Volatility (Ann.) : {bench_ann_vol:.2%}")
    print("-" * 50)

    # Check Constraints
    if vol_ratio < 1.2:
        print("[PASS] Volatility Constraint Satisfied.")
    else:
        print("[FAIL] Adjust Volatility Penalty.")

    # 8. Visualization (Cumulative Return)
    test_df['Cum_Bench'] = (1 + test_df['KOSPI_Ret']).cumprod()
    test_df['Cum_Strat'] = (1 + test_df['Strategy_Ret']).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(test_df.index, test_df['Cum_Bench'], label='KOSPI Benchmark', color='gray', linestyle='--')
    plt.plot(test_df.index, test_df['Cum_Strat'], label='AI Model Strategy', color='red', linewidth=2)
    plt.title(f'Bonus Project: KOSPI Prediction (Sharpe: {sharpe:.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the pipeline
if __name__ == "__main__":
    run_bonus_project()