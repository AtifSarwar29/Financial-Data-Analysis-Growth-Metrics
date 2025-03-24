# Rating Criteria

## Is the Data Properly Organized?
Yes, the data is well-organized with financial fundamentals (e.g., revenueGrowth, netProfitMargin) and price data (open, close, volume). Engineered features like RSI (momentum) and MACD (trend) enhance its predictive power. Preprocessing is appropriate, but using close (not adjClose) may ignore stock splits/dividends, and a chronological split is needed for time-series data.

## Is the Dataset Suitable for Model Training?
Yes, it’s suitable with relevant features (price, financials, RSI for momentum, MACD for trends) and balanced training data (via SMOTE: 382 samples per class). However, the small size (645 rows, 215 features) risks overfitting, and the stratified split may cause data leakage in a time-series context.

# Model Training and Selection Issues

## Model and Training Method for Predicting 10% Growth in 90 Days

**XGBoost (Current Model):** Model: XGBoost with scale_pos_weight=1 (balanced post-SMOTE).
Training Method: Use a chronological split (first 80% for training, last 20% for testing), apply SMOTE, train with early stopping (10 rounds, logloss), and lower the prediction threshold to 0.2 to improve recall for class 1.

**Reason:** XGBoost performs well (95% accuracy, 0.91 F1-score for class 1), leveraging features like macd (trend) and rsi (momentum), but the stratified split likely inflates performance due to leakage.

**LSTM (Recommended Model):** Model: LSTM with 1-2 LSTM layers, dropout, and a dense layer with sigmoid activation.
Training Method: Reshape data into 90-day sequences (e.g., close, rsi, macd), use a chronological split, apply class weighting, train with binary cross-entropy loss, use early stopping, and evaluate with F1-score for class 1.

**Reason:** LSTM captures temporal patterns in rsi (momentum shifts) and macd (trend changes) better than XGBoost, which is critical for time-series stock prediction.

## Extract Key Impact Columns

**XGBoost:** The script uses get_score(importance_type='gain'), identifying cashRatio (104.54), macd (14.36, trend indicator), and payoutRatio (12.96) as top features. Use SHAP values for deeper insights.

**LSTM:** Use permutation importance (shuffle features like macd and rsi, measure F1-score drop) or add an attention layer to highlight key features.

## How to Infer More Effective New Data Fields?

**Both Models:**
- Add lagged features for key columns (e.g., rsi_lag1, macd_lag1) to capture momentum and trend shifts over time.
- Create interactions (e.g., rsi * macd to combine momentum and trend signals).
- Add technical indicators: Stochastic Oscillator (for additional momentum) and ATR (for volatility).
- Compute rolling stats (e.g., rsi_ma20, macd_ma20) and ratios (e.g., macd / signal_line).
- Create financial ratios (e.g., operatingCashFlow / totalDebt).
