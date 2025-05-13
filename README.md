# Deep-Models-with-Engineered-Features

## 🧠 Final Model: `MyModel` — Ensemble of Random Forest, CNN & LSTM

### ✅ Summary

This project builds an ensemble forecasting model for **AAPL stock returns** using an advanced, domain-informed feature set and three complementary machine learning architectures: **Random Forest**, **LSTM**, and **CNN**.

---

### 🧱 Feature Engineering Highlights

All features are generated using the `compute_phase4_features` function, designed with deep financial insight:

- **Lagged Returns**:  
  - `AAPL_Return`, `Target`, `{ticker}_Return_Lag1`  
  - Captures autocorrelation and momentum

- **Temporal Patterns**:  
  - `Day_of_Week`, `Month` → one-hot encoded  
  - Models weekly and seasonal return patterns

- **Volatility Features**:  
  - `Squared_Return`, `Vol_Clustering_20`, `Volatility_20`, `Vol_Adjusted_Return`  
  - Quantifies return clustering and persistence of risk

- **Technical Indicators**:  
  - `MACD`, `MACD_Signal`, `MACD_Diff` (clipped to [-0.5, 0.5])  
  - `Stochastic_K`, `Stochastic_D` (14/3 window)  
  - Helps detect momentum and overbought/oversold zones

- **Market Regime Indicators**:  
  - `MA_200`, `Market_Regime` (bullish/bearish)  
  - Encodes long-term market positioning

- **Higher-Order Statistical Features**:  
  - `Skewness_126`, `Kurtosis_126`  
  - Captures fat tails and distributional asymmetry

- **Cross-Asset Interactions**:  
  - `Volatility_MACD_Interact`, `{ticker}_Vol_Interact` (e.g., with SPY, XLK)  
  - Captures co-movement with market/sector ETFs

---

### 🧮 Model Architecture: `MyModel`

The final prediction is an **inverse-MSE weighted ensemble** of:

- **Random Forest**  
  - Learns from the full engineered feature set  
  - Handles non-linearity and avoids overfitting via averaging

- **LSTM**  
  - Captures temporal dependencies in lagged returns, volatility, and time-based features  
  - Ideal for time series modeling via memory cells

- **CNN**  
  - Trained on PCA-transformed sequences  
  - Extracts spatial patterns from indicators and interactions

Predictions from each model (`y_pred_rf`, `y_pred_lstm`, `y_pred_cnn`) are combined using weights saved in `ensemble_weights_*.json`.

---

### 📌 Final Remarks

This pipeline reflects a **strategic fusion of financial domain knowledge and machine learning innovation**, combining:

- Domain-specific signals (MACD, Skewness, Market Regimes)
- Careful preprocessing (standardization, clipping, PCA)
- Ensemble modeling across tree-based, recurrent, and convolutional networks

The model is robust, extensible, and adaptable to other equities with appropriate data preparation.
