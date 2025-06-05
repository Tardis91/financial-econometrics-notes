# Stationarity in Time Series

Understanding stationarity is fundamental in financial econometrics. Many time series models — including ARMA, GARCH, and VAR — assume the data is **stationary**.

---

## 🧠 What Is Stationarity?

A time series is **stationary** if its:
- Mean is constant over time
- Variance is constant over time
- Autocovariance does not depend on time, only on lag

In simpler terms: **the statistical properties of the series don’t change as time passes**.

---

## ❗ Why It Matters

Non-stationary data can lead to:
- **Spurious regressions**
- **Unreliable forecasts**
- **Incorrect model selection**

You usually want to **transform** non-stationary data into stationary form using techniques like **differencing** or **log transformations**.

---

## 📉 Deutsche Post Weekly Log Returns

Below is a sample plot of weekly log returns for Deutsche Post AG:

![Log Returns](images/deutsche_post_returns_plot.png)

---

## 🧪 Stationarity Tests

We use the following tests:

| Test | Null Hypothesis (H₀) | Stationary if... |
|------|----------------------|------------------|
| ADF  | Has unit root        | p-value < 0.05 ✅ |
| KPSS | Is stationary        | p-value > 0.05 ✅ |

### Test Results:

**ADF Test**
- Statistic: `-14.74`
- p-value: `< 0.001`
- ✅ Reject H₀ → Likely Stationary

**KPSS Test**
- Statistic: `0.195`
- p-value: `> 0.1`
- ✅ Fail to reject H₀ → Likely Stationary

➡️ *Both tests agree: this series is likely stationary.*

---

## 🔁 Making a Series Stationary (if it isn't)

If your series fails these tests:
- Apply **log transformation**
- Apply **differencing**: `returns.diff().dropna()`
- **Re-test** stationarity afterward

---

## 🐍 Python Example

```python
from statsmodels.tsa.stattools import adfuller, kpss

# ADF Test
adf_result = adfuller(returns)
print("ADF p-value:", adf_result[1])

# KPSS Test
kpss_result = kpss(returns, regression='c')
print("KPSS p-value:", kpss_result[1])
