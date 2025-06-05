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

## 📏 How to Test for Stationarity

We can use the following tests:

| Test | Null Hypothesis (H₀) | Stationarity if... |
|------|----------------------|--------------------|
| Augmented Dickey–Fuller (ADF) | Series has a unit root (non-stationary) | p-value < 0.05 |
| KPSS | Series is stationary | p-value > 0.05 |

![Log Returns](images/deutsche_post_log_returns_plot.png)

---

## 🐍 Python Example

Here’s how to run both ADF and KPSS tests using `statsmodels`.

```python
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.pyplot as plt

# Load your time series
df = pd.read_csv('data/deutsche_post_returns.csv')
returns = df['log_returns']

# Plot the series
returns.plot(title='Log Returns')
plt.show()

# Augmented Dickey-Fuller Test
adf_result = adfuller(returns.dropna())
print("ADF Statistic:", adf_result[0])
print("p-value:", adf_result[1])

# KPSS Test
kpss_result = kpss(returns.dropna(), regression='c')
print("KPSS Statistic:", kpss_result[0])
print("p-value:", kpss_result[1])
