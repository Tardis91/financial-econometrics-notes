# ARMA Models

ARMA models combine Autoregressive (AR) and Moving Average (MA) components.

## The ARMA(p, q) Model

$X_t = c + \sum_{i=1}^p \phi_i X_{t-i} + \sum_{j=1}^q \theta_j \varepsilon_{t-j} + \varepsilon_t$

Where:
- $X_t$ is the time series value at time $t$
- $p$ is the order of the AR part
- $q$ is the order of the MA part

## Python Example

```python
import pandas as pd
import statsmodels.api as sm

data = pd.read_csv("returns.csv")
returns = data['log_returns']

model = sm.tsa.ARMA(returns, order=(1,1)).fit()
print(model.summary())
