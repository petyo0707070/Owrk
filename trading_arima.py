import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from pmdarima.arima.utils import ndiffs
from pykalman import KalmanFilter
import mplfinance as mpf



df = pd.read_csv("trading_view_spy_daily.csv", names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Close Detrended'] = df['Close'].diff()

df= df.dropna(subset = ['Close Detrended'])


ndiffs(df['Close'], test ='adf')


#Specifying the model
model = ARIMA(df['Close'], order=(1,1,1))
model_fit = model.fit()

# Summary of the model fit
print(model_fit.summary())

#Getting the residuals
residuals = model_fit.resid
residuals.plot()
plt.show()
residuals.plot(kind = 'kde')
plt.show()
print(residuals.describe())

# Compare Actual vs Predicted
plot_predict(
    model_fit,
    start = 2500,
    end=2550,
    dynamic= False,
)
plt.show()
# Stationarity Test Dickey Fuller Test /For d
#from statsmodels.tsa.stattools import adfuller
#result = adfuller(df['Close Detrended'])
#print(f"P-value is {round(result[1], 4)} and test statistic {result[0]}")


#For p Try 6
#from statsmodels.graphics.tsaplots import plot_pacf
#plot_pacf(df['Close Detrended'], lags = 10)
#plt.show()

#For q Try 6
#from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(df['Close Detrended'], lags = 10)
#plt.show()
