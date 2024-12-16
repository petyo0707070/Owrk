import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_predict
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from pmdarima.arima.utils import ndiffs
from pykalman import KalmanFilter
import mplfinance as mpf

df = pd.read_csv("trading_view_spy_daily.csv", names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])

df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)

kf = KalmanFilter(transition_matrices = [1], 
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance = 1,
                  transition_covariance = .01
                  )

state_means, _ = kf.filter(df['Close'])
 
df['Kalman'] = state_means
df['Kalman Detrended'] = df['Kalman'].diff()
df= df.dropna()


print(df)



ndiffs(df['Kalman'], test ='adf')


#Specifying the model
model = ARIMA(df['Kalman'], order=(1,1,1))
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
#result = adfuller(df['Kalman Detrended'])
#print(f"P-value is {round(result[1], 4)} and test statistic {result[0]}")


#For p Try 6
#from statsmodels.graphics.tsaplots import plot_pacf
#plot_pacf(df['Kalman Detrended'], lags = 10)
#plt.show()

#For q Try 18
#from statsmodels.graphics.tsaplots import plot_acf
#plot_acf(df['Kalman Detrended'], lags = 20)
#plt.show()
