import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.graphics.tsaplots import plot_predict
import mplfinance as mpf
import numpy as np


df = pd.read_csv("trading_view_spy_1_and_2_converted.csv", names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
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

print(len(state_means))



add_plot = mpf.make_addplot(state_means, color='blue', width=1)
mpf.plot(df, type='candle', style='charles', addplot=add_plot, title='OHLC Chart', ylabel='Price')

plt.show()