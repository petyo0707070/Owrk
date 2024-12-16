import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
import mplfinance as mpf


df = pd.read_csv("trading_view_spy_daily.csv", names = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)


print(df)
# split into train and test sets
X = df['Close'].values
Y = df['High'].values
Z = df['Low'].values
size = int(len(X) * 0.66)
train, test = X[0:3000], X[3000:3200]
train_high, test_high = Y[0:3000], Y[3000:3200]
train_low, test_low = Z[0:3000], Z[3000:3200]

history = [x for x in train]
history_high = [y for y in train_high]
history_low = [z for z in train_low]

predictions = list()
predictions_high = list()
predictions_low = list()

# walk-forward validation
for t in range(len(test)):
	# For close
	model = ARIMA(history, order=(1,1,1))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	
    # For high
	model_high = ARIMA(history_high, order = (1,1,1))
	model_fit_high = model_high.fit()
	output_high = model_fit_high.forecast()
	yhat_high = output_high[0]
	predictions_high.append(yhat_high)
	
    # For low
	model_low = ARIMA(history_low, order = (1,1,1))
	model_fit_low = model_low.fit()
	output_low = model_fit_low.forecast()
	yhat_low = output_low[0]
	predictions_low.append(yhat_low)

	obs = test[t]
	obs_high = test_high[t]
	obs_low = test_low[t]

	history.append(obs)
	history_high.append(obs_high)
	history_low.append(obs_low)

	if t % 100 == 0:
		print(t)
	#print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
rmse_high = sqrt(mean_squared_error(test_high, predictions_high))
rmse_low = sqrt(mean_squared_error(test_low, predictions_low))


print('Test RMSE: %.3f' % rmse)
print('Test RMSE High: %.3f' % rmse_high)
print('Test RMSE Low: %.3f' % rmse_low)


# plot forecasts against actual outcomes
fig, (ax1, ax2) = plt.subplots(2,1)

mpf.plot(df[3000:3200], type = 'candle', ax = ax1, volume = False, style = 'classic' )
ax1.plot(predictions, color='red')
ax1.plot(predictions_high, color='blue')
ax1.plot(predictions_low, color='green')


plt.show()