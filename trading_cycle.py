import pandas as pd
import matplotlib.pyplot as plt

##########################
#Cycle analysis using a Price Detrending Technique


df = pd.read_csv('trading_view_corn_daily.csv', names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)


monthly_avg = df['Close'].resample('ME').mean()
df_new = pd.DataFrame({
    'Date': monthly_avg.index,
    'Price': monthly_avg.values,
    'Month': monthly_avg.index.month
})

df_new['Price'] = df_new['Price'].astype(float)

df_new['Price Detrended'] = df_new['Price'].diff()
print(df_new)
yearly_mean = df_new['Price'].mean()


monthly_avg_dict = df_new.groupby('Month')['Price Detrended'].mean().to_dict()
monthly_avg_dict = {month: value / yearly_mean for month, value in monthly_avg_dict.items()}

df_new['Positive'] = df_new['Price Detrended'] > 0
monthly_positive_percentage_dict = df_new.groupby('Month')['Positive'].mean().to_dict()


print(list(monthly_positive_percentage_dict.values()))
plt.plot(monthly_avg_dict.values(), linestyle='-', color='b')
plt.show()
