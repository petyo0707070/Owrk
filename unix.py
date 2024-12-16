import pandas as pd

df = pd.read_csv('trading_view_btc_1hr.csv')
print(df)
df['time'] = pd.to_datetime(df['time'], unit = 's')
#df.drop('EMA', axis = 1, inplace = True)
df.set_index('time', inplace = True)
print(df)
df.to_csv('trading_view_btc_1hr.csv', header = False, index = True)