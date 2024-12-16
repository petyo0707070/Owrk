import pandas as pd

df = pd.read_csv('futures_es_daily.csv',header = None, index_col = 0, parse_dates=True, names = ['time', 'open', 'high', 'low', 'close'])
df = df[499:]
print(df)

df_2 = pd.read_csv('trading_view_spy_1_and_2_converted.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
print(df_2)