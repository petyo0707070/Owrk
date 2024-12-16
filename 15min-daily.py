import pandas as pd

df = pd.read_csv('futures_es_15m.csv', names = ['time', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")
df.set_index('time', inplace=True)


df = df.resample('D').agg({
    'open': 'first',   # Take the first value in the day as the open
    'high': 'max',     # Take the maximum value in the day as the high
    'low': 'min',      # Take the minimum value in the day as the low
    'close': 'last'    # Take the last value in the day as the close
})
print(df)
df.dropna(inplace=True)

print(df)
df.to_csv('futures_es_daily.csv', header = False, index = True)
