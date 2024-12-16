import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('futures_es_15m.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
columns_to_log = ['open', 'high', 'low', 'close']
df[columns_to_log] = df[columns_to_log].apply(np.log)
df['close-open'] = df['close'] - df['open']
df['high-open'] = df['high'] - df['open']
df['low-open'] = df['low'] - df['open']
df['open-close(-1)'] = df['open'] - df['close'].shift(-1)

columns_to_shuffle = ['close-open', 'high-open', 'low-open', 'open-close(-1)']
df[columns_to_shuffle] = df[columns_to_shuffle].apply(np.random.permutation)

first_open = df['open'].iloc[0]
df['open'] = df['open'].shift(1) + df['open-close(-1)']
df['open'].iloc[0] = first_open
df['high'] = df['open'] + df['high-open']
df['close'] = df['open'] + df['close-open']
df['low'] = df['open'] + df['low-open']

df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(np.exp).round(2)
print(df)

df['close'].plot(kind='line', title='ES 15min futures', xlabel='Index', ylabel='Values')
plt.show()

