import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt
df = dd.read_csv('btc_tick_data_1.csv')


df = df[["timestamp", "tick_direction", "price", "index_price", "amount", "buy"]]
#df['diff'] = df['index_price'] - df['price']


df['return'] = (df['price'] - df['price'].shift(1)) / df['price'].shift(1)
df['tick_direction_shifted'] = df['tick_direction'].shift(1)
df['accumulated_buy'] = df['buy'].cumsum()/df.index

df = df.dropna()


df_train, df_test = df.random_split([0.7, 0.3])

print(df_train.head(50))


print(df_train['tick_direction_shifted'].corr(df_train['buy']).compute())
print(f"Correlation coefficient between return and tick direction is {df_train['accumulated_buy'].corr(df_train['return']).compute()}")

