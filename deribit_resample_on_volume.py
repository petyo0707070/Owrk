import dask.dataframe as dd
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

df = pd.read_csv('btc_tick_data_1.csv', names = ['trade_sequence', 'timestamp', 'tick_direction','price', 'market_price', 'index_price', 'amount', 'buy'], header = 1)

df = df.drop(['market_price', 'index_price'], axis = 1)


def volume_bar_cum(df, m):
    aux = df.reset_index()
    cum_v = aux.amount.cumsum()  
    th = m
    buy_sell_ratio = []
    tick_direction_ratio = []
    high = []
    low = []
    idx = []

    for i, c_v in cum_v.items():

        if i % 100000==0:
            print(i)


        if c_v >= th:
            th = th + m
            idx.append(i)

            if len(idx) != 1:

                subset = aux[idx[-2]:i][['buy', 'tick_direction', 'price']]
                buy_sell_ratio.append((subset['buy'].sum())/(i - idx[-2]))
                tick_direction_ratio.append((subset['tick_direction'].sum())/(i - idx[-2]))
                high.append(subset['price'].max())
                low.append(subset['price'].min())

            else:
                subset = aux[0: i][['buy', 'tick_direction', 'price']]
                buy_sell_ratio.append(subset['buy'].sum()/i)
                tick_direction_ratio.append(subset['tick_direction'].sum()/i)
                high.append(subset['price'].max())
                low.append(subset['price'].min())
    


    output = aux.loc[idx].set_index('trade_sequence')

    output = output.reset_index(drop = True)

    output['buy_sell_ratio'] = buy_sell_ratio
    output['tick_direction_ratio'] = tick_direction_ratio
    output['low'] = low
    output['high'] = high

    return output[['timestamp','price', 'high', 'low', 'buy_sell_ratio', 'tick_direction_ratio']]\




df_new = volume_bar_cum(df, 1000000)

df_new['return_log'] = np.log(df_new['price']/df_new['price'].shift(1))

df_new.loc[:,'buy_sell_ratio'] = df_new['buy_sell_ratio'].shift(1)
df_new.loc[:,'tick_direction_ratio'] = df_new['tick_direction_ratio'].shift(1) 
df_new['open'] = df_new['price'].shift(1)



print(df_new)



print(f"Log Returns have mean: {df_new['return_log'].mean()}, standard deviation: {df_new['return_log'].std()}, skew: {df_new['return_log'].skew()} and kurtosis: {df_new['return_log'].kurtosis()}")
print(f"Correlation between previous buy_sell ratio and current return is {df_new['buy_sell_ratio'].corr(df_new['return_log'])}")
print(f"Correlation between previous tick_direction_ratio and current return is {df_new['tick_direction_ratio'].corr(df_new['return_log'])}")

df_new.to_csv("btc_sampled_tick_data.csv", index = True)



