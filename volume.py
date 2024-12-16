import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('russel_1hr.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )

df['return'] = df['close'].shift(-1) / df['close'] - 1
df['hour_index'] = pd.factorize(df.index.hour)[0]
df['day_index'] = df.index.dayofweek

df = df[df['hour_index'] == 0]
df = df[df['day_index'] == 0]

#df['volume_indexed'] = df.groupby('day_index')['volume'].apply(lambda x: x.rolling(window=20, min_periods=1).mean())

df['volume_indexed'] = df['volume']/(df['volume'].rolling(window=100).mean()) * 100
#df = df[df['volume_indexed'] > 200]
print(df)



def histogram_volume_indexed():
    sns.histplot(df['volume_indexed'])
    plt.show()

def volume_return_scatterplot():
    df.dropna(subset=['return'], inplace=True)
    sns.regplot(x='volume_indexed', y='return', data=df, scatter_kws={'s': 50}, line_kws={"color":"r","alpha":0.7})
    print(f"Volume_Index < 100, return: {round(df.loc[(df['volume_indexed'] >= 0) & (df['volume_indexed'] <= 100), 'return'].mean(),4)}, standard dev: {round(df.loc[(df['volume_indexed'] >= 0) & (df['volume_indexed'] <= 100), 'return'].std(),4)}")
    print(f"Volume_Index 100-200, return: {round(df.loc[(df['volume_indexed'] >= 100) & (df['volume_indexed'] <= 200), 'return'].mean(),4)}, standard dev: {round(df.loc[(df['volume_indexed'] >= 100) & (df['volume_indexed'] <= 200), 'return'].std(),4)}")
    print(f"Volume_Index 200-300, return: {round(df.loc[(df['volume_indexed'] >= 200) & (df['volume_indexed'] <= 300), 'return'].mean(),4)}, standard dev: {round(df.loc[(df['volume_indexed'] >= 200) & (df['volume_indexed'] <= 300), 'return'].std(),4)}")
    print(f"Volume_Index > 300, return: {round(df.loc[(df['volume_indexed'] >= 300) & (df['volume_indexed'] <= 1000), 'return'].mean(),4)}, standard dev: {round(df.loc[(df['volume_indexed'] >= 300) & (df['volume_indexed'] <= 1000), 'return'].std(),4)}")
    plt.show()



volume_return_scatterplot()


def detect_number_of_outliers():
    def detect_outliers_iqr(data):
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (data < lower_bound) | (data > upper_bound)

    df['volume_bin'] = pd.qcut(df['volume_indexed'], q=4)  # Dividing into 4 quantile-based bins
    df['is_outlier'] = df.groupby('volume_bin')['return'].transform(detect_outliers_iqr)
    outlier_counts = df.groupby('volume_bin')['is_outlier'].sum()
    plt.figure(figsize=(10, 6))
    outlier_counts.plot(kind='bar', color='skyblue')
    plt.title('Number of Outliers in Return Next 5 Rows across Volume Indexed Intervals')
    plt.xlabel('Volume Indexed Interval')
    plt.ylabel('Number of Outliers')
    plt.show()

#detect_number_of_outliers()

