import pandas as pd
import csv

df_array = []

with open('es-15m.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')

    for row in reader:
        row[0] = row[0] + ' ' +row[1]
        del row[1]
        df_array.append(row)


print(df_array)
df = pd.DataFrame(df_array, columns = ['time', 'open', 'high', 'low', 'close', 'volume'])
df['time'] = pd.to_datetime(df['time'], format="%d/%m/%Y %H:%M:%S")
#df['time'] = df['time'].dt.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='NaT')
def adjust_for_dst(row):
    if row['time'].dst() != pd.Timedelta(0):
        return row['time'] + pd.Timedelta(hours=1)
    return row['time']

# Apply the adjustment
df['time'] = df.apply(adjust_for_dst, axis=1)
df.set_index('time', inplace= True)
print(df)
df.to_csv('futures_es_15m.csv', header = False, index = True)
