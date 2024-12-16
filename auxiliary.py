import csv
import pandas as pd

date = []
open_array = []
high = []
low = []
close = []
volume = []
with open('sp500_monthly.csv', mode ='r') as file:
    csv_reader = csv.reader(file)

    for row in csv_reader:
        row = row[0].split(';')
        row[0] = row[0] + " " + row[1]
        del(row[1])
        date.append(row[0])
        open_array.append(row[1])
        high.append(row[2])
        low.append(row[3])
        close.append(row[4])
        volume.append(row [5])
    
data = {'date': date,
        'open': open_array,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume}
df = pd.DataFrame(data)
df = df.drop(index=[0, 495, 496, 497,498])

print(df)

df.set_index('date', inplace = True)

df.to_csv('sp_500_monthly.csv', header = False, index = True)