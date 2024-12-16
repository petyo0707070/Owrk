import math
from datetime import date
import datetime
import backtrader.indicators as btind
import backtrader as bt
import pandas
import backtrader.feeds as btfeed
from timeit import default_timer as timer
import numpy as np


def main():
    start = timer()
    #dataframe = pandas.read_csv('60_minute_data_filled.csv',
                                #skiprows=0,
                                #header=0,
                                #parse_dates=True,
                                #index_col=0)
    #data = bt.feeds.PandasData(dataname = dataframe)
    #data = MyHLOC(dataname = '60_minute_data_filled.csv')
    data = SPY(dataname = 'spy_trading_hours_data.csv')
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addindicator(PIVOT)
    cerebro.run(maxcpus=None, optdatas = True, runonce = True)
    end=timer()
    print(f"Elapsed time is {end - start}")
    cerebro.plot(style = 'candlestick')


class PIVOT(bt.Indicator):
    lines = ('resistance', 'support',)

    params = dict(
        bars_lookback = 15
    )

    plotinfo = {'plot': True,
                'subplot': False
                }

    def __init__(self):
        def find_pivot_high():

            #This looks to the left and finds the bar with the highest high and its index
            for i in range(-2 * self.params.bars_lookback, - self.params.bars_lookback, 1):
                if i == -2 * self.params.bars_lookback:
                    highest_left_high = 0
                    index_highest_left = 0#-2 * self.params.bars_lookback
                    lowest_left_low = 1000000000000000000000000000
                else:
                    index_highest_left = bt.If(self.data.high(i) > highest_left_high, i, index_highest_left)
                    highest_left_high = bt.If(self.data.high(i) > highest_left_high, self.data.high(i), highest_left_high)
                    lowest_left_low = bt.If(self.data.low(i) < lowest_left_low, self.data.low(i), lowest_left_low)


            for k in range(- self.params.bars_lookback + 1, 0, 1):
                if k == - self.params.bars_lookback + 1:
                    highest_right_high = 0
                    index_highest_right = - self.params.bars_lookback + 1
                    lowest_right_low = 1000000000000000000000000000
                else:
                    index_highest_right = bt.If(self.data.high(k) > highest_right_high, k, index_highest_right)
                    highest_right_high =  bt.If(self.data.high(k) > highest_right_high, self.data.high(k),highest_right_high)
                    lowest_right_low =  bt.If(self.data.low(k) < lowest_right_low, self.data.low(k), lowest_right_low)

                # Sees if the criterua for a pivot high are fullfileed
                reference_bar_higher_left = bt.If(self.data.high(-self.params.bars_lookback) > highest_left_high, 1, 0)
                reference_bar_higher_right = bt.If(self.data.high(-self.params.bars_lookback) > highest_right_high, 1, 0)

                # Checks if the criteria for a pivot low are fullfilled
                reference_bar_lower_left = bt.If(self.data.low(-self.params.bars_lookback) < lowest_left_low, 1, 0)
                reference_bar_lower_right = bt.If(self.data.low(-self.params.bars_lookback) < lowest_right_low, 1, 0)

                # Is it a pivot high or low or not
                pivot_high = bt.And(reference_bar_higher_left, reference_bar_higher_right)
                pivot_low = bt.And(reference_bar_lower_left, reference_bar_lower_right)

            return pivot_high, pivot_low


        self.pivot_high, self.pivot_low = find_pivot_high()
        self.pivot_high_array = []
        self.pivot_low_array = []



    def next(self):
        # Checks if we actually have a pivot high if so we assign a new value to resistance if not it stays the same
        if self.pivot_high[0] == 1:
            self.lines.resistance[0] = self.data.high[-self.params.bars_lookback]
            # Saves all unique resistances in an array
            self.pivot_high_array.append(self.data.high[-self.params.bars_lookback])
        else:
            self.lines.resistance[0] = self.lines.resistance[-1]

        # Checks of we actually have a pivot low if so we assign a new value to support if not unchanged
        if self.pivot_low[0] == 1:
            self.lines.support[0] = self.data.low[-self.params.bars_lookback]
            #Saves all unique supports in an temp_array
            self.pivot_low_array.append(self.data.low[-self.params.bars_lookback])
        else:
            self.lines.support[0] = self.lines.support[-1]







class MyHLOC(btfeed.GenericCSVData):

  params = (
    ('nullvalue', 0.0),
    ('dtformat', ('%Y-%m-%d %H:%M:%S')),
    ('tmformat', ('%H:%M:%S')),
    ('datetime', 0),
    ('open', 1),
    ('high', 2),
    ('low', 3),
    ('close', 4),
    ('volume', 5),
    ('openinterest', -1)
)

class SPY(btfeed.GenericCSVData):

  params = (
    ('nullvalue', 0.0),
    ('dtformat', ('%d/%m/%Y %H:%M:%S')),
    ('tmformat', ('%H:%M:%S')),
    ('datetime', 0),
    ('open', 1),
    ('high', 2),
    ('low', 3),
    ('close', 4),
    ('volume', 5),
    ('openinterest', -1)
)

class PandasCSV(btfeed.DataBase):

    params = (
        # Possible values for datetime (must always be present)
        #  None : datetime is the "index" in the Pandas Dataframe
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('datetime', 0),

        # Possible values below:
        #  None : column not present
        #  -1 : autodetect position or case-wise equal name
        #  >= 0 : numeric index to the colum in the pandas dataframe
        #  string : column name (as index) in the pandas dataframe
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', None),
    )


if __name__ == '__main__':
    main()
