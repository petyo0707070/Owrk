from datetime import date
import backtrader as bt
import backtrader.feeds as btfeed
import backtrader.indicators as btind
from timeit import default_timer as timer
import pandas as pd
import math
import numpy as np
import sys

def main():
    df = pd.read_csv('trading_view_btc_daily_2017-.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
    cerebro = bt.Cerebro()
    df_training = df[0:1000]
    print(df_training)
    
    data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Minutes, compression = 60)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)


    start = timer()
    cerebro.adddata(data)
    cerebro.addindicator(Hawkes)
    cerebro.run(maxcpus=None, optdatas = True, runonce = True)
    cerebro.plot( style = 'candlestick')
    end = timer()
    print(end - start)


class ATR(bt.Indicator):

    lines = ('log_atr',
             'norm_range',)
    params = dict(volume_lookback = 60)
    plotinfo = {'plot': True,
                'subplot': True}

    def __init__(self):
        self.first = 1
        self.log_high = []
        self.log_close = []
        self.log_low = []
        self.true_range = []
    
    def next(self):
        if len(self) == self.p.volume_lookback:
            self.log_high = [math.log(self.data.high[-i]) for i in range(self.p.volume_lookback -1, -1 ,-1)]
            self.log_close = [math.log(self.data.close[-i]) for i in range(self.p.volume_lookback -1 , -1, -1)]
            self.log_low = [math.log(self.data.low[-i]) for i in range(self.p.volume_lookback -1 , -1, -1)]
            for i in range (self.p.volume_lookback , 1, -1):
                tr_1 = self.log_high[-i] - self.log_low[-i]
                tr_2 = abs(self.log_high[-i] - self.log_close[-i + 1])
                tr_3 = abs(self.log_low[-i] - self.log_close[-i + 1])
                true_ran = max(tr_1, tr_2, tr_3)
                self.true_range.append(true_ran)
            
            self.l.log_atr[0] = sum(self.true_range)/(self.p.volume_lookback)
            self.l.norm_range[0] = (self.log_high[0] - self.log_low[0])/self.l.log_atr[0]

        
        
        elif len(self) > self.p.volume_lookback:
            self.log_high.pop(0)
            self.log_high.append(math.log(self.data.high[0]))

            self.log_close.pop(0)
            self.log_close.append(math.log(self.data.close[0]))

            self.log_low.pop(0)
            self.log_low.append(math.log(self.data.low[0]))

            tr_1 = self.log_high[-1] - self.log_low[-1]
            tr_2 = abs(self.log_high[-1] - self.log_close[-2])
            tr_3 = abs(self.log_low[-1] - self.log_close[-2])
            true_ran = max(tr_1, tr_2, tr_3)

            self.true_range.append(true_ran)
            self.true_range.pop(0)

            self.l.log_atr[0] = sum(self.true_range)/(self.p.volume_lookback)
            self.l.norm_range[0] = (self.log_high[0] - self.log_low[0])/self.l.log_atr[0]


    

class Hawkes(bt.Indicator):
    lines = ('hawkes',
            'hawkes_95',
            'hawkes_5')
    params = dict(
            volume_lookback = 180,
            kappa = 0.5,
            hawkes_lookback = 90,
    )
    plotinfo = {'plot': True,
                'subplot': True}
                
    plotlines = dict(hawkes_q95 = dict(ls = '-'), hawkes_q5 = dict(linestyle = 'solid'))



    def __init__(self):

        # Calculate the ATR
        self.norm_range = ATR(self.data, volume_lookback = self.p.volume_lookback).norm_range
        self.alpha = np.exp(self.p.kappa)
        self.data_array = []

        self.first = 1


    
    def next(self):

        if len(self) < self.p.volume_lookback:
            pass

        else:

            if self.first == 1:
                self.l.hawkes[0] = self.norm_range[0]
                self.first = 0
            else:
                self.l.hawkes[0] = (self.l.hawkes[-1] * self.alpha + self.norm_range[0]) * self.p.kappa
        

            if len(self) == self.p.volume_lookback:
                self.data_array = np.array([self.l.hawkes[-i] for i in range (self.p.hawkes_lookback -1, -1, -1)])
                self.l.hawkes_95[0] = np.quantile(self.data_array, 0.975)
                self.l.hawkes_5[0] = np.quantile(self.data_array, 0.025)

            elif len(self) > self.p.hawkes_lookback + self.p.volume_lookback:
                self.data_array = np.delete(self.data_array, 0)
                self.data_array = np.append(self.data_array, self.l.hawkes[0])
                self.l.hawkes_95[0] = np.quantile(self.data_array, 0.95)
                self.l.hawkes_5[0] = np.quantile(self.data_array, 0.05)


        
    
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

if __name__ == '__main__':
    main()
