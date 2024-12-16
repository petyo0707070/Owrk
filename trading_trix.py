import pandas as pd
import numpy as np
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed


def main():
    data = SPY_TRVIEW(dataname = 'trading_view_spy_daily.csv')
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addindicator(TRIX)
    cerebro.run()
    cerebro.plot( style = 'candlestick')


class TRIX(bt.Indicator):

    lines = ('trix', 'trix_original', 'trix_original_ma')
    params = dict(
        n = 9
    )
    plotinfo = {'plot': True,
                'subplot': True,
                'plotyticks': [-10, 10]}

    plotlines = dict(trix = dict(ls = 'solid'), trix_original = dict(linestyle = '--'))


    def __init__(self):

        self.s = 2/(self.p.n + 1)
        self.e_1 = bt.ind.EMA(self.data.close, period = self.p.n)
        self.e_2 = bt.ind.EMA(self.e_1, period = self.p.n)
        self.e_3 = bt.ind.EMA(self.e_2, period = self.p.n)
        self.trix_1 = (self.e_3(0) - self.e_3(-1))/self.e_3(-1)
        self.l.trix = (self.trix_1(0) - self.trix_1(-1))*10000
        self.l.trix_original = self.trix_1(0) * 10000
        self.l.trix_original_ma = bt.ind.SMA(self.l.trix_original, period = 3)




    def next(self):
        pass
class SPY_TRVIEW(btfeed.GenericCSVData):

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