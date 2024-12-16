import math
from datetime import date
import datetime
import backtrader.indicators as btind
import backtrader as bt
import backtrader.feeds as btfeed
from timeit import default_timer as timer

def main():
    start = timer()
    data = SPY(dataname = 'spy_trading_hours_data.csv')
    #data = MyHLOC(dataname = '60_minute_data_filled.csv')
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addindicator(PMARP)
    cerebro.run(maxcpus=None, optdatas = True, runonce = True)
    cerebro.plot( style = 'candlestick')
    end = timer()
    print(end - start)


class VWMA(bt.Indicator):
    lines = ('vwma',)
    params = dict(
            period = 20
    )

    plotinfo = {'plot': True,
                'subplot': False
                }

    def __init__(self):
        def calculate_vwma():
            volume_sum = 0
            vwma = 0
            for i in range(0, self.params.period):
                volume_sum += self.data.volume(-i)
            for i in range(0, self.params.period):
                vwma += (self.data.close(-i) * self.data.volume(-i))/volume_sum
            return vwma
        self.lines.vwma = bt.If(self.data.volume(0) > 0, calculate_vwma(), 0)



class PMARP(bt.Indicator):
    lines = ('pmarp',
            'pmarp_ma',)
    params = dict(
        ma_period = 20,
        ma_type = 'VWMA',
        lookback = 350,
        pmarp_ma_period = 20,
        pmarp_ma_type = 'SMA'
    )
    plotinfo = {'plot': False,
                'subplot': True,
                'plotyticks': [0, 100]}
    plotlines = dict(pmarp = dict(ls = '-'), pmarp_ma = dict(linestyle = 'solid'))



    def __init__(self):
        def calculate_percentile_init():
            sum = 0
            for i in range(1, self.params.lookback):
                 increment = bt.If(self.pmar(-i) < self.pmar(0), 1, 0)
                 sum += increment

            return (sum/self.params.lookback) * 100

        ma_dict = {
            'VWMA' : VWMA(self.data, period = self.params.ma_period),
            'SMA' : bt.ind.SMA(self.data, period = self.params.ma_period),
            'EMA': bt.ind.EMA(self.data, period = self.params.ma_period)
        }

        self.ma = ma_dict[self.params.ma_type]
        self.pmar = self.data.close(0) / self.ma(0)
        self.lines.pmarp = calculate_percentile_init()

        pmarp_ma_dict = {
            'SMA' : bt.ind.SMA(self.lines.pmarp, period = self.params.pmarp_ma_period),
            'EMA': bt.ind.EMA(self.lines.pmarp, period = self.params.pmarp_ma_period)
        }

        self.lines.pmarp_ma = pmarp_ma_dict[self.params.pmarp_ma_type]




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
