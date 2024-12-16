import pandas as pd
import matplotlib.pyplot as plt
import backtrader.analyzers as btanalyzers
import backtrader as bt
import numpy as np
import backtrader.feeds as btfeed
import networkx as nx
from ts2vg import HorizontalVG, NaturalVG
from trading_hawkes_process import Hawkes
from trading_pivot_point import PIVOT


##########################################################
#Trying VGs 
##########################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        #df =  df[ df['volume'] != 0]
        #df = df[df['high'] != df['low']]
        df_2 = pd.read_csv('futures_es_daily.csv',header = None, index_col = 0, parse_dates=True, names = ['time', 'open', 'high', 'low', 'close'])
        df_training = df[500:]
        print(df_training)
        df_training_2 = df_2[499:]
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Days, compression = timeframe)
        #data_2 = bt.feeds.PandasData(dataname = df_training_2, timeframe = bt.TimeFrame.Days, compression = 1)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)


        cerebro.adddata(data)
        #cerebro.adddata(data_2)
        cerebro.addstrategy(Quant_Strategy_5)
        cerebro.broker = bt.brokers.BackBroker(cash = 100000)
        cerebro.broker.setcommission(commission=0.0001, leverage = 10.0)
        cerebro.addsizer(bt.sizers.PercentSizer, percents=leverage)


        cerebro.addobserver(bt.observers.DrawDown)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.Returns)
        cerebro.addanalyzer(bt.analyzers.DrawDown)
        cerebro.addobserver(bt.observers.Benchmark, data =data, timeframe = bt.TimeFrame.NoTimeFrame)
    # Run the strategy
        results = cerebro.run(maxcpus=None, optdatas = True, runonce = False)

    # Get analyzer results
        sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
        returns = results[0].analyzers.returns.get_analysis()
        max_drawdown = results[0].analyzers.drawdown.get_analysis()
        print("Sharpe Ratio:", sharpe_ratio['sharperatio'])
        print("Returns:", returns['rtot'])
        print("Max Drawdown:", max_drawdown['max']['drawdown'])
        cerebro.plot(style = 'candlestick')

    run_strategy('trading_view_btc_daily_2017-.csv', timeframe = 60, leverage = 200)

# Class to get a datafeed of a generic CSV data set

class MyHLOC(btfeed.GenericCSVData):

  params = (
    ('nullvalue', 0.0),
    ('dtformat', ('%Y-%m-%d')),
    ('tmformat', ('%H.%M.%S')),
    ('datetime', 0),
    ('open', 1),
    ('high', 2),
    ('low', 3),
    ('close', 4),
    ('volume', -1),
    ('openinterest', -1)
)

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
        ('datetime', None),

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




class Quant_Strategy_5(bt.Strategy):
    lines =('kama',)
    params = dict(
        volume_lookback = 120,
        hawkes_lookback = 60
                        )
    plotinfo = {
                'subplot': True,
                'plot': True
                }

    def log(self, txt, dt=None, doprint=False):

        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.trade_array =[]
        self.return_array = []
        self.long_bar = None
        self.first_close = None
        self.order = None

        self.long_stoploss = 0
        self.short_stoploss = 1000000000


        #self.dummy = bt.ind.SMA(self.data0.close, period = self.p.lookback, plot = False)
        self.ema_21 = bt.ind.EMA(self.data0.close, period = 21, plot= True, plotname = '21 EMA')
        self.ema_55 = bt.ind.EMA(self.data0.close, period = 55, plot= True, plotname = '55 EMA')

        # Hawkes
        self.hawkes = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = True).hawkes
        self.hawkes_95 = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = False).hawkes_95
        self.hawkes_5 = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = False).hawkes_5
        self.last_close_below = 0
        self.first = 1

        #PIVOT
        self.pivot_high = PIVOT(self.data0).resistance
        self.pivot_low = PIVOT(self.data0).support

        # RSI
        self.up = bt.Max(self.data0.close(0) - self.data0.close(-1), 0)
        self.down = bt.Max(self.data0.close(-1) - self.data0.close(0), 0)
        self.maup = bt.ind.MovAv.Smoothed(self.up, period = 14, plot = False)
        self.madown = bt.ind.MovAv.Smoothed(self.down, period =14, plot = False)
        self.rs = self.maup(0) / self.madown(0)
        self.rsi = 100 *(self.rs(0)/(1 + self.rs(0)))


        self.atr = bt.ind.ATR(self.data0, period = 14, plot = False )


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Nothing to be done the order is already entered to the broker
            return

        # Checks if an order is completed as the broker could reject it if not enough margin
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.5f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))
                self.buyprice = order.executed.price

            elif order.issell():

                self.log('SELL EXECUTED, Price: %.5f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))


        # Says that for some reason or another the given order did not go through
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected, status {order.getstatusname()}')
        # This means that there is not a pending order and allows for new orders
        self.order = None

    def notify_trade(self, trade):
        #If the trade is not closed we can not do a thing
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
            (trade.pnl, trade.pnlcomm))
         #Saves the PNL of every trade into array as a %
        self.trade_array[len(self.trade_array) - 1].append(trade.pnl / self.broker.get_value() * 100)
        self.order = None


    def next(self):
        # Ensures we start with all the values preloaded, that is due to iniitialization in next and multiple scripts stacked on top of eachother
        if np.isnan(self.hawkes_95[-1]):
            return
        

        if self.first_close is None:
            self.first_close = self.data.close[0]
            return
        
        if self.hawkes[0] < self.hawkes_5[0]:
            self.last_close_below = self.data.close[0]

        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
# This executes if we are not already in the market to enter long

        if self.data0.close[-1] < self.pivot_high[-1] and self.data0.close[0] > self.pivot_high[0] and self.position.size == 0 and self.rsi >= 60: 
            self.order = self.buy(data = self.data0)
            # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())

            self.long_stoploss = self.data0.close[0] - self.atr 


        if self.data0.close[-1] > self.pivot_low[-1] and self.data0.close[0] < self.pivot_low[0] and self.rsi <= 40 and self.position.size == 0:  


            self.order = self.sell(data = self.data0)

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())
            #self.short_stoploss = self.data.close[0] + 2 * (self.data.high[0] - self.data.close[0])

            self.short_stoploss = self.data0.close[0] + self.atr

            
        
        if self.position.size > 0 and self.hawkes[-1] > self.hawkes_5[-1] and self.hawkes[0] < self.hawkes_5[0]:
            # Closes the position
            self.close(data = self.data0)

            self.long_stoploss = 0


        if self.position.size < 0 and self.hawkes[-1] > self.hawkes_5[-1] and self.hawkes[0] < self.hawkes_5[0]:
            self.short_stoploss = None 
            self.close(data = self.data0)

            self.short_stoploss = 1000000000

        if self.position.size > 0 and self.data.close[0] < self.long_stoploss:
            self.close(data = self.data0)
            self.long_stoploss = 0

        if self.position.size < 0 and self.data.close[0] > self.short_stoploss:
            self.close(data = self.data0)
            self.short_stoploss = 1000000000



    def stop(self):
        pass
        #df = pd.DataFrame(self.rsi_array, columns = ['RSI'])
        #sns.histplot(df['RSI'], kde = True)
        #print(f"Mean is {df['RSI'].mean()}")
        #print(f"Stand Deviation is {df['RSI'].std()}")
        #print(f"Skewness is {df['RSI'].skew()}")
        #print(f"Kurtosis is {df['RSI'].kurt()}")
        #plt.show()

      



if __name__ == "__main__":
    main()
