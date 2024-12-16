import pandas as pd
from datetime import date
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
import numpy as np
import sys 

###################################################################
#Based on the Donchian Channels very good for Corn and could be used for currencies
####################################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        df['volume'] = 1
        df_training = df[0:]
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Days, compression = timeframe)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)


        cerebro.adddata(data)
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
        results = cerebro.run(maxcpus=None, optdatas = True, runonce = True)

    # Get analyzer results
        sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
        returns = results[0].analyzers.returns.get_analysis()
        max_drawdown = results[0].analyzers.drawdown.get_analysis()
        print("Sharpe Ratio:", sharpe_ratio['sharperatio'])
        print("Returns:", returns['rtot'])
        print("Max Drawdown:", max_drawdown['max']['drawdown'])
        cerebro.plot(style = 'candlestick')

    run_strategy('trading_view_eur_usd_daily.csv', timeframe = 1, leverage = 100)

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
    lines =('trix_original', 'trix_original_ma')
    params = dict(
        n = 9
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

        self.slow_ma = bt.ind.SMA(self.data, period = 20)
        self.fast_ma = bt.ind.SMA(self.data, period = 5)
        self.atr = bt.ind.ATR(self.data, period = 20)


        self.order= None

        self.buy_signal = bt.And(self.data.close(0) > self.fast_ma(-1) + self.atr(-1), self.data.close(0) > self.slow_ma(-1) + self.atr(-1) )
        self.sell_signal = bt.And(self.data.close(0) < self.fast_ma(-1) - self.atr(-1), self.data.close(0) < self.slow_ma(-1) - self.atr(-1))

        self.long_exit = bt.Or(self.data.close(0) < self.fast_ma(-1) - self.atr(-1), self.data.close(0) < self.slow_ma(-1) - self.atr(-1))
        self.short_exit = bt.Or(self.data.close(0) > self.fast_ma(-1) + self.atr(-1), self.data.close(0) > self.slow_ma(-1) + self.atr(-1))


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

        if self.first_close is None:
            self.first_close = self.data.close[0]



        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
# This executes if we are not already in the market to enter long
        if not self.position:
            # Enters Long
            if self.buy_signal:

                self.order = self.buy()

                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'
                self.trade_array.append([])
                self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                self.trade_array[len(self.trade_array) - 1].append(1)
                self.trade_array[len(self.trade_array) - 1].append(len(self))
                self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())

            # Enter Short
            if self.sell_signal:


                # Always wait for the exact limit to be hit even if there is a gap
                self.order = self.sell()

                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'
                self.trade_array.append([])
                self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                self.trade_array[len(self.trade_array) - 1].append(2)
                self.trade_array[len(self.trade_array) - 1].append(len(self))
                self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())


        if self.position.size > 0 and self.long_exit:

            # Closes the position
            self.close()
            self.log(f"Trade closed at index {len(self)}")



        if self.position.size < 0 and self.short_exit:

            # Closes the position
            self.close()
            self.log(f"Trade closed at index {len(self)}")




    def stop(self):
        pass

      

if __name__ == "__main__":
    main()
