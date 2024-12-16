import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
from trading_pivot_point import PIVOT
import sys
import datetime


##########################################################
#Linda Rashcke's ORB Strategy, this is a simplified version and worlds quite well
##########################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        #df =  df[ df['volume'] != 0]
        #df = df[df['high'] != df['low']]
        #df['openinterest'] = pd.factorize(df.index.hour)[0]
        df_training = df[0:]
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Minutes, compression = timeframe)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)
        cerebro.adddata(data)


        df_1 = pd.read_csv('futures_es_daily.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'])
        data1 = bt.feeds.PandasData(dataname = df_1,timeframe = bt.TimeFrame.Days, compression = 1 )
        cerebro.adddata(data1)
        print(df_1)





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

    run_strategy('futures_es_15m.csv', timeframe = 15, leverage = 200)

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
    lines =('vwmacd',
            'vwmacd_signal_line',)
    params = dict(
        buffer = 0.006
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
        self.short_stoploss = None
        self.bar_entered = None

        self.high_morning_range = None
        self.low_morning_range = None

        self.daily_roc = (self.data1.close(-1) - self.data1.close(-2))/self.data1.close(-2)

        self.up = bt.Max(self.daily_roc(0) - self.daily_roc(-1), 0)
        self.down = bt.Max(self.daily_roc(-1) - self.daily_roc(0), 0)
        self.maup = bt.ind.MovAv.Smoothed(self.up, period = 3, plot = False)
        self.madown = bt.ind.MovAv.Smoothed(self.down, period =3, plot = False)

        self.roc_rs = self.maup(0) / self.madown(0)
        self.roc_rsi = 100 *(self.roc_rs(0)/(1 + self.roc_rs(0)))


        self.signal = bt.And(self.data.high(0) - self.data.low(0) < self.data.high(-1) - self.data.low(-1), self.data.high(0) - self.data.low(0) < self.data.high(-2) - self.data.low(-2), self.data.high(0) - self.data.low(0) < self.data.high(-3) - self.data.low(-3))
        self.sell_signal = bt.And(self.data.high(0) - self.data.low(0) < self.data.high(-1) - self.data.low(-1), self.data.high(0) - self.data.low(0) < self.data.high(-2) - self.data.low(-2), self.data.high(0) - self.data.low(0) < self.data.high(-3) - self.data.low(-3))
        


        self.long_exit = 0#bt.And(self.va(0) < 0)
        self.short_exit = 0#bt.And(self.va(0) > 0)


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
        #self.trade_array[len(self.trade_array) - 1].append(trade.pnl / self.broker.get_value() * 100)
        self.order = None


    def next(self):
        print(len(self))

        # This records the high and low of the morning trading range, which I have adjusted to be uniformally at 10:30
        if self.data.datetime.time() == datetime.time(10, 15):
            self.high_morning_range = max(self.data.high[0], self.data.high[-1], self.data.high[-2], self.data.high[-3])
            self.low_morning_range = min(self.data.low[0], self.data.low[-1], self.data.low[-2], self.data.low[-3])
        

        # This checks that we are in the trading session and that the 3-period RSI of 1-period ROC is below 30, if there is not an open position, a buy limit is placed on the high 
        if self.data.datetime.time() >= datetime.time(10, 15) and self.data.datetime.time() <= datetime.time(15, 45) and self.roc_rsi[0] <= 30 and self.position.size == 0:
            self.buy(data = self.data0, price = self.high_morning_range , exectype= bt.Order.Stop, valid = datetime.timedelta(minutes = 15))
            #self.buy(data = self.data0, price = self.high_morning_range , plimit = self.high_morning_range ,exectype= bt.Order.StopLimit, valid = datetime.timedelta(1))

        # This places a stop-loss if we are during the trading session and in a long a SL at the bottom of the morning range
        if self.data.datetime.time() >= datetime.time(10, 15) and self.data.datetime.time() <= datetime.time(15, 45) and self.position.size > 0:#self.position.size > 0:#
            self.sell(data = self.data0, price = self.low_morning_range, exectype= bt.Order.StopLimit, plimit= self.low_morning_range, valid = datetime.timedelta(minutes= 15))



 
        if self.first_close is None:
            self.first_close = self.data.close[0]


        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
# This executes if we are not already in the market to enter long








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
