import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
import sys 


##########################################################
# Some good results with equities and oil, but very prone to blowing up
##########################################################




def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        df['volume'] = 0
        df = df[df['high'] != df['low']]
        df_training = df[0:]
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Days, compression = timeframe)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)


        cerebro.adddata(data)
        cerebro.addstrategy(Quant_Strategy_5, lookback = 2)
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

    run_strategy('trading_view_eur_usd_daily.csv', timeframe = 1, leverage = 200)

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
    lines =('rsi',)
    params = dict(
        period = 9,
        lookback = 1
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
        self.trade_length = 0

        self.bp = self.data.high(0) - self.data.open(0)
        self.sp = self.data.close(0) - self.data.low(0)

        self.drf = (self.bp + self.sp)/(2*(self.data.high(0) - self.data.low(0)))

        self.drf_smoothened = bt.ind.EMA( self.drf, period = 2)

        self.drf_plot = bt.ind.SMA(self.drf_smoothened, period = 1, plotname = 'DRF', subplot = True, plot = True)


        self.buy_signal = bt.If(self.drf_smoothened(0) < 0.2, 1, 0)
        self.sell_signal = bt.If(self.drf_smoothened(0) > 0.8, 1, 0)


        self.long_exit = bt.Or(self.drf_smoothened > 0.8, bt.And(self.drf_smoothened(0) - self.drf_smoothened(-2) < 0, self.trade_length == 2 ), self.trade_length == 6)
        self.short_exit = bt.Or(self.drf_smoothened < 0.2, bt.And(self.drf_smoothened(0) - self.drf_smoothened(-2) > 0, self.trade_length == 2 ), self.trade_length == 6)


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

        if self.position.size != 0:
            print(self.trade_length)
            self.trade_length += 1


        if self.buy_signal and self.position.size == 0:
            self.order = self.buy()
            # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())


        if self.sell_signal and self.position.size == 0:

            self.order = self.sell()

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())
        
  
       
        
        if self.position.size > 0 and self.long_exit == 1:
            self.trade_length = 0

            # Closes the position
            self.close()

        if self.position.size < 0 and self.short_exit == 1:
            self.trade_length = 0

            self.close()






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
