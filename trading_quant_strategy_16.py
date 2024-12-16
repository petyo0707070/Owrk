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
#An Intraday Strategy based around volatility and mean reversion
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


        #df_1 = pd.read_csv('futures_es_daily.csv', header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'])
        #data1 = bt.feeds.PandasData(dataname = df_1,timeframe = bt.TimeFrame.Days, compression = 1 )
        #cerebro.adddata(data1)
        #print(df_1)





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

        self.atr_09_30 = None
        self.atr_09_30_array = []
        self.atr_09_45 = None
        self.atr_09_45_array = []
        self.atr_10_00 = None
        self.atr_10_00_array = []
        self.atr_10_15 = None
        self.atr_10_15_array = []
        self.atr_10_30 = None
        self.atr_10_30_array = []
        self.atr_10_45 = None
        self.atr_10_45_array = []
        self.atr_11_00 = None
        self.atr_11_00_array = []
        self.atr_11_15 = None
        self.atr_11_15_array = []
        self.atr_11_30 = None
        self.atr_11_30_array = []
        self.atr_11_45 = None
        self.atr_11_45_array = []
        self.atr_12_00 = None
        self.atr_12_00_array = [] 
        self.atr_12_15 = None
        self.atr_12_15_array = []
        self.atr_12_30 = None
        self.atr_12_30_array = []  
        self.atr_12_45 = None
        self.atr_12_45_array = [] 
        self.atr_13_00 = None
        self.atr_13_00_array = []
        self.atr_13_15 = None
        self.atr_13_15_array = []
        self.atr_13_30 = None
        self.atr_13_30_array = []
        self.atr_13_45 = None
        self.atr_13_45_array = []
        self.atr_14_00 = None
        self.atr_14_00_array = []   
        self.atr_14_15 = None
        self.atr_14_15_array = []
        self.atr_14_30 = None
        self.atr_14_30_array = []
        self.atr_14_45 = None
        self.atr_14_45_array = []
        self.atr_15_00 = None
        self.atr_15_00_array = []
        self.atr_15_15 = None
        self.atr_15_15_array = []
        self.atr_15_30 = None
        self.atr_15_30_array = []
        self.atr_15_45 = None
        self.atr_15_45_array = []        

        
        self.order = None
        self.bar_entered = 0
        self.high_entry_bar = None
        self.low_entry_bar = None
        self.last_daily_close_atr_average = None


        self.time_range = pd.date_range(start = '9:30', end = '16:00', freq = '15T').time



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

        if self.data.datetime.time() >= datetime.time(9,30) and self.data.datetime.time() <= datetime.time(15,45):
            # Creates a dynamic variable name that automizes the process and allows for comparison between the same value of ATR in the same bar
            time = str(self.data.datetime.time())
            time = time[0:2] + '_' + time[3:5]
            atr = f"self.atr_{time}"
            atr_array = f"self.atr_{time}_array"


            # Apparently Python differentiates between class attributes and global variables, so I am creating a whole new set of variables just to hold the array, variables get created automatically
            if atr_array not in globals():
                globals()[atr_array] = []



                       
            # We will be comparing with the average ATR of the last 100 candles
            if len(globals()[atr_array]) < 100:
                globals()[atr] = max(self.data0.high[0] - self.data0.low[0], abs(self.data0.high[0] - self.data0.close[-1]), abs(self.data0.low[0] - self.data0.close[-1]))
                globals()[atr_array].append(max(self.data0.high[0] - self.data0.low[0], abs(self.data0.high[0] - self.data0.close[-1]), abs(self.data0.low[0] - self.data0.close[-1])))
                return

            if len(globals()[atr_array]) >= 100:
                globals()[atr_array].pop(0)
                globals()[atr_array].append(max(self.data0.high[0] - self.data0.low[0], abs(self.data0.high[0] - self.data0.close[-1]), abs(self.data0.low[0] - self.data0.close[-1])))
                globals()[atr] = max(self.data0.high[0] - self.data0.low[0], abs(self.data0.high[0] - self.data0.close[-1]), abs(self.data0.low[0] - self.data0.close[-1]))


            # Saves the last daily atr mean which needs to be used for calculating if we enter on the opening candle
            if self.data.datetime.time() == datetime.time(15, 45):
                self.last_daily_close_atr_average = sum(globals()[atr_array])/len(globals()[atr_array])

            # If the Last Daily close atr average is missing
            if self.last_daily_close_atr_average is None:
                return

            # Calculates the atr mean for the 100 candles at the same time in the last 100 days
            atr_mean = sum(globals()[atr_array])/len(globals()[atr_array])

            # Time for entry conditions

            # There is a specific condition if it is the opening candle
            if self.data.datetime.time() == datetime.time(9, 30) and self.position.size == 0:

                # Enters long if the current ATR read on the opening candle is 3 * Average ATR of yesterday's closing candle and the close is in the bottom 20% of the range
                if globals()[atr] > 3 * self.last_daily_close_atr_average and self.data0.close[0] > self.data0.low[0] and self.data0.close[0] <= (self.data0.low[0] + 0.2 * globals()[atr]):
                    self.buy(data = self.data0)
                    self.high_entry_bar = self.data0.high[0]
                    self.low_entry_bar = self.data0.low[0]
                    self.bar_entered = len(self)

                # Enters long if the current ATR read on the opening candle is 3 * Average ATR of yesterday's closing candle and the close is in the top 20% of the range    
                if globals()[atr] > 3 * self.last_daily_close_atr_average and self.data0.close[0] < self.data0.high[0] and self.data0.close[0] >= (self.data0.high[0] - 0.2 * globals()[atr]):
                    pass
                    #self.sell(data = self.data0)            
                    #self.high_entry_bar = self.data0.high[0]
                    #self.low_entry_bar = self.data0.low[0]
                    #self.bar_entered = len(self)

            # Entry Conditions for all other candles except the closing one 
            if self.data.datetime.time() >= datetime.time(9, 45) and self.data.datetime.time() <= datetime.time(15, 30) and self.position.size == 0:

                # Enters long if the current ATR read is 2 * Average ATR of the average for this candle and the close is in the bottom 20% of the range
                if globals()[atr] > 2 * atr_mean and self.data0.close[0] > self.data0.low[0] and self.data0.close[0] <= (self.data0.low[0] + 0.2 * globals()[atr]):
                    self.buy(data = self.data0)
                    self.high_entry_bar = self.data0.high[0]
                    self.low_entry_bar = self.data0.low[0]
                    self.bar_entered = len(self)

                # Enters long if the current ATR read on the opening candle is 3 * Average ATR of yesterday's closing candle and the close is in the top 20% of the range    
                if globals()[atr] > 2 * atr_mean and self.data0.close[0] < self.data0.high[0] and self.data0.close[0] >= (self.data0.high[0] - 0.2 * globals()[atr]):
                    pass
                    #self.sell(data = self.data0)            
                    #self.high_entry_bar = self.data0.high[0]
                    #self.low_entry_bar = self.data0.low[0]
                    #self.bar_entered = len(self)
            
            # Waits for 2 bars before placing a stoploss, then it chooses the lowest low for longs
            if len(self) - self.bar_entered == 2 and self.position.size > 0:
                stop = min(self.data0.low[0], self.data0.low[-1], self.data0.low[-2])
                self.close(data = self.data0, exectype = bt.Order.StopLimit, price = stop, plimit = stop, valid = 0)
                self.high_entry_bar = None
                self.low_entry_bar = None
                self.bar_entered = 0
            
            if len(self) - self.bar_entered == 2 and self.position.size < 0:
                stop = max(self.data0.high[0], self.data0.high[-1], self.data0.high[-2])
                self.close(data = self.data0, exectype = bt.Order.StopLimit, price = stop, plimit = stop, valid = 0)
                self.high_entry_bar = None
                self.low_entry_bar = None
                self.bar_entered = 0

            if datetime.time(15, 45) and self.position.size != 0:
                self.close(data = self.data0)
                self.high_entry_bar = None
                self.low_entry_bar = None
                self.bar_entered = 0
            




            #Time for the entry conditions

            

                        
      

 
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
