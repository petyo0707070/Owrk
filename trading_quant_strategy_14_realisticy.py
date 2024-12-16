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
#Try to replicate the book strategy
##########################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        #df =  df[ df['volume'] != 0]
        #df = df[df['high'] != df['low']]
        #df['openinterest'] = pd.factorize(df.index.hour)[0]
        df_training = df[300000:]
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Minutes, compression = timeframe)
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
        slow = 20,
        fast = 8
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

        self.first_morning_candle = 1
        self.first_afternoon_candle = 1

        self.order = None
        self.short_stoploss = None

        self.morning_close_array = []
        self.morning_high_array = []
        self.morning_low_array = []


        self.morning = bt.And(self.data.datetime.time() >= datetime.time(9, 30,0), self.data.datetime.time() < datetime.time(13, 0,0) )
        self.afternoon = bt.And(self.data.datetime.time() >= datetime.time(13, 0,0), self.data.datetime.time() <= datetime.time(16, 30,0) )

        self.highest_high = None
        self.lowest_low = None
        self.highest_close = None
        self.lowest_close = None

        self.lowest_quarter = None
        self.highest_quarter = None
        

        #self.buy_signal_1 = bt.And(self.data.close > self.lowest_low, self.data.close < self.lowest_quarter, self.afternoon)
        #self.short_close_signal_1 = bt.And(self.data.close > self.lowest_low, self.data.close < self.lowest_quarter, self.afternoon)

        #self.short_signal_1 = bt.And(self.data.close < self.highest_high, self.data.close > self.highest_quarter, self.afternoon)
        #self.long_close_signal_1 = bt.And(self.data.close < self.highest_high, self.data.close > self.highest_quarter, self.afternoon) 

        self.buy_signal = 0#bt.And(self.va(-1) < 0, self.va(0) > 0)
        self.sell_signal = 0#bt.And(self.va(-1) > 0, self.va(0) < 0)
        


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
        self.trade_array[len(self.trade_array) - 1].append(trade.pnl / self.broker.get_value() * 100)
        self.order = None


    def next(self):



        if self.data.datetime.time() >= datetime.time(9, 30,0) and self.data.datetime.time() < datetime.time(13, 0,0):


            if self.first_morning_candle:
                self.first_afternoon_candle = 1
                self.first_morning_candle = 0

                self.morning_close_array.clear()
                self.morning_high_array.clear()
                self.morning_low_array.clear()

            
                self.highest_high = None
                self.lowest_low = None
                self.highest_close = None
                self.lowest_close = None

                self.morning_close_array.append(self.data.close[0])
                self.morning_high_array.append(self.data.high[0])
                self.morning_low_array.append(self.data.low[0])
            
            else:
                self.morning_close_array.append(self.data.close[0])
                self.morning_high_array.append(self.data.high[0])
                self.morning_low_array.append(self.data.low[0])

        if self.data.datetime.time() >= datetime.time(13, 0,0) and self.data.datetime.time() <= datetime.time(16, 0,0):


            if self.first_afternoon_candle:

                self.first_afternoon_candle = 0
                self.first_morning_candle = 1

                self.highest_high = max(self.morning_high_array)
                self.lowest_low = min(self.morning_low_array)
                self.highest_close = max(self.morning_close_array)
                self.lowest_close = min(self.morning_close_array)
                self.lowest_quarter = self.lowest_low + 0.25 * (self.highest_high - self.lowest_low)
                self.highest_quarter = self.highest_high - 0.25 * (self.highest_high - self.lowest_low)

                print(f'Highest High: {self.highest_high}, Lowest Low: {self.lowest_low} highest quarter: {self.highest_quarter}, lowest quarter: {self.lowest_quarter}')

            else:


                # Close a long and enter short if price trades below the low of the morning range
                if self.position.size > 0 and self.data.open[0] > self.lowest_low:
                    self.close(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta(minutes  =15))#self.sell(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta(1))
                    #self.sell(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta( minutes = 15))#self.sell(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta(1))
                    

                if self.position.size < 0 and self.data.open[0] < self.highest_high:
                    self.close(exectype= bt.Order.StopLimit, price =  self.highest_high, plimit= self.highest_high, valid = datetime.timedelta(minutes = 15))#self.sell(exectype= bt.Order.StopLimit, price =  self.highest_high, plimit = self.highest_high, valid = datetime.timedelta(1))
                    #self.buy(exectype= bt.Order.StopLimit, price =  self.highest_high, plimit= self.highest_high,valid = datetime.timedelta(minutes = 15))#self.buy(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.highest_high, valid = datetime.timedelta(1))
                    

                #if self.position.size == 0 and self.data.open[0] > self.lowest_low and self.data.open[0] < self.highest_high:         
                #    self.sell(exectype= bt.Order.StopLimit, price = self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta(minutes = 15))#self.buy(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.highest_high, valid = datetime.timedelta(1))
                #    self.buy(exectype= bt.Order.Stop, price =  self.highest_high, plimit = self.highest_high,valid = datetime.timedelta(minutes = 15))#self.sell(exectype= bt.Order.StopLimit, price =  self.lowest_low, plimit = self.lowest_low, valid = datetime.timedelta(1))

                # Close a short if the close is the bottom 1/4 of the morning range
                if self.data.open[0] > self.lowest_quarter and self.data.close[0] > self.lowest_low and self.data.close[0] < self.lowest_quarter  and self.position.size < 0:
                    self.close()

                # Closes a long if the close is the top 1/4 of the morning trading range
                if self.data.open[0] < self.highest_quarter and self.data.close[0] < self.highest_high and self.data.close[0] > self.highest_quarter  and self.position.size > 0:
                    self.close()



                # Enters a long if the close is in the bottom 1/4 of the morning trading range
                if self.data.close[0] > self.lowest_low and self.data.close[0] < self.lowest_quarter and self.position.size == 0:
                    self.order = self.buy()
                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

                    self.trade_array.append([])
                    self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                    self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                    self.trade_array[len(self.trade_array) - 1].append(1)
                    self.trade_array[len(self.trade_array) - 1].append(len(self))
                    self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())


                # Enters a short if the close is in the top 1/4 of the morning trading range
                if self.data.close[0] < self.highest_high and self.data.close[0] > self.highest_quarter and self.position.size == 0:

                    self.order = self.sell()
                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

                    self.trade_array.append([])
                    self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                    self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                    self.trade_array[len(self.trade_array) - 1].append(1)
                    self.trade_array[len(self.trade_array) - 1].append(len(self))
                    self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())
        
        if datetime.time(16, 0, 0) == self.data.datetime.time():
            self.order = self.close()






        if self.first_close is None:
            self.first_close = self.data.close[0]


        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
# This executes if we are not already in the market to enter long





        if self.sell_signal and self.position.size == 0:

            self.order = self.sell()

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())
            #self.short_stoploss = self.data.close[0] + 2 * (self.data.high[0] - self.data.close[0])

        
  
       
        
        if self.position.size > 0 and self.long_exit == 1:
            # Closes the position
            self.close()


        if self.position.size < 0 and self.short_exit == 1:
            self.short_stoploss = None 
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
