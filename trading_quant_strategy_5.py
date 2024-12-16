import pandas as pd
import math
from datetime import date
import datetime
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import sys 





def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro(cheat_on_open=True)

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        df_training = df[5000:]
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
    lines =('arima_high',
            'arima_close',
            'arima_low',
            'dummy_high',
            'dummy_close',
            'dummy_low', )
    params = dict(
        start = 365
       
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

        #For the prediction it is very important to not include the current candle
        self.close_history = [self.data.close[-i] for i in range (1, self.p.start)]
        self.high_history = [self.data.high[-i] for i in range (1, self.p.start)]
        self.low_history = [self.data.low[-i] for i in range (1, self.p.start)]


        # Since add minimum period does not work, adding an sma fixes the problem
        self.sma = bt.ind.SMA(self.data, period = 252)
        self.dummy_sma = bt.ind.SMA(self.data, period = self.p.start)
        self.ema_21 = bt.ind.EMA(self.data, period = 21)
        self.ema_55 = bt.ind.EMA(self.data, period = 55)
        self.ema_89 = bt.ind.EMA(self.data, period = 89)

        self.trend = bt.If(self.ema_55 > self.ema_89, 1, 0)



        self.order= None
        self.order_sell = None

        self.stopprice_long = None
        self.stopprice_short = None


        self.buy_signal = 0
        self.sell_signal = 0


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

        self.lines.arima_high[0] = self.lines.dummy_high[-1]
        self.lines.arima_close[0] = self.lines.dummy_close[-1]
        self.lines.arima_low[0] = self.lines.dummy_low[-1]


        if self.first_close is None:
            self.first_close = self.data.close[0]

# Append the current candle's information to the history arrays
        self.close_history.append(self.data.close[0])
        self.high_history.append(self.data.high[0])
        self.low_history.append(self.data.low[0])

        self.close_history.pop(0)
        self.high_history.pop(0)
        self.low_history.pop(0)



        model_close = ARIMA(self.close_history, order = (1,1,1))
        model_close_fit = model_close.fit()
        model_close_fit_forecast = model_close_fit.forecast()
        close_forecast = float(model_close_fit_forecast[0])
        self.lines.dummy_close[0] = close_forecast

        #Calculates the high forecast
        model_high = ARIMA(self.high_history, order = (1,1,1))
        model_high_fit = model_high.fit()
        model_high_fit_forecast = model_high_fit.forecast()
        high_forecast = float(model_high_fit_forecast[0])
        self.lines.dummy_high[0] = high_forecast

        #Calculates the low forecast
        model_low = ARIMA(self.low_history, order = (1,1,1))
        model_low_fit = model_low.fit()
        model_low_fit_forecast = model_low_fit.forecast()
        low_forecast = float(model_low_fit_forecast[0])
        self.lines.dummy_low[0] = low_forecast

        self.buy_signal = 1 if (self.lines.arima_high[0] > self.data.close[0] and self.lines.arima_low[0] < self.data.close[0] and self.data.close[0] > self.sma[0]) else 0#and self.lines.dummy_close[-1] < self.l.dummy_close[0]
        self.sell_signal = 1 if (self.lines.arima_high[0] > self.data.close[0] and self.lines.arima_low[0] < self.data.close[0] and self.data.close[0] < self.sma[0]) else 0#and self.lines.dummy_close[-1] > self.l.dummy_close[0]

 

        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        #if self.order:
        #    return
# This executes if we are not already in the market to enter long
        if not self.position:
            # Enters Long
            if self.buy_signal:
                self.log(f"Buy order for the next candle at price {self.lines.dummy_high[0]}")

                # Strict Limit order if gap up wait for a retest

                self.order = self.buy(price = self.lines.dummy_high[0], plimit = self.lines.dummy_high[0],exectype= bt.Order.StopLimit, valid = datetime.timedelta(1))

                # Enters if the new candle gaps up and does not wait for the limit to be hit
                #self.order = self.buy(price = self.lines.dummy_high[0], exectype= bt.Order.Stop, valid = datetime.timedelta(1))
    

                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'
                self.trade_array.append([])
                self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                self.trade_array[len(self.trade_array) - 1].append(1)
                self.trade_array[len(self.trade_array) - 1].append(len(self))
                self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())

            # Enter Short
            if self.sell_signal:

                self.log(f"Sell order for the next candle at price {self.lines.dummy_low[0]}")

                # Always wait for the exact limit to be hit even if there is a gap
                self.order = self.sell(price = self.lines.dummy_low[0], plimit= self.lines.dummy_low[0],exectype= bt.Order.StopLimit, valid = datetime.timedelta(1))

                # Enters if the new candle gaps down and does not wait for the limit to be hit
                #self.order = self.sell(price = self.lines.dummy_low[0],exectype= bt.Order.Stop, valid = datetime.timedelta(1))


                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'
                self.trade_array.append([])
                self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
                self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
                self.trade_array[len(self.trade_array) - 1].append(2)
                self.trade_array[len(self.trade_array) - 1].append(len(self))
                self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())


        if self.position.size > 0:

            # Closes the position
            self.close()
            self.log(f"Trade closed at index {len(self)}")



        if self.position.size < 0:

            # Closes the position
            self.close()
            self.log(f"Trade closed at index {len(self)}")




    def stop(self):
        pass

      

if __name__ == "__main__":
    main()
