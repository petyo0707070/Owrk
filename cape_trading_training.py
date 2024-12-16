import pandas as pd
from datetime import date
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
import numpy as np
import sys

###################################################################

####################################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()

        df_daily = pd.read_csv('trading_view_spy_daily.csv', index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'])
        df_training_daily = df_daily[1:3739]

        df_monthly = pd.read_csv(datastream, header = None, parse_dates=True, names = ['date','open', 'high', 'low', 'close', 'volume', 'openinterest'] )
        df_monthly['date'] = pd.to_datetime(df_monthly['date'], format = '%d/%m/%Y')
        df_monthly.set_index(df_monthly['date'], inplace=True)
        df_training_monthly = df_monthly[117:round(0.6 * len(df_monthly))]


        data1 = bt.feeds.PandasData(dataname = df_training_daily, timeframe = bt.TimeFrame.Days, compression = timeframe)
        data = bt.feeds.PandasData(dataname = df_training_monthly, timeframe = bt.TimeFrame.Months, compression = timeframe)

        cerebro.adddata(data1)
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

    run_strategy('sp_500_monthly.csv', timeframe = 1, leverage = 100)



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
        ('cape', 6),
    )




class Quant_Strategy_5(bt.Strategy):

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

        self.cape_ratio = self.data1.openinterest
        self.cape_rsi = bt.ind.RSI(self.cape_ratio, period = 14, plotname = 'Cape Ratio RSI')

        self.roc_fast = (self.data0.close(0) - self.data0.close(-5))/self.data0.close(-5)
        self.roc_slow = (self.data0.close(0) - self.data0.close(-252))/self.data0.close(-252)

        self.buy_signal = bt.And(self.roc_fast(0) < self.roc_slow(0), self.roc_fast(-1) < self.roc_slow(-1))
        self.long_exit = bt.And(self.roc_fast(0) > self.roc_slow(0), self.roc_fast(-1) > self.roc_slow(-1))


    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Nothing to be done the order is already entered to the broker
            return

        # Checks if an order is completed as the broker could reject it if not enough margin
        if order.status in [order.Completed]:
            if order.isbuy():
                #self.log(
                #    'BUY EXECUTED, Price: %.5f, Cost: %.2f, Comm %.2f' %
                #    (order.executed.price,
                #     order.executed.value,
                #     order.executed.comm))
                self.buyprice = order.executed.price

            elif order.issell():
                pass

                #self.log('SELL EXECUTED, Price: %.5f, Cost: %.2f, Comm %.2f' %
                #         (order.executed.price,
                #          order.executed.value,
                #          order.executed.comm))


        # Says that for some reason or another the given order did not go through
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected, status {order.getstatusname()}')
        # This means that there is not a pending order and allows for new orders
        self.order = None

    def notify_trade(self, trade):
        #If the trade is not closed we can not do a thing
        if not trade.isclosed:
            return

        #self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
        #    (trade.pnl, trade.pnlcomm))
         #Saves the PNL of every trade into array as a %
        self.trade_array[len(self.trade_array) - 1].append(trade.pnl / self.broker.get_value() * 100)
        self.order = None


    def next(self):
        

        if self.first_close is None:
            self.first_close = self.data.close[0]



        # Enters Long
        if self.buy_signal and self.position.size == 0 and self.cape_rsi[-1] < 50:

            self.order = self.buy(data = self.data0)

                # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'
            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())



        # Closes The Long Position
        if self.position.size > 0 and self.long_exit == 1:
            self.close(data=self.data0)

        # In case the position does not reverse correctly this closes it
        if self.position.size < 0:
            self.close(data=self.data0)







    def stop(self):
        print('The results of the training dataset / 1993 - 2007/, where we used the previous monthly \n CAPE Ratio Values to calculate RSI and entered on daily candles using Rate Of Change methodology')

      

if __name__ == "__main__":
    main()
