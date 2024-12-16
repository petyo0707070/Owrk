import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import backtrader as bt


# You need to install the following packages in case you do not have them:
# backtrader -> pip install backtrader
# pandas -> pip install pandas
# numpy -> pip install numpy
# matplotlib -> pip install matplotlib
# sklearn -> pip install sklearn


# The trading strategy uses 2x leverage, 0.01% commision per order and 1 tick slippage. It revolves around computing the Rate of Change
# Rate Of Change(ROC) = (Close - Close(lookback))/Close(lookback), where lookback is the candle close n candle's ago
# ROC(5) would be used to refer to the rate of change between the current candle and the candle 5 period's ago
# RSI is calculated for the CAPE Ratio for the previous 14 months. 
# Trades are executed on a daily timeframe, using data for CAPE from the previous months and back
# Entry Long Condition: ROC(5) < ROC(252) & ROC(1:6) < ROC(1:253) & Cape RSI(1) /Last month's/ < 50
# Keep in mind ROC(1:6) refers to the rate of change between yesterday's and the candle 6 days ago, ROC uses days!
# Positions are entered/exited on the open of daily candles, technically orders are submitted on the close but executed on the open
# Exit Long Condition: ROC(5) > ROC(252) & ROC(1:6) > ROC(1:253)
# Keep in mind Backtrader reports returns as arithmetic sums instead of geometric ones, therefore, look at the red line on the plot for the actual realized return
# The Blue return Line on the plot is the benchmark return (SP500)

def main():
    # Prepare the Data for manipulation
    df = prepare_data()

    # Replicate the Paper Results using monthly instead of yearly data
    replicate_paper_results(df)

    # Trains the model and outputs results for the training data set
    training()

    #Tests the model and outputs the results of the testing data set
    testing()


def prepare_data():
    # Load the dataset
    df = pd.read_csv('cape.csv', names = ['Date', 'Real Total Price Return', 'Real TR Earnings', '10 Year Forward Stock Return'])
    df = df[8:1849]

    #Load the columns as float values
    df['Real Total Price Return'] = df['Real Total Price Return'].astype(float)
    df['Real TR Earnings'] = df['Real TR Earnings'].astype(float)
    df['10 Year Forward Stock Return'] = df['10 Year Forward Stock Return'].astype(float)

    # Calculate the TR Cape Ratio
    df['Cape Ratio'] = df['Real Total Price Return']/(df['Real TR Earnings'].rolling(window=120).mean())
    #Calculate the monthly Return
    df['Return'] = df['Real Total Price Return'].shift(-1)/df['Real Total Price Return'] - 1

    # Drop the data before 1881 as needs 10 years to calculate the Cape Ratio
    df = df[120:]

    #Split into train and test df for later to be used in the trading strategy
    train_df = df[:round(0.6 * len(df))]
    test_df = df[round(0.6 * len(df)):]
    return df

def replicate_paper_results(df):
    # Fit the model
    regression_df = df.dropna(subset=['10 Year Forward Stock Return'])
    model = LinearRegression()
    model.fit(np.log(regression_df[['Cape Ratio']]), regression_df[['10 Year Forward Stock Return']])

    #Get predictions and convert them into a workable format i.e. 1D array
    y_predict = model.predict(np.log(df[['Cape Ratio']])).flatten()
    y_actual = np.array(regression_df[['10 Year Forward Stock Return']]).flatten()    

    #Calculate the R2 and MSE
    r2 = r2_score(y_actual, y_predict[0:len(y_actual)])
    mse = mean_squared_error(y_actual, y_predict[0:len(y_actual)])

    # Add some NaN observations to the actual 10 Year Forward Stock Return, as it cuts out in 2014 and we need to plot until 2024
    y_actual = np.pad(y_actual, (0, len(y_predict) - len(y_actual)), constant_values=np.nan)

    df.set_index('Date', inplace = True)
    # Plot Figure 1

    
    plt.plot(df.index, y_predict, label = 'Forecasted Values', color = 'black', linewidth = 1)
    plt.plot(df.index, y_actual, label = 'Actual Values', color = 'blue', linewidth = 1)
    plt.legend()
    plt.ylabel('Real Stock Returns')
    plt.xticks(ticks = df.index[::200])
    plt.title('120 Month Forward Stock Return')
    plt.show()

    print(f"The regression result is: 10 Year Forward Stock Return = {round(model.intercept_[0],5)} {round(model.coef_[0][0],3)}log(CAPE Ration)")
    print(f"Model R2 is {round(r2,3)} with a Mean Squared Error of: {round(mse, 5)}")



def training():
    def run_training():

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
            print('\nThe results of the training dataset / 1993 - 2007/, where we used the previous monthly \n CAPE Ratio Values to calculate RSI and entered on daily candles using Rate Of Change methodology')

    run_training()

def testing():
    def run_testing():

        def run_strategy(datastream, timeframe, leverage):
            cerebro = bt.Cerebro()

            df_daily = pd.read_csv('trading_view_spy_daily.csv', index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'])
            df_training_daily = df_daily[3739:]

            df_monthly = pd.read_csv(datastream, header = None, parse_dates=True, names = ['date','open', 'high', 'low', 'close', 'volume', 'openinterest'] )
            df_monthly['date'] = pd.to_datetime(df_monthly['date'], format = '%d/%m/%Y')
            df_monthly.set_index(df_monthly['date'], inplace=True)
            df_training_monthly = df_monthly[round(0.6 * len(df_monthly)):]


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
            print('\nThe results of the testing dataset / 2007 - 2024/, where we used the previous monthly \n CAPE Ratio Values to calculate RSI and entered on daily candles using Rate Of Change methodology')

    run_testing()

if __name__ == '__main__':
    main()