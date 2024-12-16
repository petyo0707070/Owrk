import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
from collections import deque
import sys
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer
import math
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation



##########################################################
#Trying A simple neural netowrk
##########################################################



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro()
        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )

        df_training = df[127:]
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Days, compression = timeframe)


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
        results = cerebro.run(maxcpus=None, optdatas = True, runonce = False)

    # Get analyzer results
        sharpe_ratio = results[0].analyzers.sharpe.get_analysis()
        returns = results[0].analyzers.returns.get_analysis()
        max_drawdown = results[0].analyzers.drawdown.get_analysis()
        print("Sharpe Ratio:", sharpe_ratio['sharperatio'])
        print("Returns:", returns['rtot'])
        print("Max Drawdown:", max_drawdown['max']['drawdown'])
        cerebro.plot(style = 'candlestick')

    run_strategy('trading_view_btc_1hr.csv', timeframe = 60, leverage = 200)

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


def martin_ratio_loss(y_true, y_pred):
    # Calculate the returns (assuming y_true and y_pred represent returns or prices)
    returns = y_pred  # Predicted returns
    peak = tf.math.cummax(y_pred, axis=0)  # Compute cumulative max (peak) over time

    # Calculate the ulcer index
    drawdowns = (peak - y_pred) / peak
    ulcer_index = tf.sqrt(tf.reduce_mean(tf.square(drawdowns), axis=0))

    # Mean return
    mean_return = tf.reduce_mean(returns, axis=0)

    # Martin Ratio
    martin_ratio = mean_return / (ulcer_index + 1e-8)  # Adding epsilon to avoid division by zero

    # Since we want to maximize the Martin Ratio, we minimize the negative Martin Ratio
    loss = -martin_ratio
    
    return loss

class Quant_Strategy_5(bt.Strategy):
    lines =('kama',)
    params = dict(
        lookback = 1000
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

        self.cmma_6 = self.data.close(0) - bt.ind.SMA(self.data, period = 6, plot = False)
        self.cmma_24 = self.data.close(0) - bt.ind.SMA(self.data, period = 24, plot = False)
        self.cmma_72 = self.data.close(0) - bt.ind.SMA(self.data, period = 72, plot = False)
        self.atr = bt.ind.ATR(self.data, period = 14)

        self.cmma_6_normalized = self.cmma_6/self.atr * math.sqrt(self.p.lookback)
        self.cmma_24_normalized = self.cmma_24/self.atr * math.sqrt(self.p.lookback)
        self.cmma_72_normalized = self.cmma_72/self.atr * math.sqrt(self.p.lookback)

        self.cmma_6_array = deque(maxlen=self.p.lookback)
        self.cmma_24_array = deque(maxlen=self.p.lookback)
        self.cmma_72_array = deque(maxlen=self.p.lookback)
        self.y_returns = deque(maxlen=self.p.lookback)

        self.model = create_lstm_model()
        self.model_outputs = deque(maxlen=self.p.lookback)
        self.previous_model_output = 0
        self.counter = 0


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
        if len(self.cmma_6_array) < self.p.lookback:
            self.cmma_6_array.append(self.cmma_6_normalized[-1])
            self.cmma_24_array.append(self.cmma_24_normalized[-1])
            self.cmma_72_array.append(self.cmma_72_normalized[-1])
            self.y_returns.append((self.data.close[0] - self.data.close[-1])/self.data.close[-1])
            self.model_outputs.append([[self.previous_model_output]])

        elif self.counter < self.p.lookback:
            self.counter += 1
            self.cmma_6_array.append(self.cmma_6_normalized[-1])
            self.cmma_24_array.append(self.cmma_24_normalized[-1])
            self.cmma_72_array.append(self.cmma_72_normalized[-1])
            self.y_returns.append((self.data.close[0] - self.data.close[-1])/self.data.close[-1])

            cum_norm_cmma_6_array = np.array(self.cmma_6_array).reshape(-1,1)
            cum_norm_cmma_24_array = np.array(self.cmma_24_array).reshape(-1,1)
            cum_norm_cmma_72_array = np.array(self.cmma_72_array).reshape(-1,1)

            qt = QuantileTransformer(output_distribution= 'normal')
            cum_norm_cmma_6_array = qt.fit_transform(cum_norm_cmma_6_array)
            cum_norm_cmma_24_array = qt.fit_transform(cum_norm_cmma_24_array)
            cum_norm_cmma_72_array = qt.fit_transform(cum_norm_cmma_72_array)

            scaler = MinMaxScaler(feature_range = (-1,1))           
            scaled_cmma_6 = scaler.fit_transform(cum_norm_cmma_6_array)
            scaled_cmma_24 = scaler.fit_transform(cum_norm_cmma_24_array)
            scaled_cmma_72 = scaler.fit_transform(cum_norm_cmma_72_array)

            cmma_matrix = np.stack([scaled_cmma_6, scaled_cmma_24, scaled_cmma_72], axis = 1)
            X_train = np.hstack((cmma_matrix, np.array(self.model_outputs)))
            X_train = X_train.reshape(self.p.lookback, 1, 4)

            Y_train = scaler.fit_transform(qt.fit_transform(np.array(self.y_returns).reshape(-1,1)))
            self.model.fit(X_train, Y_train, epochs = 10, batch_size = 4)
            result = self.model.predict(X_train[-1].reshape(1,1,4))
            self.previous_model_output = float(result[0][0])
            self.model_outputs.append([[self.previous_model_output]])
            print(self.model_outputs)

            
        elif self.counter >= self.p.lookback:
            self.counter += 1
            self.cmma_6_array.append(self.cmma_6_normalized[-1])
            self.cmma_24_array.append(self.cmma_24_normalized[-1])
            self.cmma_72_array.append(self.cmma_72_normalized[-1])
            self.y_returns.append((self.data.close[0] - self.data.close[-1])/self.data.close[-1])
            self.model_outputs.append([[self.previous_model_output]])

            cum_norm_cmma_6_array = np.array(self.cmma_6_array).reshape(-1,1)
            cum_norm_cmma_24_array = np.array(self.cmma_24_array).reshape(-1,1)
            cum_norm_cmma_72_array = np.array(self.cmma_72_array).reshape(-1,1)

            qt = QuantileTransformer(output_distribution= 'normal')
            cum_norm_cmma_6_array = qt.fit_transform(cum_norm_cmma_6_array)
            cum_norm_cmma_24_array = qt.fit_transform(cum_norm_cmma_24_array)
            cum_norm_cmma_72_array = qt.fit_transform(cum_norm_cmma_72_array)

            scaler = MinMaxScaler(feature_range = (-1,1))           
            scaled_cmma_6 = scaler.fit_transform(cum_norm_cmma_6_array)
            scaled_cmma_24 = scaler.fit_transform(cum_norm_cmma_24_array)
            scaled_cmma_72 = scaler.fit_transform(cum_norm_cmma_72_array)

            cmma_matrix = np.stack([scaled_cmma_6, scaled_cmma_24, scaled_cmma_72], axis = 1)
            X_train = np.hstack((cmma_matrix, np.array(self.model_outputs)))
            X_train = X_train.reshape(self.p.lookback, 1, 4)

            Y_train = scaler.fit_transform(qt.fit_transform(np.array(self.y_returns).reshape(-1,1)))
            self.model.fit(X_train, Y_train, epochs = 10, batch_size = 4)
            result = self.model.predict(X_train[-1].reshape(1,1,4))
            print(result)
            print(f"Current Candle is {self.data.close[0]}, previous one is {self.data.close[-1]}")
            self.previous_model_output = float(result[0][0])
            self.model_outputs.append([[self.previous_model_output]])

        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
#
        if False: 
            self.order = self.buy(data = self.data0)
            # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())

            self.long_stoploss = self.data0.close[0] - self.atr 


        if False:  


            self.order = self.sell(data = self.data0)

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())
            #self.short_stoploss = self.data.close[0] + 2 * (self.data.high[0] - self.data.close[0])

            self.short_stoploss = self.data0.close[0] + self.atr

        

    def stop(self):
        pass
        #df = pd.DataFrame(self.rsi_array, columns = ['RSI'])
        #sns.histplot(df['RSI'], kde = True)
        #print(f"Mean is {df['RSI'].mean()}")
        #print(f"Stand Deviation is {df['RSI'].std()}")
        #print(f"Skewness is {df['RSI'].skew()}")
        #print(f"Kurtosis is {df['RSI'].kurt()}")
        #plt.show()

      


def create_lstm_model():
    model = Sequential()
    
    # First LSTM layer with 4 neurons (input shape: 1 timestep, 4 features)
    model.add(LSTM(4, input_shape=(1, 4), return_sequences=True, activation='tanh'))
    # Second LSTM layer with 3 neurons
    model.add(LSTM(3, return_sequences=True, activation='tanh'))
    # Third LSTM layer with 2 neurons
    model.add(LSTM(2, return_sequences = True,activation='tanh'))
    # Output layer with 1 neuron and tanh activation (output between -1 and 1)
    model.add(Dense(1, activation='tanh'))
    model.add(Activation('tanh'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    main()
