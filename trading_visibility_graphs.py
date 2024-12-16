import pandas as pd
import matplotlib.pyplot as plt
import backtrader.analyzers as btanalyzers
import backtrader as bt
import numpy as np
import backtrader.feeds as btfeed
import networkx as nx
from ts2vg import HorizontalVG, NaturalVG
from trading_hawkes_process import Hawkes


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
        df_training = df[0:1000]
        print(df_training)
        df_training_2 = df_2[499:]
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Minutes, compression = timeframe)
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

    run_strategy('trading_view_euro_bund_daily.csv', timeframe = 1, leverage = 200)

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
        lookback = 100,
        lookback_2 = 12,
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
        self.hawkes = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = False).hawkes
        self.hawkes_95 = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = False).hawkes_95
        self.hawkes_5 = Hawkes(self.data, volume_lookback = self.p.volume_lookback,hawkes_lookback = self.p.hawkes_lookback, plot = False).hawkes_5

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

        # Here is the actual script
        close_array = [self.data0.close[-i] for i in range (0, self.p.lookback)]
        network = ts_to_vg(close_array)
        self.forecast = network_prediction(network, close_array)

        close_array_1 = [self.data0.close[-i] for i in range (0, self.p.lookback_2)]
        network_1 = ts_to_vg(close_array)
        positive, negative = shortest_path_length_trading(close_array_1, network_1)

        #close_array_2 = [self.data1.close[-i] for i in range (-1, self.p.lookback)]
        #network_2 = ts_to_vg(close_array_2)
        #self.forecast_2 = network_prediction(network_2, close_array_2)



        if self.first_close is None:
            self.first_close = self.data.close[0]
            return
        
    

        #self.return_array.append([bt.num2date(self.data.datetime[0]),self.broker.getvalue(), self.broker.getvalue()/100000 - 1, self.data.close[0]/self.first_close - 1,(self.broker.getvalue()/100000 -1) - (self.data.close[0]/self.first_close - 1)])

# This checks if an oder is pending if true a second one can not be sent
        if self.order:
            return
# This executes if we are not already in the market to enter long



        if self.forecast > 0 and self.position.size == 0 and positive > negative and self.hawkes[0] > self.hawkes_95[0] : 
            self.order = self.buy(data = self.data0)
            # Creates an array inside of an array for every trade where it includes "Trade number", "Entry", "Stop Loss", 'PNL'

            self.trade_array.append([])
            self.trade_array[len(self.trade_array) - 1].append(len(self.trade_array) - 1)
            self.trade_array[len(self.trade_array) - 1].append(self.data.close[0])
            self.trade_array[len(self.trade_array) - 1].append(1)
            self.trade_array[len(self.trade_array) - 1].append(len(self))
            self.trade_array[len(self.trade_array) - 1].append(self.data.datetime.datetime())

            self.long_stoploss = self.data0.close[0] - self.atr 


        if self.forecast < 0  and self.position.size == 0 and negative > positive and self.hawkes[0] > self.hawkes_95[0]:  


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

      

def shortest_path_length_trading(close: np.array, lookback: int):

    avg_short_dist_p = np.zeros(len(close))
    avg_short_dist_n = np.zeros(len(close))

    avg_short_dist_p = np.nan
    avg_short_dist_n = np.nan


    pos = NaturalVG()
    pos.build(close)

    neg = NaturalVG()
    neg.build([-elem for elem in close])

    neg = neg.as_networkx()
    pos = pos.as_networkx()
    
        # you could replace shortest_path_length with other networkx metrics..
    avg_short_dist_p = nx.average_shortest_path_length(pos)
    avg_short_dist_n = nx.average_shortest_path_length(neg)
        
        # Another possibility...
        #nx.degree_assortativity_coefficient(pos)
        #nx.degree_assortativity_coefficient(neg)
        # All kinds of stuff here
        # https://networkx.org/documentation/stable/reference/algorithms/index.html 
    return avg_short_dist_p, avg_short_dist_n

def network_prediction(network, data, times=None):

    '''
    May 21, 2023. This code will be covered in a future video.
    Implementation of this paper:

    Zhan, Tianxiang & Xiao, Fuyuan. (2021). 
    A novel weighted approach for time series forecasting based on visibility graph. 

    https://arxiv.org/abs/2103.13870
    '''
    if times is None:
        times = np.arange(len(data))

    n = len(data)
    degrees = np.sum(network, axis=1)
    num_edges = np.sum(network) # Number of edges * 2 (network is symmetric)

    # Transition probability matrix
    p = network.copy()
    for x in range(n):
        p[x, :] /= degrees[x]

    # Forecast vector
    forecasts = np.zeros(n -1)
    v = data[n-1]
    for x in range(n-1): # Forecast slope, not next val
        forecasts[x] = (v - data[x]) / (times[n-1] - times[x])
    

    srw = np.zeros(n-1)
    lrw_last = None
    walk_x = np.identity(n)
    t = 1
    while True:
        for x in range(n):
            walk_x[x,:] = np.dot(p.T, walk_x[x,:])
       
        # Find similarity with last node (most recent value)
        lrw = np.zeros(n-1) # -1 because not including last
        y = n - 1
        for x in range(n-1):
            lrw[x] =  (degrees[x] / num_edges) * walk_x[x, y]
            lrw[x] += (degrees[y] / num_edges) * walk_x[y, x]

        srw += lrw 
        if (lrw == lrw_last).all():
            #print(t)
            break
        lrw_last = lrw
        t += 1
        if t > 1000:
            break

    forecast_weights = srw / np.sum(srw)
    forecast = np.dot(forecast_weights, forecasts)

    return forecast

def ts_to_vg(data: np.array, times: np.array = None, horizontal: bool = False):
    # Convert timeseries to visibility graph with DC algorithm

    if times is None:
        times = np.arange(len(data))

    network_matrix = np.zeros((len(data), len(data)))

    # DC visablity graph func
    def dc_vg(x, t, left, right, network):
        if left >= right:
            return
        k = np.argmax(x[left:right+1]) + left # Max node in left-right
        #print(left, right, k)
        for i in range(left, right+1):
            if i == k:
                continue

            visible = True
            for j in range(min(i+1, k+1), max(i, k)):
                # Visiblity check, EQ 1 from paper 
                if horizontal:
                    if x[j] >= x[i]:
                        visible = False
                        break
                else:
                    if x[j] >= x[i] + (x[k] - x[i]) * ((t[j] - t[i]) / (t[k] - t[i])):
                        visible = False
                        break

            if visible:
                network[k, i] = 1.0
                network[i, k] = 1.0
        
        dc_vg(x, t, left, k - 1, network) 
        dc_vg(x, t, k + 1, right, network) 

    dc_vg(data, times, 0, len(data) - 1, network_matrix)
    return network_matrix



if __name__ == "__main__":
    main()
