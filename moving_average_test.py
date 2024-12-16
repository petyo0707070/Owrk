import pandas as pd
import math
import backtrader.indicators as btind
import backtrader.analyzers as btanalyzers
import backtrader as bt
import backtrader.feeds as btfeed
from pykalman import KalmanFilter
import numpy as np

def monte_carlo_permutation(df):
    columns_to_log = ['open', 'high', 'low', 'close']
    df[columns_to_log] = df[columns_to_log].apply(np.log)
    df['close-open'] = df['close'] - df['open']
    df['high-open'] = df['high'] - df['open']
    df['low-open'] = df['low'] - df['open']
    df['open-close(-1)'] = df['open'] - df['close'].shift(-1)

    columns_to_shuffle = ['close-open', 'high-open', 'low-open', 'open-close(-1)']
    df[columns_to_shuffle] = df[columns_to_shuffle].apply(np.random.permutation)

    first_open = df['open'].iloc[0]
    df['open'] = df['open'].shift(1) + df['open-close(-1)']
    df['open'].iloc[0] = first_open
    df['high'] = df['open'] + df['high-open']
    df['close'] = df['open'] + df['close-open']
    df['low'] = df['open'] + df['low-open']

    df = df.loc[:, ['open', 'high', 'low', 'close', 'volume']]
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].apply(np.exp).round(2)
    return df



def main():

    def run_strategy(datastream, timeframe, leverage):
        cerebro = bt.Cerebro(cheat_on_open=True)

        df = pd.read_csv(datastream, header = None, index_col = 0, parse_dates=True, names = ['open', 'high', 'low', 'close', 'volume'] )
        df_training = df[0:]
        #df_training = monte_carlo_permutation(df_training)
        print(df_training)
        data = bt.feeds.PandasData(dataname = df_training, timeframe = bt.TimeFrame.Minutes, compression = timeframe)
        #data = SPY_TRVIEW(dataname = datastream,timeframe=bt.TimeFrame.Minutes, compression = timeframe)


        cerebro.adddata(data)
        cerebro.addindicator(Kalman)
        cerebro.run(maxcpus=None, optdatas = True, runonce = True)
        cerebro.plot(style = 'candlestick')

    run_strategy('trading_view_spy_1_and_2_converted.csv', timeframe = 60, leverage = 200)

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




class Kalman(bt.Indicator):
    lines =('kalman_ma',
            'upper_band',
            'lower_band')
    params = dict(
        start = 365,
       
        )
    plotinfo = {
                'subplot': False,
                'plot': True
                }


    def __init__(self):
        self.long_bar = None
        self.first_close = None

        self.dummy_ma = bt.ind.SMA(self.data, period = self.p.start)
        self.atr = bt.ind.ATR(self.data, period = 14)
        self.close_history = [self.data.close[-i] for i in range (0, self.p.start )]

        self.entered_above = 0
        self.entered_below = 0

        self.use_kalman = 1
        #self.sma = bt.ind.SMA(self.data, period = 200)



        self.successful_bounces = 0
        self.failed_bounces = 0

        self.successful_pushbacks = 0
        self.failed_pushbacks = 0


    def next(self):

        if len(self) % 100 == 0:
            print(len(self))

        if self.use_kalman == 1:

            kf = KalmanFilter(transition_matrices = [1], 
                    observation_matrices = [1],
                    initial_state_mean = 0,
                    initial_state_covariance = 1,
                    observation_covariance = 1,
                    transition_covariance = .01
                    )

            state_means, _ = kf.filter(self.close_history)

            self.l.kalman_ma[0] = state_means[-1]
            self.close_history.append(self.data.close[0])
            self.l.upper_band[0] = self.l.kalman_ma[0] + 0.5 * self.atr[0]
            self.l.lower_band[0] = self.l.kalman_ma[0] - 0.5 * self.atr[0]

        else:
            self.l.upper_band[0] = self.sma[0] + 0.5 * self.atr[0]
            self.lower_band[0] = self.sma[0] - 0.5 * self.atr[0]





        # Checks if price entered from above
        if self.entered_above == 0 and self.entered_below == 0 and self.data.close[-1] > self.l.upper_band[-1] and self.data.close[0] < self.l.upper_band[0] and self.data.close[0] > self.l.lower_band[0]:
           self.entered_above = 1
        
        # Checks if price entered above and penetrated on the same candle
        if self.entered_above == 0 and self.entered_below == 0 and self.data.close[-1] > self.l.upper_band[-1] and self.data.close[0] < self.l.lower_band[0]:
           self.failed_bounces += 1
        
        #Checks if price entered from below
        if self.entered_above == 0 and self.entered_below == 0 and self.data.close[-1] < self.l.lower_band[-1] and self.data.close[0] > self.l.lower_band[0] and self.data.close[0] < self.l.upper_band[0]:
           self.entered_below = 1
        
        #Checks if price entered below and penetrated on the same candle
        if self.entered_above == 0 and self.entered_below == 0 and self.data.close[-1] < self.l.lower_band[-1] and self.data.close[0] > self.l.upper_band[0]:
           self.failed_pushbacks += 1

        # Checks if price enetered above and successfully bounced
        if self.entered_above == 1 and self.data.close[-1] < self.l.upper_band[-1] and self.data.close[-1] > self.l.lower_band[-1] and self.data.close[0] > self.l.upper_band[0]:
           self.entered_above = 0
           self.successful_bounces += 1

        # Checks if price entered above and unsuccessfully bounced
        if self.entered_above == 1 and self.data.close[-1] < self.l.upper_band[-1] and self.data.close[-1] > self.l.lower_band[-1] and self.data.close[0] < self.l.lower_band[0]:
           self.entered_above = 0
           self.failed_bounces += 1

        # Checks if price entered below and successfuly pushed back
        if self.entered_below == 1 and self.data.close[-1] < self.l.upper_band[-1] and self.data.close[-1] > self.l.lower_band[-1] and self.data.close[0] < self.l.lower_band[0]:
           self.entered_below = 0
           self.successful_pushbacks += 1

        if self.entered_below == 1 and self.data.close[-1] < self.l.upper_band[-1] and self.data.close[-1] > self.l.lower_band[-1] and self.data.close[0] > self.l.upper_band[0]:
           self.entered_below = 0
           self.failed_pushbacks += 1

        if len(self) % 2000 == 0:
            print(f"There were a total of {self.successful_bounces + self.failed_bounces} bounces with a success rate of {round(self.successful_bounces/(self.successful_bounces + self.failed_bounces),4) * 100}%")
            print(f"There were a total of {self.successful_pushbacks + self.failed_pushbacks} pushbacks with a success rate of {round(self.successful_pushbacks/(self.successful_pushbacks + self.failed_pushbacks),3) * 100}%")          
        


if __name__ == '__main__':
   main()