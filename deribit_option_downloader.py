import requests
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import sys
import time

class Options:
    def __init__(self, currency="BTC"):
        """
        Initializing `Options` class.

        Parameters
        ----------
        currency: string ("BTC" or "ETH")
            The cryptocurrency that will be used to retrieve all option
            and volatility related data. The default parameter is "BTC".

        Example
        ---------
        >>> import deribit_data as dm
        >>> data = dm.Options(currency = "BTC")
        """
        self.url = 'https://www.deribit.com/api/v2/public/'
        self.currency = str.lower(currency)
        self.itt = 0

    def get_hist_vol(self, save_csv=False):

        """
        Retrieves the asset's annualised historical volatility measured
        every hour for the past 15 days.

        Parameters
        -------------
        save_csv: boolean
            Select True to save the historical volatility to a csv file.

        Returns
        -------------
        dataframe:
            A time-series dataframe

        csv:
            A csv file will be saved if save_csv is set to True

        Example
        -------------
        >>> df = data.get_hist_vol()
        >>> df.tail()
        date                    btc_hist_vol
        2020-05-13 21:00:00     95.228999
        2020-05-13 22:00:00     95.501434
        2020-05-13 23:00:00     95.553333
        2020-05-14 00:00:00     95.572422
        2020-05-14 01:00:00     95.564853
        """
        data = {"currency": self.currency}
        d = requests.get(self.url + "get_historical_volatility?", data)
        df = pd.DataFrame(d.json()['result'])
        df.columns = ['date', str(self.currency.lower())+'_hist_vol']
        df['date'] = pd.to_datetime(df.date, unit='ms')
        df = df.set_index("date")
        if save_csv == True:
            df.to_csv(str(self.currency) + "_hist_vol.csv")
        return df

    def get_options_list(self):
        """
        This method is used to retrieve the option type and instrument name.
        The output of this method is used as an input to the `get_option_stats`
        method.

        Returns
        -------------
        dataframe:
            A dataframe with relevant information which is fed into the
            `get_option_stats` method.

        Example
        -------------
        >>> df = data.get_options_list()
        >>> df.head()
        expiration_timestamp option_type instrument_name strike
        0	1590134400000	put	    BTC-22MAY20-9250-P  9250.0
        1	1590134400000	call	BTC-22MAY20-12000-C	12000.0
        2	1593158400000	call	BTC-26JUN20-8000-C	8000.0
        3	1589443200000	call	BTC-14MAY20-8875-C	8875.0
        4	1601020800000	put	    BTC-25SEP20-9000-P  9000.0
        """
        data = {'currency': self.currency, 'kind': 'option'}


        r = requests.get(self.url + 'get_instruments', data)
        df = pd.DataFrame(r.json()['result'])
        cols = ['expiration_timestamp', 'option_type', 'instrument_name', 'strike']
        return df[cols]

    def get_option_urls(self):
        """
        Used to retrieve the URL links for all options which are inputted to the
        `collect_data` method.

        Returns
        -------------
        list:
            A dataframe with relevant information which is fed into the
            `get_option_stats` method.

        Example
        -------------
        >>> data.get_options_urls()
        ['https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-4SEP20-13250-P',
        'https://www.deribit.com/api/v2/public/get_order_book?instrument_name=BTC-26MAR21-8000-P',
        ....]
        """
        url_storage = []
        options_list = self.get_options_list()
        request_url = self.url + 'get_order_book?instrument_name='
        for option in range(len(options_list)):
            data = request_url + options_list.instrument_name.values[option]
            url_storage.append(data)
        return url_storage


    def request_get(self, url):


        """
        An intermediate function used in conjunction with the `collect_data` method.
        """

        page = requests.get(url)        

        try:
            self.itt += 1
            return page.json()['result']
        except:
            print(f"{self.itt} options were loaded in {time.time() - self.start}")


    def get_option_trade_data(self):
       

        data = {'instrument_name': 'BTC-5DEC24-100000-C', 'count': 1000}

        r = requests.get('https://history.deribit.com/api/v2/public/get_last_trades_by_instrument', data)


        result = pd.DataFrame(r.json()['result'])


        '''
        This code snippet insures that the output is in a format where the json['result'] part is parced into columns
                    trade_id           timestamp  tick_direction   price  mark_price      iv  ... direction  contracts amount  combo_id  block_trade_id liquidation
        trade_seq                                                                             ...
        1          330892577 2024-12-02 08:14:55               1  0.0061    0.006238   54.82  ...      sell       50.0   50.0       NaN          170992         NaN     
        2          330893652 2024-12-02 08:18:02               2  0.0060    0.006034   56.61  ...       buy        0.1    0.1       NaN             NaN         NaN     
        3          330909818 2024-12-02 09:35:43               2  0.0050    0.005255   55.69  ...       buy        0.1    0.1       NaN             NaN         NaN     
        4          330925967 2024-12-02 11:57:28               2  0.0044    0.004334   55.80  ...       buy        0.1    0.1       NaN             NaN         NaN     
        5          330928767 2024-12-02 12:24:05               2  0.0043    0.004319   55.73  ...       buy        0.1    0.1       NaN             NaN         NaN         NaN         NaN

        Also that the trades are sorted based on when they occured as well as that timestamp is in a readible format
        '''
        result_ = result['trades'].to_numpy()
        df = pd.DataFrame(result_)
        df = pd.json_normalize(df.iloc[: , 0])
        df.set_index('trade_seq', inplace=True)
        df = df.sort_index()
        df['timestamp'] = pd.to_datetime(round(df['timestamp']/1000), unit = 's')

        return df



    def collect_data(self, max_workers=20, save_csv=False):
        """
        Retrieves the price, implied volatility, volume, open interest, greeks
        and other relevant data for all options of the selected asset.

        Parameter:
        ------------
        max_workers: integer
            Select the maximum number of threads to execute calls asynchronously (Default 20)

        save_csv: boolean
            Select True to save the options data to a csv file (Default False)

        Returns
        -------------
        dataframe:
            A dataframe with corresponding statistics for each option

        csv:
            A csv file will be saved if save_csv is set to True (Default is False)

        Example
        -------------
        >>> df = data.collect_data()
        >>> df.columns
        Index(['expiration_timestamp', 'option_type', 'instrument_name', 'strike',
               'underlying_price', 'underlying_index', 'timestamp', 'stats', 'state',
               'settlement_price', 'open_interest', 'min_price', 'max_price',
               'mark_price', 'mark_iv', 'last_price', 'interest_rate',
               'instrument_name', 'index_price', 'greeks', 'estimated_delivery_price',
               'change_id', 'bids', 'bid_iv', 'best_bid_price', 'best_bid_amount',
               'best_ask_price', 'best_ask_amount', 'asks', 'ask_iv'],
                dtype='object')

        >>> df[['instrument_name', 'last_price', 'mark_iv', 'open_interest']].head()
            instrument_name	  last_price mark_iv open_interest
        0	BTC-22MAY20-9250-P	0.0460	77.72	138.8
        1	BTC-22MAY20-12000-C	0.0050	112.28	110.6
        2	BTC-26JUN20-8000-C	0.1825	90.26	771.0
        3	BTC-14MAY20-8875-C	0.0410	90.65	24.7
        4	BTC-25SEP20-9000-P	0.1770	83.37	266.9
        """
        raw_data = []
        pool = ThreadPoolExecutor(max_workers=20)
        print("Collecting data...")

        self.start = time.time() 

        for asset in pool.map(self.request_get, self.get_option_urls()[0:100]):
            raw_data.append(asset)
        df = pd.DataFrame(raw_data)
        df['option_type'] = [df.instrument_name[i][-1] for i in range(len(df))]
        df = df.loc[:, ~df.columns.duplicated()]
        label = datetime.now().strftime(str(self.currency) + "_options_data" +'-%Y_%b_%d-%H_%M_%S.csv')


        """
        This updates the best bid ask and last price so that it is in usd instead of BTC
        """
        df['best_ask_price'] = df['best_ask_price'] * df['index_price']
        df['best_bid_price'] = df['best_bid_price'] * df['index_price']
        df['last_price'] = df['last_price'] * df['index_price']


        if save_csv == True: df.to_csv(label)
        print("Data Collected")
        return df
    

btc_data = Options('BTC')





def exploratory_analysis():
    df = btc_data.get_option_trade_data()

    print(df)
    print(df.columns)
    
    fig, ax1 = plt.subplots()
    ax1.plot(df['timestamp'] , df['price'], label = "Option Price BTC", color='tab:blue')
    ax1.tick_params(axis='y')
    ax1.set_ylabel('Option Price BTC', color='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(df['timestamp'], df['index_price'], label = 'BTC Price', color='tab:red')
    ax2.tick_params(axis='y')
    ax2.set_ylabel('Index Price', color='tab:red')



exploratory_analysis()

plt.show()
