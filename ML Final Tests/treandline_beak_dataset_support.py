import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from trendline_automation import fit_trendlines_single, fit_upper_trendline
import mplfinance as mpf



def trendline_breakout_dataset_support(
        ohlcv: pd.DataFrame, lookback: int, 
        hold_period:int=12, tp_mult: float=3.0, sl_mult: float=3.0, 
        atr_lookback: int=168
):
    assert(atr_lookback >= lookback)

    close = np.log(ohlcv['close'].to_numpy())
    high = np.log(ohlcv['high'].to_numpy())
    low = np.log(ohlcv['low'].to_numpy())
   
    # ATR for normalizing, setting stop loss take profit
    atr = ta.atr(np.log(ohlcv['high']), np.log(ohlcv['low']), np.log(ohlcv['close']), atr_lookback)
    atr_arr = atr.to_numpy()
   
    # Normalized volume
    vol_arr = (ohlcv['volume'] / ohlcv['volume'].rolling(atr_lookback).median()).to_numpy() 
    adx = ta.adx(ohlcv['high'], ohlcv['low'], ohlcv['close'], lookback)
    adx_arr = adx['ADX_' + str(lookback)].to_numpy()

    trades = pd.DataFrame()
    trade_i = 0

    in_trade = False
    tp_price = None
    sl_price = None
    hp_i = None
    for i in range(atr_lookback, len(ohlcv)):
        # NOTE window does NOT include the current candle
        window = close[i - lookback: i]

        s_coefs, r_coefs = fit_trendlines_single(window)

        # Find current value of line
        r_val = r_coefs[1] + lookback * r_coefs[0]
        s_val = s_coefs[1] + lookback * s_coefs[0]

        # Entry
        if not in_trade and (close[i] < s_val):
            
            tp_price = close[i] + atr_arr[i] * tp_mult
            sl_price = close[i] - atr_arr[i] * sl_mult
            hp_i = i + hold_period
            in_trade = True

            trades.loc[trade_i, 'entry_i'] = i
            trades.loc[trade_i, 'entry_p'] = close[i]
            trades.loc[trade_i, 'atr'] = atr_arr[i]
            trades.loc[trade_i, 'sl'] = sl_price 
            trades.loc[trade_i, 'tp'] = tp_price 
            trades.loc[trade_i, 'hp_i'] = i + hold_period
            
            #trades.loc[trade_i, 'slope_r'] = r_coefs[0]
            trades.loc[trade_i, 'slope_s'] = s_coefs[0]

            #trades.loc[trade_i, 'intercept_r'] = r_coefs[1]
            trades.loc[trade_i, 'intercept_s'] = s_coefs[1]



            # Trendline features
            # Resist slope
            #trades.loc[trade_i, 'resist_s'] = r_coefs[0] / atr_arr[i]

            # Support slope
            trades.loc[trade_i, 'support_s'] = s_coefs[0] / atr_arr[i] 

       
            # Resist erorr
            line_vals_r = (r_coefs[1] + np.arange(lookback) * r_coefs[0])
            err_r = np.sum(line_vals_r  - window ) / lookback
            err_r /= atr_arr[i]
            #trades.loc[trade_i, 'tl_err_r'] = err_r


            # Support erorr
            line_vals_s = (s_coefs[1] + np.arange(lookback) * s_coefs[0])
            err_s = np.sum(line_vals_s  - window ) / lookback
            err_s /= atr_arr[i]
            trades.loc[trade_i, 'tl_err_s'] = err_s

            # Max distance from resist
            diff_r = line_vals_r - window
            #trades.loc[trade_i, 'max_dist_r'] = diff_r.max() / atr_arr[i]

            # Max distance from support
            diff_s = line_vals_s - window
            trades.loc[trade_i, 'max_dist_s'] = diff_s.max() / atr_arr[i]

            # Volume on breakout
            trades.loc[trade_i, 'vol'] = vol_arr[i]

            # ADX
            trades.loc[trade_i, 'adx'] = adx_arr[i]


        if in_trade:
            if high[i] >= tp_price:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = tp_price
                
                in_trade = False
                trade_i += 1
            
            elif low[i] <= sl_price or i >= hp_i:
                trades.loc[trade_i, 'exit_i'] = i
                trades.loc[trade_i, 'exit_p'] = sl_price
                
                in_trade = False
                trade_i += 1


    trades['return'] = trades['exit_p'] - trades['entry_p']
    
    # Features
    data_x = trades[["support_s" , "tl_err_s",'vol','max_dist_s','adx']]
    # Label
    data_y = pd.Series(0, index=trades.index)
    data_y.loc[trades['return'] > 0] = 1


    return trades, data_x, data_y