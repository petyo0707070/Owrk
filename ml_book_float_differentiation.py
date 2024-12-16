import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getWeights_FFD(d,thres):

    w,k=[1.],1

    while True:

        w_=-w[-1]/k*(d-k+1)

        if abs(w_)<thres:break

        w.append(w_);k+=1

    return np.array(w[::-1]).reshape(-1,1)

# This makes a series stationary
def fracDiff_FFD(series,d,thres=1e-5):
#Constant width window (new solution)
#Note 1: thres determines the cut-off weight for the window
#Note 2: d can be any positive fractional, not necessarily bounded [0,1].
#1) Compute weights for the longest series
    w=getWeights_FFD(d,thres)
    width=len(w)-1
#2) Apply weights to values
    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].ffill().dropna(),pd.Series()
        for iloc1 in range(width,seriesF.shape[0]):
            loc0,loc1=seriesF.index[iloc1-width],seriesF.index[iloc1]
            if not np.isfinite(series.loc[loc1,name]):continue # exclude NAs
            df_[loc1]=np.dot(w.T,seriesF.loc[loc0:loc1])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

 # calculate the minimum d / float differentiator to make a series stationary
def plotMinFFD(df):
    from statsmodels.tsa.stattools import adfuller
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,11):
        df1=np.log(df[['close']]).resample('1D').last() # downcast to daily obs
        df2=fracDiff_FFD(df1,d,thres=1e-4/2)
        corr=np.corrcoef(df1.loc[df2.index,'close'],df2['close'])[0,1]
        df2=adfuller(df2['close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # with critical value

    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    print(out)
    plt.show()
    return out[out['pVal'] < 0.05].index[0]


if __name__=='__main__':
    df = pd.read_csv('trading_view_spy_daily.csv', index_col= 0, parse_dates= True, names =['time','open', 'high', 'low', 'close', 'volume'])
    differentiator = plotMinFFD(df)
    stationary_series = fracDiff_FFD(np.log(df[['close']]), differentiator, 1e-4/2)
    print(stationary_series)
    plt.plot(stationary_series)
    plt.show()
    