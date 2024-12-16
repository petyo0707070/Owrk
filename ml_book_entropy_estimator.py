import numpy as np
import struct
import pandas as pd
import matplotlib.pyplot as plt




def matchLength(msg,i,n):
# Maximum matched length+1, with overlap.
# i>=n & len(msg)>=i+n
    subS=''
    for l in range(n):
        msg1=msg[i:i+l+1]
        for j in range(i-n,i):
            msg0=msg[j:j+l+1]
            if msg1==msg0:
                subS=msg1
                break # search for higher l.
    return len(subS)+1,subS # matched length + 1




def konto(msg,window=None):

    out={'num':0,'sum':0,'subS':[]}
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    if window is None:
        points=range(1,   round(len(msg)/2+1))
    else:
        window=min(window,len(msg)/2)
        points=range(window,len(msg)-window+1)

    for i in points:
        if window is None:
            l,msg_=matchLength(msg,i,i)
            out['sum']+=np.log2(i+1)/l # to avoid Doeblin condition
        else:
            l,msg_=matchLength(msg,i,window)
            out['sum']+=np.log2(window+1)/l # to avoid Doeblin condition
        out['subS'].append(msg_)
        out['num']+=1
    out['h']=out['sum']/out['num']
    out['r']=1-out['h']/np.log2(len(msg)) # redundancy, 0<=r<=1
    return out




if __name__=='__main__':
    window_ = 1


    df = pd.read_csv('trading_view_spy_1_and_2_converted.csv', index_col= 0, parse_dates= True, names =['time','open', 'high', 'low', 'close', 'volume'])
    df['return'] = (df['close']).pct_change()
    df = df.dropna(subset=['return'])

    df['bin'] = pd.qcut(df['return'], q= 10 , labels= False) + 1

    df['bin'] = df['bin'].rolling(window = window_).sum()

    ##############################
    df = df.dropna(subset=['bin'])
    ##############################

    ##############################
    df['bin'] = df['bin'].apply(lambda x: bin(round(x))[2:] +'0')
    ##############################


    #df.loc[:, 'bin'] = df['bin'].apply(lambda x: str(round(x, 4)))

    print(df)
    df.loc[:, 'entropy'] = df['bin'].apply(lambda x: konto(x) )

    df.loc[:, 'entropy'] = df['entropy'].apply(lambda x: x['r'])


    df['entropy'].plot(kind='hist', bins = 10)
    print(df)
    print(f"Mean entropy is {df['entropy'].mean()} adn the standard div is {df['entropy'].std()}")
    print(f"Correlation between entropy and returns is {df['entropy'].corr(abs(df['return']))}")
    plt.show()


    msg='“11100001”'
    print( konto(msg))
    print((msg+msg[::-1]))