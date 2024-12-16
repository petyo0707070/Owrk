import pandas as pd
import numpy as np
import sys
import multiprocessing as mp
import datetime as dt
import time
import matplotlib.pyplot as plt 


def main():
    df = pd.read_csv('futures_es_daily.csv', index_col= 0, parse_dates= True, names =['time','open', 'high', 'low', 'close', 'volume'])
    df = df[0:2000]
    def meta_classifier(df):

        # Gets all the bars where the absolute return exceeded the threshold of 1 standard deviation of the last 100 bars
        sigma = getDailyVol(df['close'], 100)
        df = df[df.index.isin(sigma.index)]
        events = getTEvents(df['close'], sigma)

        # This adds a vertical barrier of maximum 2 days in position
        t1 = df['close'].index.searchsorted(events + pd.Timedelta(days = 3))
        t1=t1[t1<df['close'].shape[0]]
        t1=pd.Series(df['close'].index[t1],index=events[:t1.shape[0]]) 
        events = getEvents(df['close'], events, [2,1], sigma, 0, 48, t1, None)


        df1 = getBins(events, df['close'])

        def unique_events():
            # events is event + target + vertical barrier , df1 is event + ratur + bin (long/close)
            # Not sure what should be used for closeIdx, I am using the close index of the overall dataset but I am getting a result for uniqueness of 0 
            numCoEvents = mpPandasObj(mpNumCoEvents, ('molecule', events.index), numThreads= 24, closeIdx = df.index, t1 = events['t1'])
            numCoEvents = numCoEvents.reindex(range(len(df)), fill_value=0)
            numCoEvents.index = df.index
            
            #numCoEvents = numCoEvents.loc[~numCoEvents.index.duplicated(keep='last')]
            #output = mpPandasObj(mpSampleTW, ('molecule', events.index), numThreads=24, t1 = t1, numCoEvents = numCoEvents)

            print(numCoEvents)
            print(numCoEvents.sum())
            plt.plot(numCoEvents, label = "Num Concurent Events")
            plt.plot(sigma * 100, label = 'Sigma')
            plt.show()
            sys.exit()

            return numCoEvents
            


        # All in all sequential bootstrapping basically makes returns more i.i.d.
        def sequential_bootstrapping():
            indMatrix = getIndMatrix(df.index, t1)
            print(indMatrix)  
                      
            # Here we calculate average uniqueness acurately
            #average_uniqueness = getAvgUniqueness(indMatrix)

            standard_uniqueness = np.random.choice(indMatrix.columns, size = 50)

            bootstrap_result = seqBootstrap(indMatrix, 50)


            #job = {'func': seqBootstrap, 'indM': indMatrix, 'sLength': 100}#indMatrix.shape[1]}
            #jobs = [job]
            #bootstrap_result = processJobs(jobs, numThreads=24)[0]


            print(f"Standard uniqueness {getAvgUniqueness(indMatrix[standard_uniqueness]).mean()}")
            print(f"Sequential Bootstrap uniqueness {getAvgUniqueness(indMatrix[bootstrap_result]).mean()}")

            return indMatrix[standard_uniqueness]
            
        def adjust_weights_on_uniqueness(A):
            return getAvgUniqueness(A) / getAvgUniqueness(A).sum()

        def adjust_weights_on_time_decay(A, slope = 1):
            weights = adjust_weights_on_uniqueness(A)
            weights.sort_index().cumsum()
            print(weights)
            weights_time_decay = weights * getTimeDecay(weights, slope)
            print(weights_time_decay)


        unique_events()
        sample = sequential_bootstrapping()

        adjust_weights_on_time_decay(sample, 0.5)


    meta_classifier(df)




# Used to roll futures
def getRolledSeries(pathIn,key):
    series=pd.read_hdf(pathIn,key='bars/ES_10k')
    series['Time']=pd.to_datetime(series['Time'],format='%Y%m%d%H%M%S%f')
    series=series.set_index('Time')
    gaps=rollGaps(series)
    for fld in ['Close','VWAP']:series[fld]-=gaps
    return series
#———————————————————————————————————————————
def rollGaps(series,dictio={'Instrument':'FUT_CUR_GEN_TICKER','Open':'PX_OPEN',  'Close':'PX_LAST'},matchEnd=True):
# Compute gaps at each roll, between previous close and next open
    rollDates=series[dictio['Instrument']].drop_duplicates(keep='first').index
    gaps=series[dictio['Close']]*0
    iloc=list(series.index)
    iloc=[iloc.index(i)-1 for i in rollDates] # index of days prior to roll
    gaps.loc[rollDates[1:]]=series[dictio['Open']].loc[rollDates[1:]]- \
    series[dictio['Close']].iloc[iloc[1:]].values
    gaps=gaps.cumsum()
    if matchEnd:gaps-=gaps.iloc[-1] # roll backward
    return gaps




# Symmetric CUSUM filter that filters only after it crosses a threshold value and then resets again\
# Tt is used for bar sampling, which is to be used for training a machine learning model4
# It expects 1-dimensional price series
# Now it is working

def getTEvents(gRaw,h):
    tEvents,sPos,sNeg=[],0,0
    diff=gRaw.diff()/gRaw.shift(1)
    for i in diff.index[1:]:
        if i in h.index: 
            sPos,sNeg=max(0,sPos+diff.loc[i]),min(0,sNeg+diff.loc[i])

            if sNeg<-h.loc[i]:
                sNeg=0;tEvents.append(i)
            elif sPos>h.loc[i]:
                sPos=0;tEvents.append(i)
    return pd.DatetimeIndex(tEvents)



# An estimate for daily volatility, using exponential smoothening
def getDailyVol(close,span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0 - 1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0


def mpNumCoEvents(closeIdx,t1,molecule):
    '''
    Compute the number of concurrent events per bar.
    +molecule[0] is the date of the first event on which the weight will be computed
    +molecule[-1] is the date of the last event on which the weight will be computed
    Any event that starts before t1[modelcule].max() impacts the count.
    '''
    #1) find events that span the period [molecule[0],molecule[-1]]
    t1=t1.fillna(closeIdx[-1]) # unclosed events still must impact other weights
    t1=t1[t1>=molecule[0]] # events that end at or after molecule[0]
    t1=t1.loc[:t1[molecule].max()] # events that start at or before t1[molecule].max()
    #2) count events spanning a bar
    iloc=closeIdx.searchsorted(np.array([t1.index[0],t1.max()]))
    count=pd.Series(0,index=closeIdx[iloc[0]:iloc[1]+1])
    for tIn,tOut in t1.items():count.loc[tIn:tOut]+=1.
    return count.loc[molecule[0]:t1[molecule].max()]


def mpSampleTW(t1,numCoEvents,molecule):
# Derive average uniqueness over the event's lifespan
    wght=pd.Series(index=molecule)
    for tIn,tOut in t1.loc[wght.index].items():
        wght.loc[tIn]=(1./numCoEvents.loc[tIn:tOut]).mean()
    return wght




# Step 1 for Sequential Bootstrapping is to get the indicator matrix
# Bar index is the index of the closing prices I think, t1 is a pandas series with the index being the time at which the feauture happened
# and the values are the times when the features were confirmed
def getIndMatrix(barIx,t1):
    indM=pd.DataFrame(0,index=barIx,columns=range(t1.shape[0]))
    for i,(t0,t1) in enumerate(t1.items()):indM.loc[t0:t1,i]=1.
    return indM


# Step 2 for Sequential Boostrapping is to get the average uniqueness of each observed feauture
# using the indicator matrix from step 1
def getAvgUniqueness(indM):
    c=indM.sum(axis=1) # concurrency
    u=indM.div(c,axis=0) # uniqueness
    avgU=u[u>0].mean() # average uniqueness
    return avgU


# Step 3 for Sequential Bootstrapping is the actual bootstrap
# There is an optional parameter sample length to determine the size, otherwise size is set to number of rows in the IndMatrix

def seqBootstrap(indM,sLength=None):
# Generate a sample via sequential bootstrap
    if sLength is None:sLength=indM.shape[1]
    phi=[]
    while len(phi)<sLength:
        if len(phi) % 10 == 0: print(len(phi))
        avgU=pd.Series()
        for i in indM:
            indM_=indM[phi+[i]] # reduce indM
            avgU.loc[i]=getAvgUniqueness(indM_).iloc[-1]
        prob=avgU/avgU.sum() # draw prob
        phi+=[np.random.choice(indM.columns,p=prob)]
    return phi



# Step 4 optional for Sequential Bootstrapping
# Implements Time Decay for sampling 

def getTimeDecay(tW,clfLastW=1.):
# apply piecewise-linear decay to observed uniqueness (tW)
# newest observation gets weight=1, oldest observation gets weight=clfLastW
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1]
    else:slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    return clfW


# Meta Labling, which is basically labeling when to be long, short or neutral
def applyPtSlOnT1(close,events,ptSl,molecule):
# apply stop loss/profit taking, if it takes place before t1 (end of event)
    events_=events.loc[molecule]
    out=events_[['t1']].copy(deep=True)
   

    if ptSl[0]>0:pt=ptSl[0]*events_['trgt']
    else:pt=pd.Series(index=events.index) # NaNs
    if ptSl[1]>0:sl=-ptSl[1]*events_['trgt']
    else:sl=pd.Series(index=events.index) # NaNs

    #print(pt)
    #print(sl)

    for loc,t1 in events_['t1'].fillna(close.index[-1]).items():
        df0=close[loc:t1] # path prices
        df0=(df0/close[loc]-1)*events_.at[loc,'side'] # path returns
        out.loc[loc,'sl']=df0[df0<sl[loc]].index.min() # earliest stop loss.
        out.loc[loc,'pt']=df0[df0>pt[loc]].index.min() # earliest profit taking.
    return out




def getEvents(close, tEvents, ptSl, trgt, minRet, numThreads, t1=False, side=None):
 
    #1) get target
    trgt=trgt.loc[tEvents]
    trgt=trgt[trgt>minRet] # minRet

    #2) get t1 (max holding period)
    if t1 is False:t1=pd.Series(pd.NaT,index=tEvents)

    #3) form events object, apply stop loss on t1
    if side is None:side_,ptSl_=pd.Series(1.,index=trgt.index),[ptSl[0],ptSl[0]]
    else:side_,ptSl_=side.loc[trgt.index],ptSl[:2]

    events=pd.concat({'t1':t1,'trgt':trgt,'side':side_}, axis=1).dropna(subset=['trgt'])
    events = events.dropna()

  
    df0=mpPandasObj(func=applyPtSlOnT1,pdObj=('molecule', events.index), numThreads= numThreads, close= close, events=events,ptSl=ptSl_)

    #print(df0)
    #events['t1']=df0.dropna(how='all').min(axis=1) # pd.min ignores nan
    #print(events)

    if side is None:events=events.drop('side',axis=1)
    return events


def getBins(events,close):

    #Compute event's outcome (including side information, if provided).
    #events is a DataFrame where:
    #—events.index is event's starttime
    #—events[’t1’] is event's endtime
    #—events[’trgt’] is event's target
    #—events[’side’] (optional) implies the algo's position side
    #Case 1: (’side’ not in events): bin in (-1,1) <—label by price action
    #Case 2: (’side’ in events): bin in (0,1) <—label by pnl (meta-labeling)


    #1) prices aligned with events
    events_= events#.dropna(subset=['t1'])
    px=events_.index.union(events_['t1'].values).drop_duplicates()
    px=close.reindex(px,method='bfill')
    #2) create out object
    out=pd.DataFrame(index=events_.index)
    out['ret']=px.loc[events_['t1'].values].values/px.loc[events_.index]-1
    if 'side' in events_:out['ret']*=events_['side'] # meta-labeling
    out['bin']=np.sign(out['ret'])
    if 'side' in events_:out.loc[out['ret']<=0,'bin']=0 # meta-labeling
    return out




# Multiprocessing function for pandas

# Auxiliary functions
def linParts(numAtoms,numThreads):
    # partition of atoms with a single loop
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,numThreads,upperTriang=False):
    # partition of atoms with an inner loop
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts


def reportProgress(jobNum,numJobs,time0,task):
    # Report progress as asynch jobs are completed
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,task=None,numThreads=24):
    # Run in parallel.
    # jobs must contain a 'func' callback, for expandCall
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    # Expand the arguments of a callback function, kargs['func']
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

def processJobs_(jobs):
    # Run jobs sequentially, for debugging
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out


# This is what allows multiprocessing
def mpPandasObj(func,pdObj,numThreads=24,mpBatches=1,linMols=True,**kargs):
    '''
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
    '''


    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out: df0 = pd.concat(out, ignore_index= True)#for i in out: df0 = pd.concat([df0, i], ignore_index= True)#for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0

if __name__ == "__main__":
    main()