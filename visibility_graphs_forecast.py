from ts2vg import HorizontalVG, NaturalVG
import pandas as pd
import numpy as np
import networkx as nx

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


if __name__ == '__main__':
    data = pd.read_csv('trading_view_spy_1_and_2_converted.csv', names = ['open', 'high', 'low', 'close', 'volume'])
    data = data['close'][11:312]

    data = data.to_numpy()

    network = ts_to_vg(data)

    #network = network.astype(int)
    #index = list(range(len(data)))
    #index = [str(x) for x in index] 
    #print("    " + " ".join(index))
    #print("    " + "-" * (len(data) * 2 - 1))
    #for i in range(len(data)):
    #    row = f"{i} | {str(network[:, i])[1:-1]}"
        #print(row)

    print(data[-2])
    print(f"Prediction {(network_prediction(network, data)/100 + 1) * data[-2]}")
    print(data[-1])


