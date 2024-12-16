from ts2vg import HorizontalVG, NaturalVG
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import networkx as nx
import scipy

def shortest_path_length(close: np.array, lookback: int):

    avg_short_dist_p = np.zeros(len(close))
    avg_short_dist_n = np.zeros(len(close))

    avg_short_dist_p[:] = np.nan
    avg_short_dist_n[:] = np.nan

    for i in range(lookback, len(close)):
        dat = close[i - lookback + 1: i+1]

        pos = NaturalVG()
        pos.build(dat)

        neg = NaturalVG()
        neg.build(-dat)

        neg = neg.as_networkx()
        pos = pos.as_networkx()
    
        # you could replace shortest_path_length with other networkx metrics..
        avg_short_dist_p[i] = nx.average_shortest_path_length(pos)
        avg_short_dist_n[i] = nx.average_shortest_path_length(neg)
        
        # Another possibility...
        #nx.degree_assortativity_coefficient(pos)
        #nx.degree_assortativity_coefficient(neg)
        # All kinds of stuff here
        # https://networkx.org/documentation/stable/reference/algorithms/index.html 
    return avg_short_dist_p, avg_short_dist_n

def plot_ts_visibility(network: np.array, data: np.array, times: np.array = None, horizontal: bool = False):
    if times is None:
        times = np.arange(len(data))

    plt.style.use('dark_background') 
    fig, axs = plt.subplots(2, 1, sharex=True)
    # Plot connections and series
    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                if horizontal:
                    axs[0].plot([times[i], times[j]], [data[i], data[i]], color='red', alpha=0.8)
                    axs[0].plot([times[i], times[j]], [data[j], data[j]], color='red', alpha=0.8)
                else:
                    axs[0].plot([times[i], times[j]], [data[i], data[j]], color='red', alpha=0.8)
    axs[0].plot(times, data)
    #axs[0].bar(times, data, width=0.1)
    axs[0].get_xaxis().set_ticks(list(times))

    # Plot graph
    for i in range(len(data)):
        axs[1].plot(times[i], 0, marker='o', color='orange')

    for i in range(len(data)):
        for j in range(i, len(data)):
            if network[i, j] == 1.0:
                Path = mpath.Path
                mid_time = (times[j] + times[i]) / 2.
                diff = abs(times[j] - times[i])
                pp1 = mpatches.PathPatch(Path([(times[i], 0), (mid_time, diff), (times[j], 0)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", transform=axs[1].transData, alpha=0.5)
                axs[1].add_patch(pp1)
    axs[1].get_yaxis().set_ticks([])
    axs[1].get_xaxis().set_ticks(list(times))
    plt.show()




if __name__ == '__main__':
    data = pd.read_csv('trading_view_spy_1_and_2_converted.csv', names = ['open', 'high', 'low', 'close', 'volume'])
    data = data[21000:22000]
    lookback = 12
    close_arr = data['close'].to_numpy()
    pos, neg = shortest_path_length(close_arr, lookback)
    data['pos'] = pos
    data['neg'] = neg

    data = data.dropna().reset_index()

    # Plot visibility graph with max and min avearge_shortest_path
    max_idx = data['pos'].idxmax()
    min_idx = data['pos'].idxmin()

    max_dat = data.iloc[max_idx - lookback +1: max_idx+1]['close'].to_numpy()
    min_dat = data.iloc[min_idx - lookback +1: min_idx+1]['close'].to_numpy()

    g = NaturalVG()
    g.build(max_dat)
    plot_ts_visibility(g.adjacency_matrix(), max_dat)


    g = NaturalVG()
    g.build(min_dat)
    plot_ts_visibility(g.adjacency_matrix(), min_dat)

    (data['close']).plot()
    plt.twinx()
    data['neg'].plot(color='red', label='neg', alpha=0.8)
    data['pos'].plot(color='green',label='pos', alpha=0.8)
    plt.legend()
    plt.show()
