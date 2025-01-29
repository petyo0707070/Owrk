import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import sys
from copulae.elliptical import StudentCopula
import plotly.graph_objs as go



def main():

    # Load the data
    iygb = pd.read_csv("iygb.csv", header = 0, index_col = 0)
    qw9a = pd.read_csv("qw9a.csv", header = 0, index_col=0)

    #Split the data 64% for training, 18 % for validation and 18% for testing
    iygb_validation_test = iygb[int(0.64 * len(iygb)):]
    iygb_validation = iygb_validation_test[:int(0.5 * len(iygb_validation_test))]
    iygb_test = iygb_validation_test[int(0.5 * len(iygb_validation_test)):]
    del iygb_validation_test
    iygb = iygb[:int(0.64 * len(iygb))]

    qw9a_validation_test = qw9a[int(0.64 * len(qw9a)):]
    qw9a_validation = qw9a_validation_test[:int(0.5 * len(qw9a_validation_test))]
    qw9a_test = qw9a_validation_test[int(0.5 * len(qw9a_validation_test)):]
    del qw9a_validation_test
    qw9a = qw9a[:int(0.64 * len(qw9a))]


    # Drop the dates that are present only in one of the two dataframes
    qw9a = qw9a[qw9a["date"].isin(iygb["date"])]


    # Reset the index
    iygb = iygb.reset_index(drop = True)
    qw9a = qw9a.reset_index(drop = True)

    # Calculate the returns and set the index as a date
    iygb["date"] = pd.to_datetime(iygb["date"])
    iygb.set_index("date", inplace= True)

    qw9a["date"] = pd.to_datetime(qw9a["date"])
    qw9a.set_index("date", inplace= True)

    iygb["return"] = iygb["PX_LAST"].diff() / iygb["PX_LAST"].shift(1)
    qw9a["return"] = qw9a["PX_LAST"].diff() / qw9a["PX_LAST"].shift(1)

    iygb = iygb.dropna()
    qw9a = qw9a.dropna()

    plot_probability_distributions(iygb["return"], qw9a["return"])

    # This fits a cdf for the empirical distribution of the training sets for the 2 indexes
    cdf_iygb = ECDF(iygb["return"].values)
    cdf_qw9a = ECDF(qw9a["return"].values)

    # The idea of the copula strategy is to go long A and short B
    # when cdf(A) = 5% and cdf(B) = 95% and vice versa, percentages are subject to the
    # confidence level we choose. There is a choince whether to use price or returns

    iygb["U"] = iygb["return"].apply(lambda x: cdf_iygb(x))
    qw9a["U"] = qw9a["return"].apply(lambda x: cdf_qw9a(x))

    df = pd.merge(iygb["U"], qw9a["U"], left_index=True, right_index=True)

    """
    cdf_iygb_prices = ECDF(iygb["PX_LAST"].values)
    cdf_qw9a_prices = ECDF(qw9a["PX_LAST"].values)
    iygb["U_1"] = iygb["PX_LAST"].apply(lambda x: cdf_iygb_prices(x))
    qw9a["U_1"] = qw9a["PX_LAST"].apply(lambda x: cdf_qw9a_prices(x))
    plt.scatter(iygb["U_1"], qw9a["U_1"])
    plt.title("Scatterplot prices")
    plt.show()
    sys.exit()
    """

    copula = StudentCopula()
    copula.fit(df)
    print(copula.params)
    print(copula.log_lik(df))






    # The 2 indexes are almost 1:1 correlated
    plt.scatter(iygb["return"], qw9a["return"])
    plt.title("Scatterplot returns")
    plt.show()


    plt.scatter(iygb["U"], qw9a["U"])
    plt.title("Scatterplot cdfs")
    plt.show()


# The timeseries are NOT cointegrated
def run_cointegration_test(y, x):
    import statsmodels.api as sm
    import statsmodels.tsa.stattools as ts


    x_y_df = pd.merge(x,y, left_index = True, right_index= True)

    x1 = x_y_df["PX_LAST_x"]
    y1 = x_y_df["PX_LAST_y"]

    result = ts.coint(y1, x1, trend = "ctt", maxlag= 5, autolag="t-stat")
    print(result)

def plot_probability_distributions(x, y):
    from scipy.stats import gaussian_kde
    from numpy import linspace

    kernel_x = gaussian_kde(x.values)
    dist_space_x = linspace(min(x.values), max(x.values), 100)

    kernel_y = gaussian_kde(y.values)
    dist_space_y = linspace(min(y.values), max(x.values), 100)

    plt.plot(dist_space_x, kernel_x(dist_space_x), label = "IYGB (Covered)")
    plt.plot(dist_space_y, kernel_y(dist_space_y), label = "QW9A (SSA)")
    plt.legend()

    plt.show()


# inverse cdf i.e. quantile function
def quantile(x, q):
    try:
        sample = x.values
    except:
        sample = x
    sample = sorted(sample)
    return sample[int(len(sample) * q)]

# Evaluating correlation dependence is needed when looking which copula to select
# The idea is that we reduce q and estimate coefficients of correlation with q going towards 0
# and we check how the correlation evolves. Important this needs to be done for the lower tail for both combined
# and separetely for the upper tail for both combined

"""
First important finding is that the tails to the downside are almost perfectly correlated,
while to the upside it is volatile , lowest it goes is 90% but still good  to know
"""
def evaluate_correlation_dependence(x, y):
    # Input needs to be a dataframe unfortunately

    lower_tail_correlations = []
    upper_tail_corrleations = []

    #This will be used for the labels
    x_label_positions = [i for i in range(1,50,5)]
    x_labels = [i/1000 for i in range(1,50,5)]

    for q in range(1, 50):
        q = q / 1000
        lower_tail_correlations.append(x["return"][0:int(q * len(x) )].corr
                                       (y["return"][0:int(q * len(y) )]))
        upper_tail_corrleations.append(x["return"][int( (1 - q) * len(x) ):].corr
                                       (y["return"][int( (1 -q)  * len(y) ):]))


    plt.plot(lower_tail_correlations, label = "Lower tail")
    plt.plot(upper_tail_corrleations, label = "Upper tail")
    #plt.plot(2 - pow(2, 1/q), label = "Theoretical Gumbel Dependence")

    plt.title("Tail Correlations based on q")
    plt.xticks(x_label_positions, x_labels)
    plt.legend()
    plt.show()

# This will be the more accurate way to select a copula, correlations are not sufficient
def evaluate_tail_dependence(x, y):
    pass


if __name__ == "__main__":
    main()