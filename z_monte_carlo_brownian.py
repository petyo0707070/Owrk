import numpy as np


def monte_carlo_price(S0, K, T, r, q, sigma, steps, N, is_put = False):
    """
    Inputs
    #S = Current stock Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = risk free interest rate
    #q = dividend yield
    # sigma = volatility

    Output
    # [steps,N] Matrix of asset paths
    """
    dt = T / steps
    # S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
    ST = np.log(S0) + np.cumsum(((r - q - sigma ** 2 / 2) * dt + \
                                sigma * np.sqrt(dt) * \
                                np.random.normal(size=(steps, N))), axis=0)

    paths = np.exp(ST)

    if is_put == False:
        payoffs = np.maximum(paths[-1] - K, 0)
    else:
        payoffs = np.maximum(K - paths[-1], 0)

    option_price = np.mean(payoffs) * np.exp( -r * T)
    return option_price

call_price = monte_carlo_price(S0 = 97718, K = 98000, T = 0.84375/365.25, r = 0.04326, q = 0, sigma = 0.55567, steps = 100, N = 1000000, is_put=False )
print(call_price)