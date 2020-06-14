
#%%
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import plotting
from pypfopt import risk_models, expected_returns
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# small portfolio to test
tickers = ['BLK', "BAC", "AAPL", "TM", "WMT",
           "JD"]  # "INTU", "MA", "UL", "CVS",
# "DIS", "AMD", "NVDA", "PBI", "TGT"]

ohlc = yf.download(tickers, period="max")

prices = ohlc["Adj Close"]
prices.tail()

#various types of expected returns; these will be used for expected future returns
mu = expected_returns.james_stein_shrinkage(prices)
mum = expected_returns.mean_historical_return(prices)
mummy = expected_returns.capm_return(prices)

#different risk models; guessing that you can also use the pandas.cov function
S = risk_models.semicovariance(prices)
T = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

plotting.plot_covariance(S)
plotting.plot_covariance(T)

#equal weights
initial_weights = np.array([1/len(tickers)]*len(tickers))
print(initial_weights)

#transaction cost objective
ef = EfficientFrontier(mum, T)
# 1% broker commission
ef.add_objective(objective_functions.transaction_cost,
                 w_prev=initial_weights, k=0.01)
ef.min_volatility()
weights = ef.clean_weights()
weights

# smaller broker comms
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.transaction_cost,
                 w_prev=initial_weights, k=0.001)
ef.min_volatility()
weights = ef.clean_weights()
weights

#limit number of zero weights
ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.transaction_cost,
                 w_prev=initial_weights, k=0.001)
ef.add_objective(objective_functions.L2_reg)
ef.min_volatility()
weights = ef.clean_weights()
weights

ef = EfficientFrontier(mu, S)
ef.add_objective(objective_functions.transaction_cost,
                 w_prev=initial_weights, k=0.001)
ef.add_objective(objective_functions.L2_reg, gamma=0.05)  # default is 1
ef.min_volatility()
weights = ef.clean_weights()
weights

# min volatility portfolio
ef = EfficientFrontier(mum, T)
ef.add_objective(objective_functions.transaction_cost,
                 w_prev=initial_weights, k=0.001)
ef.add_objective(objective_functions.L2_reg, gamma=0.05)  # default is 1
ef.min_volatility()
weights = ef.clean_weights()
weights

sector_mapper = {
    "BLK": "Oil",
    "BAC": "Banks",
    "AAPL": "Tech",
    "TM": "Oil",
    "WMT": "Retail",
    "JD": "Oil",
    "XOM": "Energy",
    "PFE": "Healthcare",
    "JPM": "Financial Services",
    "UNH": "Healthcare",
    "ACN": "Misc",
    "DIS": "Media",
    "GILD": "Healthcare",
    "F": "Auto",
    "TSLA": "Auto"
}

sector_lower = {
    #"Consumer Staples": 0.1, # at least 10% to staples
    #"Tech": 0.05 # at least 5% to tech
    # For all other sectors, it will be assumed there is no lower bound
}

sector_upper = {
    "Tech": 0.5,
    "Oil":0.25,
    "Banks": 0.1,
    "Retail":0.50
}

ef = EfficientFrontier(mum, T)

ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)

# amzn_index = ef.tickers.index("AMZN")
# ef.add_constraint(lambda w: w[amzn_index] == 0.10)

# tsla_index = ef.tickers.index("TSLA")
# ef.add_constraint(lambda w: w[tsla_index] <= 0.05)

# ef.add_constraint(lambda w: w[10] >= 0.05)

ef.min_volatility()
weights = ef.clean_weights()
weights

pd.Series(weights).plot.pie(figsize=(10,10))



import cvxpy as cp



skew = {
    "BLK" : 2.5,
    "BAC" : 1.0,
    "AAPL" : 1.5,
    "TM" :  0.5,
    "JD" : 0.25,
    "WMT" : 0.10

}

skw = pd.Series(skew)

def minimize_negative_skew(w, skew, negative=True):
    """
    Calculate the (negative) "Skew" of a portfolio

    :param w: asset weights in the portfolio
    :type w: np.ndarray OR cp.Variable
    :param skew: expected return of each asset
    :type skew: np.ndarray
    :param cov_matrix: covariance matrix
    :type cov_matrix: np.ndarray
    :param negative: whether quantity should be made negative (so we can minimise) 
    :type negative: boolean
    :return: (negative) Sharpe ratio
    :rtype: float
    """
    sk = w @ skew
    sign = -1 if negative else 1
    return sign * sk

# def portfolio_return(w, expected_returns, negative=True):
#     """
#     Calculate the (negative) mean return of a portfolio

#     :param w: asset weights in the portfolio
#     :type w: np.ndarray OR cp.Variable
#     :param expected_returns: expected return of each asset
#     :type expected_returns: np.ndarray
#     :param negative: whether quantity should be made negative (so we can minimise) 
#     :type negative: boolean
#     :return: negative mean return
#     :rtype: float
#     """
#     sign = -1 if negative else 1
#     mu = w @ expected_returns
#     return _objective_value(w, sign * mu)


# def sharpe_ratio(w, expected_returns, cov_matrix, risk_free_rate=0.02, negative=True):
#     """
#     Calculate the (negative) Sharpe ratio of a portfolio

#     :param w: asset weights in the portfolio
#     :type w: np.ndarray OR cp.Variable
#     :param expected_returns: expected return of each asset
#     :type expected_returns: np.ndarray
#     :param cov_matrix: covariance matrix
#     :type cov_matrix: np.ndarray
#     :param risk_free_rate: risk-free rate of borrowing/lending, defaults to 0.02.
#                            The period of the risk-free rate should correspond to the
#                            frequency of expected returns.
#     :type risk_free_rate: float, optional
#     :param negative: whether quantity should be made negative (so we can minimise) 
#     :type negative: boolean
#     :return: (negative) Sharpe ratio
#     :rtype: float
#     """
#     mu = w @ expected_returns
#     sigma = cp.sqrt(cp.quad_form(w, cov_matrix))
#     sign = -1 if negative else 1
#     sharpe = (mu - risk_free_rate) / sigma
#     return _objective_value(w, sign * sharpe)


#try to add a convex objective
ef = EfficientFrontier(mum, T, weight_bounds = (0.025, 0.5))
ef.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
ef.add_objective(minimize_negative_skew, skew= skw)
#ef.convex_objective(minimize_negative_skew, skew = skw)
ef.min_volatility()
weights = ef.clean_weights()
weights

# %%
