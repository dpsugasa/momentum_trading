# %%
'''
Sample CTA in 10 line of code (Not really 10 but pretty short!)

https://www.linkedin.com/pulse/implement-cta-less-than-10-lines-code-thomas-schmelzer
'''


import pandas as pd
import numpy as np
from statsmodels.stats.stattools import jarque_bera
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf
from tia.bbg import LocalTerminal
from datetime import datetime
from operator import itemgetter
from scipy import stats
from scipy.stats import norm
from IPython.display import IFrame
import statsmodels.api as sm
from statsmodels import regression
from pyfinance.ols import PandasRollingOLS, OLS
import plotly
import chart_studio.plotly as py  # for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio

pio.templates.default = "none"

# %%

if __name__ == "__main__":

    # we recommend to use log prices instead of prices! Oscillator designed for additive prices!
    # build initial dataframe
    # set dates, securities, and fields
    start_date = "01/01/1990"
    end_date = "{:%m/%d/%Y}".format(datetime.now())

    tickers = {
        "SPX": "ES1 Index",  # Trinseo 5.375% 09/01/25
        "SX5E": "VG1 Index",  # Ineos 2.125% 11/15/25
        "Oil": "CL1 Comdty",
        "TY": "TY1 Comdty",
        # Ineos 5.625% 08/01/24
    }

    cfields = ["LAST PRICE"]

    fut = LocalTerminal.get_historical(
        list(tickers.values()), cfields, start_date, end_date, period="DAILY"
    ).as_frame()
    fut.columns = fut.columns.droplevel(-1)

    # fut = (
    #     pd.read_csv("data/prices.csv", index_col=0, parse_dates=True)
    #     .ffill()
    #     .truncate(before=pd.Timestamp("1990-01-01"))
    # )
    # compute volatility adjusted returns and winsorize them
    volatility = np.log(fut).diff().ewm(com=32).std()
    # move back into a "price"-space by accumulating filtered log returns
    prices = (np.log(fut).diff() / volatility).clip(-4.2, 4.2).cumsum()

    # compute the oscillator
    def osc(prices, fast=32, slow=96):
        f, g = 1 - 1 / fast, 1 - 1 / slow
        return (
            prices.ewm(span=2 * fast - 1).mean() -
            prices.ewm(span=2 * slow - 1).mean()
        ) / np.sqrt(1.0 / (1 - f * f) - 2.0 / (1 - f * g) + 1.0 / (1 - g * g))

    # compute the currency position and apply again some winsorizing to avoid extremely large positions, strictly speaking volatility of percent returns needed here...
    CurrencyPosition = (50000 * np.tanh(osc(prices)) /
                        volatility).clip(-5e7, 5e7)

    # the profit today is the return today times the position of yesterday
    Profit = (fut.pct_change() * CurrencyPosition.shift(periods=1)).sum(axis=1)

    # simulate the compounding over time
    (1 + Profit / 7e7).cumprod().plot(logy=True, grid=True)
# %%
