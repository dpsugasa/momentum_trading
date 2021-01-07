# import needed modules

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from pathlib import Path
import yfinance as yf
from concurrent import futures
from tia.bbg import LocalTerminal

from statsmodels.stats.stattools import jarque_bera
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.covariance import LedoitWolf
import pandas as pd
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
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import plotting
from pypfopt import risk_models, expected_returns

pio.templates.default = "none"

input_dir = Path(r"D:\OneDrive - Northlight Group")
input_dir.mkdir(parents=True, exist_ok=True)

output_dir = Path(r"D:\OneDrive - Northlight Group\Images\PortfolioWeights")
output_dir.mkdir(parents=True, exist_ok=True)


load_dotenv()

# read tickers, create dataframes, longs, shorts
df_cox = pd.read_csv(fr"{input_dir}\cox_portfolio_output_06_11_2020.csv")
df_cox = df_cox.replace("FALSE", np.nan)
df_cox["carryPtsToTiming"] = df_cox["carryPtsToTiming"].replace("#VALUE!", np.nan)
df_cox.scaledQuantity = pd.to_numeric(df_cox["scaledQuantity"], downcast="float")
df_cox.upsidePrice = pd.to_numeric(df_cox["upPrice"], downcast="float")
df_cox.downsidePrice = pd.to_numeric(df_cox["downPrice"], downcast="float")
df_cox.xLastPrice = pd.to_numeric(df_cox["xLastPrice"], downcast="float")
df_cox.carryPtsToTiming = pd.to_numeric(
    df_cox["carryPtsToTiming"], downcast="float", errors="coerce"
)
df_cox.carryPtsToTiming = df_cox.carryPtsToTiming.fillna(0)
long_df = df_cox.loc[
    (df_cox["foxInstrumentType"] == "BOND")
    & (df_cox["scaledQuantity"] > 0)
    & (df_cox["Score"] > 0)
]
# short_df = df.loc[(df["Class"] == "Corporate Bonds") & (df["Quantity"] < 0)]
# df = df.loc[df["Class"] == "Corporate Bonds"]

# tickers for longs
long_ids = list(long_df["ID_BB_GLOBAL_input"].values)
long_ids = [i + " FIGI" for i in long_ids]
long_names = list(long_df["foxInstrumentName"].values)

long_tickers = dict(zip(long_names, long_ids))
# del long_tickers["MODACIN NM"]
# del long_tickers["NFLX 3 06/15/25"]
# del long_tickers["NSINO 8 02/24/21"]
# del long_tickers["NEXIIM 1 3/4 04/24/27"]
# del long_tickers["VERISR Float 04/15/25"]

#%%

"""
deleting a few additional for ease
"""
# deletions = [
#     "FINRSK 6 7/8 11/15/26",
#     "HEMABV Float 07/15/22",
#     "HEMABV 8 1/2 01/15/23",
#     "WDIGR 0 1/2 09/11/24",
#     "SPCHEM 6 1/2 10/01/26",
#     "SFRFP 8 05/15/27",
# ]

# for i in deletions:
#     del long_tickers[i]

# N = 32
# long_tickers = dict(list(long_tickers.items())[N:])

# #tickers for shorts
# short_ids = list(short_df['Investment'].values)
# short_ids = [i + " FIGI" for i in short_ids]
# short_names = list(short_df['Security'].values)

# short_tickers = dict(zip(short_names, short_ids))
# del short_tickers['ASTONM 6 1/2 04/15/22']
# del short_tickers['ASTONM 5 3/4 04/15/22']
# del short_tickers['SFRFP 8 05/15/27']

# #tickers for longs
# ids = list(df['Investment'].values)
# ids = [i + " FIGI" for i in ids]
# names = list(df['Security'].values)

# tickers = dict(zip(names, ids))
# del tickers["MODACIN NM"]
# del tickers['NFLX 3 06/15/25']
# del tickers['NSINO 8 02/24/21']
# del tickers['NEXIIM 1 3/4 04/24/27']
# del tickers['VERISR Float 04/15/25']

# build initial dataframe
# set dates, securities, and fields
start_date = "10/01/2019"
end_date = "{:%m/%d/%Y}".format(datetime.now())


market = {
    "BBG Barclays US HY": "LF98TRUU Index",
    "Crude Oil": "CL1 Comdty",
    "2s_10s": "USYC2Y10 Index",
    "USD_Index": "DXY Index",
    "IHYG": "IHYG LN Equity",
    "Oil_Equipment_Services": "XES US Equity",
    "Oil_E": "XOP US Equity",
    "OIH ETF": "OIH US Equity",
}  # BBG Barclays US HY

cfields = ["LAST PRICE"]

df = LocalTerminal.get_historical(
    list(long_tickers.values()), cfields, start_date, end_date, period="DAILY"
).as_frame()
df.columns = df.columns.droplevel(-1)
#%%
for i, j in long_tickers.items():
    df = df.rename(columns={j: i})

#%%
# df_price = df.copy().dropna()
# df = df.pct_change()

sector_mapper = {}
sector_upper = {}
sector_lower = {}
ups = {}
down = {}
carr = {}
curr = {}
exp_ret_upside = {}
skew = {}

for i in df.columns:
    sector_mapper[i] = str(
        long_df.loc[long_df["foxInstrumentName"] == i]["foxSector"].values[0]
    )
    upside = long_df.loc[long_df["foxInstrumentName"] == i]["upsidePrice"].values[0]
    ups[i] = upside
    current = long_df.loc[long_df["foxInstrumentName"] == i]["xLastPrice"].values[0]
    curr[i] = current
    carry = long_df.loc[long_df["foxInstrumentName"] == i]["carryPtsToTiming"].values[0]
    carr[i] = carry
    downside = long_df.loc[long_df["foxInstrumentName"] == i]["downsidePrice"].values[0]
    down[i] = downside
    exp_ret_upside[i] = ((upside + carry) - current) / current
    skew[i] = (((upside + carry) - current) / current) / (
        (current - (downside)) / current
    )

exp_ret = pd.Series(exp_ret_upside)
skw = pd.Series(skew)
#%%

sector_upper["Oil & Oil Services"] = 0.25
sector_upper["Consumer & Retail"] = 0.25

mu = expected_returns.mean_historical_return(df)
S = risk_models.CovarianceShrinkage(df).ledoit_wolf()
#plotting.plot_covariance(S)
corr = df.corr()


layout = go.Layout(title='Correlation Heatmap',
                   xaxis=dict(tickfont = dict(size = 6)
                              ),
                   yaxis=dict(tickfont = dict(size = 6)
                              ),
                   showlegend=True,
                   )

fig = go.Figure(data=go.Heatmap(
    z=corr,
    x=df.columns,
    y=df.columns,
    ), layout = layout)

fig.show()
fig.write_image(fr"{output_dir}\CorrelationHeatmap.png")
#py.iplot(fig, filename='Beta/Untreated_Heatmap')


#fig.write_image(fr"{output_dir3}\{name}_range_graph.png")


# min volatility portfolio
ef_minvol = EfficientFrontier(exp_ret, S)
# ef.add_objective(objective_functions.transaction_cost,
#                  w_prev=initial_weights, k=0.01)
ef_minvol.min_volatility()
weights_minvol = ef_minvol.clean_weights()
weights_minvol
ef_minvol.portfolio_performance(verbose=True)

# max sharpe portfolio
ef_maxsharpe = EfficientFrontier(exp_ret, S)
ef_maxsharpe.max_sharpe(risk_free_rate=0.005)
weights_maxsharpe = ef_maxsharpe.clean_weights()
weights_maxsharpe
ef_maxsharpe.portfolio_performance(verbose=True)

# max quadratic utility
ef_maxquad = EfficientFrontier(exp_ret, S, weight_bounds=(0.025, 0.15))
ef_maxquad.max_quadratic_utility(risk_aversion=1, market_neutral=False)
weights_maxquad = ef_maxquad.clean_weights()
weights_maxquad
ef_maxquad.portfolio_performance(verbose=True)

# efficient risk
ef_effrisk = EfficientFrontier(exp_ret, S, weight_bounds=(0.0, 0.15))
ef_effrisk.efficient_risk(0.15, market_neutral=False)
weights_effrisk = ef_effrisk.clean_weights()
weights_effrisk
ef_effrisk.portfolio_performance(verbose=True)

# efficient return
ef_effret = EfficientFrontier(exp_ret, S, weight_bounds=(0.03, 0.15))
ef_effret.efficient_return(target_return=0.30, market_neutral=False)
weights_effret = ef_effret.clean_weights()
weights_effret
ef_effret.portfolio_performance(verbose=True)


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


ef_skew = EfficientFrontier(exp_ret, S, weight_bounds=(0.0, 0.15))
ef_skew.add_sector_constraints(sector_mapper, sector_lower, sector_upper)
ef_skew.add_objective(objective_functions.L2_reg, gamma=20)
#ef.add_objective(minimize_negative_skew, skew= skw)
ef_skew.convex_objective(minimize_negative_skew, skew=skw)
#ef.max_sharpe()
weights_skew = ef_skew.clean_weights()
weights_skew
ef_skew.portfolio_performance(verbose=True)

# %%

weight_dict = { 
    "Miniumum Variance" : weights_minvol,
    "Maximum Sharpe Ratio" : weights_maxsharpe,
    "Maximum Quadratic Utility" : weights_maxquad, 
    "Efficient Target Risk" : weights_effrisk,
    "Efficient Target Return" : weights_effret, 
    "Maximum Skew" : weights_skew
}

for i, j in weight_dict.items():
    trace = go.Bar(
                    x = list(j.keys()),
                    y = list(j.values())
    )
    layout = go.Layout(title= i,
                   xaxis=dict(tickfont = dict(size = 7),
                                tickangle = 45 
                              ),
                   yaxis=dict(#tickfont = dict(size = 6),
                               title = "Weights" 
                              ),
                   showlegend=False,
                   )
    fig = go.Figure(data = trace, layout = layout)
    fig.show()
    fig.write_image(fr"{output_dir}\{i}_chart.png")

'''
calculate historical performance
'''

hist_start_date = "08/01/2019"
hist_end_date = "09/30/2019"
market = "IHYG LN Equity"

df_hist = LocalTerminal.get_historical(
    list(long_tickers.values()), cfields, hist_start_date, hist_end_date, period="DAILY"
).as_frame()
df_hist.columns = df_hist.columns.droplevel(-1)
df_mkt = LocalTerminal.get_historical(market, cfields, hist_start_date, hist_end_date, period="DAILY"
).as_frame()
df_mkt.columns = df_mkt.columns.droplevel(-1)
df_hist_total = pd.concat([df_hist, df_mkt], join="outer", axis =1)
#%%
for i, j in long_tickers.items():
    df_hist_total = df_hist_total.rename(columns={j: i})

df_hist_total = df_hist_total.fillna(method = 'ffill')
df_hist_total = df_hist_total.fillna(value = market)



