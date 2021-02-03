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

pio.templates.default = "none"

input_dir = Path(r"D:\OneDrive - Northlight Group")
input_dir.mkdir(parents=True, exist_ok=True)

load_dotenv()

#read tickers, create dataframes, longs, shorts
df = pd.read_csv(fr"{input_dir}\Portfolio_Bond_Loan_24042020.csv")
long_df = df.loc[(df["Class"] == "Corporate Bonds") & (df["Quantity"] > 0)]
#short_df = df.loc[(df["Class"] == "Corporate Bonds") & (df["Quantity"] < 0)]
#df = df.loc[df["Class"] == "Corporate Bonds"]

#tickers for longs
long_ids = list(long_df['Investment'].values)
long_ids = [i + " FIGI" for i in long_ids]
long_names = list(long_df['Security'].values)

long_tickers = dict(zip(long_names, long_ids))
del long_tickers["MODACIN NM"]
del long_tickers['NFLX 3 06/15/25']
del long_tickers['NSINO 8 02/24/21']
del long_tickers['NEXIIM 1 3/4 04/24/27']
del long_tickers['VERISR Float 04/15/25']

'''
deleting a few additional for ease
'''
deletions = ['FINRSK 6 7/8 11/15/26',
                 'HEMABV Float 07/15/22',
                 'HEMABV 8 1/2 01/15/23',
                 'WDIGR 0 1/2 09/11/24',
                 'SPCHEM 6 1/2 10/01/26',
                 'SFRFP 8 05/15/27'
]

for i in deletions:
    del long_tickers[i]

N=32
long_tickers = dict(list(long_tickers.items())[N:])

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
start_date = "01/04/2019"
end_date =  "02/28/2020" #"{:%m/%d/%Y}".format(datetime.now())


market = {"BBG Barclays US HY": "LF98TRUU Index",
          "Crude Oil": "CL1 Comdty",
          "2s_10s": "USYC2Y10 Index",
          "USD_Index": "DXY Index",
          "IHYG": "IHYG LN Equity",
          "Oil_Equipment_Services": "XES US Equity",
          "Oil_E": "XOP US Equity",
          "OIH ETF": "OIH US Equity"
          }  # BBG Barclays US HY

cfields = ["LAST PRICE"]

df = LocalTerminal.get_historical(
    list(long_tickers.values()), cfields, start_date, end_date, period="DAILY"
).as_frame()
df.columns = df.columns.droplevel(-1)
#%%
for i, j in long_tickers.items():
    df = df.rename(columns={j: i})

df_price = df.copy().dropna()
#df = df.pct_change()

'''
selected = ['CNP', 'F', 'WMT', 'GE', 'TSLA', 'SPY', 'QQQ', 'IWM']
select_string = ' '.join(selected)

def download_yf(long_tickers):
    df = yf.download(long_tickers=long_tickers,
                        period = 'max',
                        freq = 'daily', 
                        auto_adjust = True,
                        threads = True
                        )
    idx = pd.IndexSlice
    df = df.loc[:, idx["Close", :]]
    df.columns = df.columns.droplevel()
    df = df.dropna()                  
    return df

table = download_yf(select_string)
'''

# calculate daily and annual returns of the stocks
returns_daily = df.tail(252).pct_change()  #one year of returns
returns_annual = returns_daily.mean() * 252

returns_exp = returns_annual.copy()
exp_ret = [0.10, 0.06, 0.25, 0.40, 0.15]
#returns_exp['exp_ret'] = exp_ret
expected_return = pd.DataFrame(index = returns_annual.index, columns = ['exp_ret'], data = exp_ret)
downside = [0.5, 0.04, 0.20, 0.40, 0.10]
expected_return['downside'] = downside
liquidity = [0.40, 0.60, 1.00, 0.80, 0.20]
expected_return['liq'] = liquidity
conv = [1.00, 0.60, 0.80, 1.0, 0.4]
expected_return['conv'] = conv
#%%
# get daily and covariance of returns of the stock
cov_daily = returns_daily.cov()
cov_annual = cov_daily * 252

# empty lists to store returns, volatility and weights of imiginary portfolios
port_returns = []
port_volatility = []
sharpe_ratio = []
stock_weights = []

# set the number of combinations for imaginary portfolios
num_assets = len(long_tickers)
num_portfolios = 50000

#set random seed for reproduction's sake
np.random.seed(101)

# populate the empty lists with each portfolios returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, returns_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# extend original dictionary to accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(long_tickers):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# make a nice dataframe of the extended dictionary
df = pd.DataFrame(portfolio)

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in long_tickers]

# reorder dataframe columns
df = df[column_order]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

# find min Volatility & max sharpe values in the dataframe (df)
min_volatility = df['Volatility'].min()
max_sharpe = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == max_sharpe]
min_variance_port = df.loc[df['Volatility'] == min_volatility]

# plot frontier, max sharpe & min Volatility values with a scatterplot
plt.style.use('seaborn-dark')
df.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                cmap='RdYlGn', edgecolors='black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()

# print the details of the 2 special portfolios
print(min_variance_port.T)
print(sharpe_portfolio.T)



