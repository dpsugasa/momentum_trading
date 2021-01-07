### [Python]
## Load libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas_datareader as dr

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/user_name/Anaconda3/Library/plugins/platforms'
# This last bit of code resolve a "QT" error we couldn't figure out. Depending on enviroment, you'll probablybe ussing a differnt path

# Note: Although many users recommend using use_condaenv(), that gave us a not recognized error. So we omitted it without any issues.

plt.style.use('ggplot')

## Load prices
# Get tickers
url = "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xli.xlsx"
tickers = pd.read_excel(url, skiprows=4)

ticker_list = list(tickers['Ticker'][:66])
ticker_list = [ticker.replace(".", "-") for ticker in ticker_list]
ticker_list = [ticker for ticker in ticker_list if ticker not in ["OTIS", "CARR"]]

# Download prices
start_date = "2005-01-01"
end_date = "2019-12-31"

try:
    prices = pd.read_pickle('xli_prices.pkl')
    print('Data loaded')
except FileNotFoundError:
    print("Data not found. Downloading...")
    prices = dr.DataReader(ticker_list,"yahoo", start_date, end_date)
    prices.to_pickle('xli_prices.pkl')
    
## Create return df
returns = prices.pct_change()[1:]

## Create correlation function
def mean_cor(df):
    corr_df = df.corr()
    np.fill_diagonal(corr_df.values, np.nan)
    return np.nanmean(corr_df.values)
    
cor_df = pd.DataFrame(index = returns.index[60:])
cor_df['corr'] = [mean_cor(returns.iloc[i-60:i,:]) for i in range(60,len(returns))]

## Create training df
tr_idx = round(len(cor_df)*.7)
train = cor_df[:tr_idx]
test = cor_df[tr_idx:]

## Plot correlation
train.plot(color = "darkblue")
plt.xlabel("")
plt.ylabel("Correlation")
plt.title("Rolling three-month correlation among S&P 500 industrials")
plt.show()

## Stats
train_mean = np.round(train['corr'].mean(),2)*100
train_max = np.round(train['corr'].max(),2)*100
train_min = np.round(train['corr'].min(),2)*100
train_09_max = np.round(train.loc["2009",'corr'].max(),2)*100

## Load XLI prices
start_date = "2005-01-01"
end_date = "2019-12-31"

xli = dr.DataReader("XLI", "yahoo", start_date, end_date)['Adj Close']
xli_ret = xli.pct_change()[1:]

xli_df = pd.DataFrame(xli_ret)
xli_df.columns = ['return']

## Merge XLI with train df
train = pd.merge(train, xli_df, how="left", on = 'Date')
train['cum_ret'] = (1+train['return']).cumprod()

## Graph correlations and index 
train_graf = train.copy()
train_graf.columns = ['Correlation', 'Return','Cumulative return']

ax = (train_graf[['Correlation','Cumulative return']]*100).plot(secondary_y = ['Correlation'], 
                                                          color=['darkblue','black'],
                                                          mark_right = False)
ax.set_ylabel("Return")
ax.right_ax.set_ylabel('Correlation')
ax.set_xlabel("")
plt.title("Rolling correlation vs. cumulative return S&P 500 industrials sector")
plt.show()

## Create forward return dfs
xli_30 = xli.pct_change(20).shift(-20)
xli_90 = xli.pct_change(60).shift(-60)

## Gragh correlation vs three-month foward return
train_graf['Return'] = xli_90['2005-04-01':'2015-07-29'].values

ax = (train_graf[['Return', 'Correlation']]*100).plot(secondary_y = ['Correlation'], 
                                                          color=['black', 'darkblue'],
                                                          mark_right = False)
ax.set_ylabel("Return (%)")
ax.right_ax.set_ylabel('Correlation (%)')
ax.set_xlabel("")
lines = ax.get_lines() + ax.right_ax.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc='upper left')
ax.set_title('Rolling correlation vs. next 90-day return S&P 500 industrials')
plt.show()

## Run linear regression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X_train = train['corr'].values.reshape(-1,1)[:-1]
y_train = train['return'].shift(-1).values[:-1]

lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
pred = lin_reg.predict(X_train)

rmse = np.sqrt(mean_squared_error(y_train, pred))
rmse_sc = rmse/np.std(y_train)
