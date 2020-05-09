
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tia.bbg import LocalTerminal
from pathlib import Path
import yfinance as yf
from concurrent import futures

output_dir = Path(r"D:\OneDrive - Northlight Group\Data\ETF")
output_dir.mkdir(parents=True, exist_ok=True)


tickers = ['SPY',
            'TLT',
            'GLD',
            'QQQ',
            'IWM',
            'LQD',
            'XLE',
            'HYG',
            'JNK',
            'USO',
            'VWO',
            'EFA',
            'IJH',
            'VUG',
            'VTV',
            'IJR',
            'USMV',
            'XLK',
            'XLF',
            'QUAL',
            'SDY',
            'IEF',
            'XLU',
            'XLY',
            'MTUM',
            'VLUE',
            'DBC'
]


def download(ticker):
    # takes a list
    df = LocalTerminal.get_historical(
        ticker, fields, start, end, period="DAILY"
    ).as_frame()
    df.columns = df.columns.droplevel()
    df = df.rename(
        columns={
            "OPEN": "open",
            "HIGH": "high",
            "LOW": "low",
            "LAST PRICE": "close",
            "VOLUME": "volume",
        }
    ).dropna()
    ticker = ticker.replace("/", ".")
    df.to_csv(fr"{output_dir}/{ticker}.csv")



def download_yf(ticker):
    #takes a list
    stock = yf.Ticker(ticker)
    df = stock.history(period = 'max',
                        freq = 'daily',
                        auto_adjust = True)
    df = df.rename(
        columns = {
            "Open"  : "open",
            "High"  : "high",
            "Low"   : "low",
            "Close" :  "close",
            "Volume" :  "volume",
            "Dividends" : "dividend",
            "Stock Splits" : "split",
            "Date"  : "date"
            }
    ) #.dropna()
    df["dividend"] = 0.0
    df['split'] = 1.0

    df.to_csv(fr"{output_dir}\{ticker}.csv", header=True, index=True)
    return df
            

with futures.ThreadPoolExecutor(max_workers= 50) as executor:
    res = executor.map(download_yf, tickers, timeout=None, chunksize=1)







# def download_csv_data(ticker, start_date, end_date, freq, path):

#     df = yf.Ticker(ticker)

#     hist = df.history(start_date, end_date, freq)
#     #divs = df.dividends
#     #splits = df.splits
#     #df_div = yahoo_financials.get_daily_dividend_data(start_date, end_date)
#     # df = pd.DataFrame(df[ticker]['prices']).drop(['date'], axis=1) \
#     #     .rename(columns={'formatted_date': 'date'}) \
#     #     .loc[:, ['date', 'open', 'high', 'low', 'close', 'volume']] \
#     #     .set_index('date')
#     # df.index = pd.to_datetime(df.index)
#     hist['Dividends'] = 0
#     hist['split'] = 1

#     # save data to csv for later ingestion
#     hist.to_csv(path, header=True, index=True)

#     # # plot the time series
#     # df.close.plot(
#     #     title='{} prices --- {}:{}'.format(ticker, start_date, end_date))
#     return hist









# download_csv_data(ticker='AAPL',
#                   start_date='2017-01-01',
#                   end_date='2020-04-30',
#                   freq='daily',
#                   path= fr'{output_dir}\abn.csv')


# download_csv_data(ticker='TLT', 
#                   start_date='2017-01-01', 
#                   end_date='2020-04-30', 
#                   freq='daily', 
#                   path= fr'{output_dir}\tlt.csv')

# df = yf.Ticker("SPY TLT")

# # get stock info
# #msft.info

# # get historical market data
# hist = yf.download(tickers = "SPY TLT", period='max', interval = "1d", auto_adjust = False)

# # show actions (dividends, splits)
# df.actions

# # show dividends
# df.dividends

# # show splits
# df.splits




