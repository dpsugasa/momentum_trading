
'''
functions to use Yahoo! Finance or Bloomberg for data.
Requires Python 3.6 to run because of f-strings
'''

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tia.bbg import LocalTerminal
from pathlib import Path
import yfinance as yf
from concurrent import futures
import os
from dotenv import load_dotenv
load_dotenv()

output_dir = Path(os.getenv('data_path_etf'))
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











