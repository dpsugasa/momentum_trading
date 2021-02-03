import pandas as pd
import numpy as np

from xbbg import blp, pipeline

blp.__version__

blp.bdh(
    tickers='SHCOMP Index', flds=['high', 'low', 'last_price'],
    start_date='2019-11', end_date='2020', Per='W', Fill='P', Days='A',
)

cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(end=cur_dt, periods=2, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))
pre_dt.date()

blp.bdtick('QQQ US Equity', dt=pre_dt).tail(10)

async for snap in blp.live(['ESA Index', 'NQA Index']):
    print(snap)

cur_dt = pd.Timestamp('today', tz='America/New_York').date()
recent = pd.bdate_range(end=cur_dt, periods=2, tz='America/New_York')
pre_dt = max(filter(lambda dd: dd < cur_dt, recent))

fx = blp.bdib('JPY Curncy', dt=pre_dt)
jp = pd.concat([
    blp.bdib(ticker, dt=pre_dt, session='day')
    for ticker in ['7974 JP Equity', '9984 JP Equity']
], axis=1)
jp.tail()