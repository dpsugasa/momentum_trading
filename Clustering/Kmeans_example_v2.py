from pandas_datareader import data 
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import Normalizer
from collections import OrderedDict



companies_dict = {
    'Amazon':'AMZN',
    'Apple':'AAPL',
    'Walgreen':'WBA',
    'Northrop Grumman':'NOC',
    'Boeing':'BA',
    'Lockheed Martin':'LMT',
    'McDonalds':'MCD',
    'Intel':'INTC',
    'Navistar':'NAV',
    'IBM':'IBM',
    'Texas Instruments':'TXN',
    'MasterCard':'MA',
    'Microsoft':'MSFT',
    'General Electrics':'GE',
    'Symantec':'SYMC',
    'American Express':'AXP',
    'Pepsi':'PEP',
    'Coca Cola':'KO',
    'Johnson & Johnson':'JNJ',
    'Toyota':'TM',
    'Honda':'HMC',
    'Mistubishi':'MSBHY',
    'Sony':'SNE',
    'Exxon':'XOM',
    'Chevron':'CVX',
    'Valero Energy':'VLO',
    'Ford':'F',
    'Bank of America':'BAC'}


data_source = 'yahoo' # Source of data is yahoo finance.
start_date = '2015-01-01' 
end_date = '2017-12-31'
df = data.DataReader(list(companies_dict.values()),data_source,start_date,end_date)

df.head()

stock_open = np.array(df['Open']).T
stock_close = np.array(df['Close']).T

stock_close.shape



movements  = stock_close - stock_open
movements.shape


sum_of_movement = np.sum(movements,1)
for i in range(len(companies_dict)):
  print('company:{}, Change:{}'.format(df['High'].columns[i],sum_of_movement[i]))

import plotly.graph_objects as go

import pandas as pd
from datetime import datetime



fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open']['AMZN'],
                high=df['High']['AMZN'],
                low=df['Low']['AMZN'],
                close=df['Close']['AMZN'])])

fig.show()

a = Normalizer()
norm_movements = a.fit_transform(movements)