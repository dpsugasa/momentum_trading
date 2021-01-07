
#%%
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

import pandas as pd
import tia.bbg.datamgr as dm
import numpy as np
from datetime import datetime
import plotly
import chart_studio.plotly as py  # for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import credentials
import plotly.io as pio

pio.templates.default = "none"
import os
from pathlib import Path


# create a DataManager for simpler api access
mgr = dm.BbgDataManager()
# set dates, securities, and fields
start_date = "01/01/2005"
end_date = "{:%m/%d/%Y}".format(datetime.now())


# IDs = ['LP01TREU Index',      #Bloomberg Barclays Pan-European High Yield Tot Ret Unhedged EUR
#        'LF98TRUU Index',      #Bloomberg Barclays US Corporate High Yield Tot Ret Unhedged
#        'IBOXXMJA Index',      #Markit iBoxx EUR Liquid High Yield Index TRI
#        'DLJHVAL Index',       #Credit Suisse High Yield Index II
#        'HFRXGLE Index',       #HFR Global Hedge Fund Index EUR
#        'HFRXGL Index',        #HFR Global Hedge Fund Index
#        'HFRXME Index',        #HFR Macro/CTA Index EUR
#        'HFRXM Index',         #HFR Macro/CTA Index
#        'HFRXEWE Index',       #HFR Equal Weighted Strategies EUR
#        'HFRXEW Index',        #HFR Equal Weighted Strateges
#        'HFRXEDE Index',       #Hedge Fund Research Event Driven EUR
#        'HFRXED Index',        #Hedge Fund Research Event Driven
#        'HFRXRVAE Index',      #Hedge Fund Research Relative Value Arbitrage EUR,
#        'HFRXRVA Index',       #Hedge Fund Research Relative Value Arbitrage
#        'I18997EU Index',      #Bloomberg Barclays TR Unhedged EUR ~15 year duration
#        'I18996EU Index',      #Bloomberg Barclays TR Unhedged EUR ~5 year duration
#        'SX5T Index',          #SX5E Total Return
#        'SPXT Index',          #SPX Total Return
#        'NKYCHTE Index',       #NKY Index TRI EUR hedged,
#        'LUATTRUU Index',      #Bloomberg Barclays TR Unhedged Treasury ~5 year duration
#        'LUTLTRUU Index'       #Bloomberg Barclays TR Unhedged Treasury ~15 year duration
#        ]

IDs = {
    "DXY Index": "USD",
    "SPX Index": "SPX",
    "NKY Index": "NKY",
    "CL1 Comdty": "Crude Oil",
    "LUATTRUU Index": "US 5Y",
    "VIX Index": "VIX",
    "BCOM Index": "BBG Commod",
    "HG1 Comdty": "Copper",
    "XAU Curncy": "Gold",
    "EURUSD Curncy": "EUR",
    "GBPUSD Curncy": "GBP",
    "USDJPY Curncy": "JPY",
    #'HYG US Equity': 'HYG',
    "LP01TREU Index": "EUR HY",
    "LF98TRUU Index": "USD HY",
    "SX5E Index": "EuroStoxx",
    "NDX Index": "Nasdaq",
    "RTY Index": "Russell 2000",
    "I18996EU Index": "GER 5Y",
}
sids = mgr[list(IDs.keys())]

fields = ["LAST PRICE"]

df = sids.get_historical(fields, start_date, end_date)
df.columns = df.columns.droplevel(-1)

    
## Create return df
returns = df.pct_change()[1:]

window = 252
## Create correlation function
def mean_cor(df):
    corr_df = df.corr()
    np.fill_diagonal(corr_df.values, np.nan)
    return np.nanmean(corr_df.values)
    
cor_df = pd.DataFrame(index = returns.index[window:])
cor_df['corr'] = [mean_cor(returns.iloc[i-window:i,:]) for i in range(window,len(returns))]

cor_df.plot()
