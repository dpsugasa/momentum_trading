
#%%

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from pathlib import Path
from xbbg import blp
from pathlib import Path
from tia.bbg import LocalTerminal
#from concurrent import futures
from datetime import datetime, timedelta
import plotly
import chart_studio.plotly as py  # for plotting
import plotly.graph_objs as go
import chart_studio.dashboard_objs as dashboard
import plotly.figure_factory as ff
import credentials  # plotly API details
import plotly.io as pio
pio.templates.default = "none"


%matplotlib inline

output_dir = Path(r"D:\OneDrive - Northlight Group\Trading\spy")
output_dir.mkdir(parents=True, exist_ok=True)

#tickers = pd.read_csv(fr'{output_dir}/tickers.csv', header=None)[1][1:].tolist()
#tickers = [d.replace("/", ".") for d in tickers]

output_dir2 = Path(r"D:\OneDrive - Northlight Group\Trading\spx_tickers")
output_dir2.mkdir(parents=True, exist_ok=True)

data_range = pd.bdate_range(start = "2006-01-01", end = datetime.now(), freq="BM")
dates = ["{:%Y%m%d}".format(date) for date in data_range]

ticker_dict = {}
IDs = ['SPX Index']
fields = ["INDX_MWEIGHT_HIST"]

def download_tickers(date):
    # takes a list
    df = blp.bds(IDs, fields, END_DATE_OVERRIDE = date)
    ticker_dict[date] = df['index_member'].to_list()

for z in dates:
   download_tickers(z)

# Save
#dictionary = {'hello':'world'}
np.save(fr'{output_dir2}/tickers.npy', ticker_dict)

# # Load
# read_dictionary = np.load('my_file.npy',allow_pickle='TRUE').item()
# print(read_dictionary['hello']) # displays "world"

#ticker_dict = np.load(fr'{output_dir2}/tickers.npy', allow_pickle=True).item()

hist_tickers = list(set().union(*ticker_dict.values()))
hist_tickers_bbg = [x + " Equity" for x in hist_tickers]
fields = ["OPEN", "HIGH", "LOW", "LAST PRICE", "VOLUME"]
start = "2000-01-01"
end = "{:%m/%d/%Y}".format(datetime.now())
df = LocalTerminal.get_historical(hist_tickers_bbg, fields, start, end, period = 'DAILY').as_frame()
df = df.reset_index()

#df.to_csv(fr'{output_dir2}/historical_data.csv')
df.to_pickle(fr'{output_dir2}/historical_data.pkl')

#df = pd.read_csv(fr'{output_dir2}/historical_data.csv')
df = pd.read_pickle(fr'{output_dir2}/historical_data.pkl')

#%%

data = {}
date_list = list(ticker_dict.keys())
#fields = ["OPEN", "HIGH", "LOW", "LAST PRICE", "VOLUME"]

#df.set_index('index', inplace=True)
long_names = df.loc[:, [len(i) > 16 for i in df.columns.get_level_values(0)]]
idx = pd.IndexSlice

# ticker_list = ticker_dict[date_list[1]]
# ticker_list = [x + " Equity" for x in ticker_list]
# #df_slice = df.isin(ticker_list)
# start = datetime.strptime(date_list[1], "%Y%m%d").strftime("%m/%d/%Y")
# end = (datetime.strptime(date_list[1+1], "%Y%m%d") - timedelta(days=1)).strftime("%m/%d/%Y")
# df_slice = df.loc[idx[start:end], idx[ticker_list, :]]




for i in range(0, len(date_list)-1):
    start = datetime.strptime(date_list[i], "%Y%m%d").strftime("%m/%d/%Y")
    end = (datetime.strptime(date_list[i+1], "%Y%m%d")).strftime("%m/%d/%Y") #timedelta(days=1)
    ticker_list = [x + " Equity" for x in ticker_dict[date_list[i]]]
    df_slice = df.loc[idx[start:end], idx[ticker_list, "LAST PRICE"]]
    #df.columns = df.columns.droplevel(-1)
    df_slice = df_slice.rename(
        columns={
            #"OPEN": "open",
            #"HIGH": "high",
            #"LOW": "low",
            "LAST PRICE": "close",
            #"VOLUME": "volume"
        }
            )
    df_slice = df_slice.pct_change()
    df_slice = df_slice.groupby(axis=0, level=0).filter(lambda d: ~np.all(d.isna())) 
    data[date_list[i]] = df_slice


frames = data.values()
df_tot = pd.concat(frames, axis = 0, join = 'outer')
#z = df_tot.loc[idx[:], idx[:, 'close']]

#%%
#z = z.pct_change()
#idx = pd.IndexSlice
z_wtf = df_tot.mean(axis=1, skipna=True)

sim_rsp = z_wtf
#sim_rsp = sim_rsp.cumprod()

rsp_start = datetime.strptime(date_list[0], "%Y%m%d").strftime("%m/%d/%Y")
rsp_end = end = "{:%m/%d/%Y}".format(datetime.now())
rsp = LocalTerminal.get_historical("SPXEWTR Index", "LAST PRICE", rsp_start, rsp_end, period = 'DAILY').as_frame()
rsp.columns = rsp.columns.droplevel(-1)
rsp = rsp.pct_change()
rsp = rsp.dropna()
#rsp = rsp.cumprod()

rsp_frames = [sim_rsp, rsp]
rsp_tot = pd.concat(rsp_frames, join='outer', axis=1)
rsp_tot = rsp_tot.dropna()
rsp_tot['SPXEWTR Index'] = rsp_tot['SPXEWTR Index'] #+ (0.002/252)

# rsp = (
#     (web.DataReader("RSP", "yahoo", sim_rsp.index[0], sim_rsp.index[-1])[
#         "Close"
#     ].pct_change() + 0.002 / 252 + 1)  # 0.20% annual ER
#     .cumprod()
#     .rename("RSP")
#)

# sim_rsp.plot(legend=True, title="RSP vs. Un-Survivorship-Biased Strategy", figsize=(12, 9))
# rsp.plot(legend=True)


#realized variance vs. variance swap
trace1 = go.Scatter(
                    x = rsp_tot.index,
                    y = (rsp_tot[0] + 1).cumprod().values,
                    name = 'Sim-RSP'
                    #yaxis = 'y1'
                    # line = dict(
                    #              color = ('#4d4dff'),
                    #             width = 2,
                    #                 ),
                        #fill = 'tonexty',
                        #opacity = 0.05,
                       
                        
    
    ) 
                                    
trace2 = go.Scatter(
                    x = rsp_tot.index,
                    y = (rsp_tot['SPXEWTR Index'] + 1).cumprod().values,
                    name = 'RSP'
                   # yaxis = 'y1',
                    # line = dict(
                    #             color = ('#e6e600'),
                    #             width = 2.0,
                    #             ),
                        #fill = 'tonexty',
                        #opacity = 0.05,
                       
                        
    
    ) 
         
layout  = {'title' : 'RSP vs. Sim-RSP (Un-Survivorship Biased Strategy)',
                   'xaxis' : {'title' : 'Date',
                              'type': 'date',
                              'fixedrange': True},
                   'yaxis' : {'title' : 'Total Return',
                              'fixedrange': True},
#                   'yaxis2' : {'title' : 'Annualized Variance',
#                               'overlaying' : 'y',
#                               'side'   : 'right',
#                               'fixedrange' : True
#                               },
#                   'shapes': [{'type': 'rect',
#                              'x0': d[name]['1m_scr_1y'].truncate(before = '2016-01-01').index[0],
#                              'y0': -2,
#                              'x1': d[name]['1m_scr_1y'].truncate(before = '2016-01-01').index[-1],
#                              'y1': 2,
#                              'name': 'Z-range',
#                              'line': {
#                                      'color': '#f48641',
#                                      'width': 2,},
#                                      'fillcolor': '#f4ad42',
#                                      'opacity': 0.1,
#                                      },],
                #    'margin' : {'l' : 100,
                #                'r' : 100,
                #                'b' : 50,
                #                't' : 50}
                   }
    
data = [trace1, trace2]
figure = go.Figure(data=data, layout=layout)
figure.show()
#py.iplot(figure, filename = f'Variance_Premium/{date_now}/{name}/Var_vs_Rlzd')



# for ticker, df in data.items():
#     df = df.reset_index().drop_duplicates(subset='date').set_index('date')
#     df.to_csv(f"survivorship-free/{fix_ticker(ticker)}.csv")
#     data[ticker] = df
# tickers = [fix_ticker(ticker) for ticker in data.keys()]
# pd.Series(tickers).to_csv("survivorship-free/tickers.csv")


# # calculate cumulative product of the mean of all daily returns
# # i.e. simulate growth of $1 by equally weighting all current S&P 500
# # constituents
# sim_rsp = (
#     (pd.concat(
#         [pd.read_csv(fr"{output_dir}/{ticker}.csv", index_col='date', parse_dates=True)[
#             'close'
#         ].pct_change()
#         for ticker in tickers],
#         axis=1,
#         sort=True,
#     ).mean(axis=1, skipna=True) + 1)
#     .cumprod()
#     .rename("SIM")
# )

# # download actual RSP data
# rsp = (
#     (web.DataReader("RSP", "yahoo", sim_rsp.index[0], sim_rsp.index[-1])[
#         "Adj Close"
#     ].pct_change() + 0.002 / 252 + 1)  # 0.20% annual ER
#     .cumprod()
#     .rename("RSP")
# )

# sim_rsp.plot(legend=True, title="RSP vs. Survivorship-Biased Strategy", figsize=(12, 9))
# rsp.plot(legend=True)

# '''webscrape to get the historical holdings of tracker etf'''

# import requests
# from bs4 import BeautifulSoup
# from datetime import datetime, timedelta
# import json

# # request page
# html = requests.get("https://www.ishares.com/us/products/239726/#tabsAll").content
# soup = BeautifulSoup(html)

# # find available dates
# holdings = soup.find("div", {"id": "holdings"})
# dates_div = holdings.find_all("div", "component-date-list")[1]
# dates_div.find_all("option")
# dates = [option.attrs["value"] for option in dates_div.find_all("option")]

# # download constituents for each date
# constituents = pd.Series()
# for date in dates:
#     resp = requests.get(
#         f"https://www.ishares.com/us/products/239726/ishares-core-sp-500-etf/1467271812596.ajax?tab=all&fileType=json&asOfDate={date}"
#     ).content[3:]
#     tickers = json.loads(resp)
#     tickers = [(arr[0], arr[1]) for arr in tickers['aaData']]
#     date = datetime.strptime(date, "%Y%m%d")
#     constituents[date] = tickers

# constituents = constituents.iloc[::-1] # reverse into cronlogical order
# constituents.head()



# # %%


# %%
