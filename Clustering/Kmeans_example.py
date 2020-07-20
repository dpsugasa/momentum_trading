
import pandas as pd
from sklearn.cluster import KMeans
from math import sqrt
import  pylab as pl
import numpy as np
import pandas as pd
from tia.bbg import LocalTerminal
import numpy as np
from datetime import datetime
from operator import itemgetter
from scipy import stats
from scipy.stats import norm
from IPython.display import IFrame
import statsmodels.api as sm
from statsmodels import regression
import blp
import plotly
import chart_studio.plotly as py  # for plotting
import plotly.offline as offline
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.io as pio

pio.templates.default = "none"
from pathlib import Path


IDs = "LP01TREU Index"  # Bloomberg Barclays BBB index
fields = "INDX_MEMBERS"
# mask = ['AV130783', 'AV131417', 'EI809518']

blp = blp.BLPInterface()
df_universe = blp.bulkRequest(IDs, fields)
# df_universe = df_universe[~df_universe.index.isin(mask)]
ticker_list = df_universe.index.values
tl2 = []
for i in ticker_list:
    tl2.append(i + " Corp")

# ets = pd.read_csv(r"D:\OneDrive - Northlight Group\Marketing\Marketing Generic\extra_tickers.csv")

# extra_ticks = ets['Ticker'].values

# tl2.append('EK3988418 Corp')
q = set(tl2)

# set dates, securities, and fields
start_date = "01/04/2017"
end_date = "{:%m/%d/%Y}".format(datetime.now())

cfields = ["LAST PRICE"]

window = 90

df = LocalTerminal.get_historical(
    tl2, cfields, start_date, end_date, period="DAILY"
).as_frame()
df.columns = df.columns.droplevel(-1)
#df = df.pct_change()
#df = df.std(axis=1)
#df = df.rolling(window=window).mean()
#df = df.dropna()

month = df.last_valid_index().month
month_full = df.last_valid_index().strftime("%B")
day = df.last_valid_index().day
year = df.last_valid_index().year

output_dir = Path(
    fr"D:\OneDrive - Northlight Group\Images\Dispersion\{year}\{month_full}"
)
output_dir.mkdir(parents=True, exist_ok=True)

returns = df.pct_change().mean()*252
variance = df.pct_change().std()*sqrt(252)
returns.columns = ["Returns"]
variance.columns = ["Variance"]
#Concatenating the returns and variances into a single data-frame
ret_var = pd.concat([returns, variance], axis = 1).dropna()
ret_var.columns = ["Returns","Variance"]

X =  ret_var.values #Converting ret_var into nummpy array
sse = []
for k in range(2,15):
    
    kmeans = KMeans(n_clusters = k)
    kmeans.fit(X)
    
    sse.append(kmeans.inertia_) #SSE for each n_clusters
pl.plot(range(2,15), sse)
pl.title("Elbow Curve")
pl.show()

kmeans = KMeans(n_clusters = 6).fit(X)
centroids = kmeans.cluster_centers_
pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
pl.show()

print(returns.idxmax())
ret_var.drop("BJ419101 Corp", inplace =True)

X = ret_var.values
kmeans =KMeans(n_clusters = 6).fit(X)
centroids = kmeans.cluster_centers_
pl.scatter(X[:,0],X[:,1], c = kmeans.labels_, cmap ="rainbow")
pl.show()

Company = pd.DataFrame(ret_var.index)
cluster_labels = pd.DataFrame(kmeans.labels_)
df = pd.concat([Company, cluster_labels],axis = 1)