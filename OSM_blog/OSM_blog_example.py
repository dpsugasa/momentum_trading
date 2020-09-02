# Built using Python 3.7.4
#%%

# Load libraries
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('ggplot')
sns.set()

# SKIP IF ALREADY HAVE DATA
# Load data
start_date = '1970-01-01'
end_date = '2019-12-31'
symbols = ["WILL5000INDFC", "BAMLCC0A0CMTRIV", "GOLDPMGBD228NLBM", "CSUSHPINSA", "DGS5"]
sym_names = ["stock", "bond", "gold", "realt", 'rfr']
filename = 'data_port_const.pkl'

try:
    df = pd.read_pickle(filename)
    print('Data loaded')
except FileNotFoundError:
    print("File not found")
    print("Loading data", 30*"-")
    data = web.DataReader(symbols, 'fred', start_date, end_date)
    data.columns = sym_names
    
data_mon = data.resample('M').last()
df = data_mon.pct_change()['1987':'2019']
dat = data_mon.pct_change()['1971':'2019']

## Simulation function
class Port_sim:
    def calc_sim(df, sims, cols):
        wts = np.zeros((sims, cols))

        for i in range(sims):
            a = np.random.uniform(0,1,cols)
            b = a/np.sum(a)
            wts[i,] = b

        mean_ret = df.mean()
        port_cov = df.cov()
    
        port = np.zeros((sims, 2))
        for i in range(sims):
            port[i,0] =  np.sum(wts[i,]*mean_ret)
            port[i,1] = np.sqrt(np.dot(np.dot(wts[i,].T,port_cov), wts[i,]))

        sharpe = port[:,0]/port[:,1]*np.sqrt(12)
        best_port = port[np.where(sharpe == max(sharpe))]
        max_sharpe = max(sharpe)
    
        return port, wts, best_port, sharpe, max_sharpe
    
    def calc_sim_lv(df, sims, cols):
        wts = np.zeros(((cols-1)*sims, cols))
        count=0

        for i in range(1,cols):
            for j in range(sims):
                a = np.random.uniform(0,1,(cols-i+1))
                b = a/np.sum(a)
                c = np.random.choice(np.concatenate((b, np.zeros(i))),cols, replace=False)
                wts[count,] = c
                count+=1

        mean_ret = df.mean()
        port_cov = df.cov()

        port = np.zeros(((cols-1)*sims, 2))
        for i in range(sims):
            port[i,0] =  np.sum(wts[i,]*mean_ret)
            port[i,1] = np.sqrt(np.dot(np.dot(wts[i,].T,port_cov), wts[i,]))

        sharpe = port[:,0]/port[:,1]*np.sqrt(12)
        best_port = port[np.where(sharpe == max(sharpe))]
        max_sharpe = max(sharpe)

        return port, wts, best_port, sharpe, max_sharpe 
    
    def graph_sim(port, sharpe):
        plt.figure(figsize=(14,6))
        plt.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
        plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
        plt.title('Simulated portfolios', fontsize=20)
        plt.xlabel('Risk (%)')
        plt.ylabel('Return (%)')
        plt.show()
        
# Calculate returns and risk for longer period
hist_mu = dat['1971':'1991'].mean(axis=0)
hist_sigma = dat['1971':'1991'].std(axis=0)
        

# Run simulation based on historical figures
np.random.seed(123)
sim1 = []

for i in range(1000):
    #np.random.normal(mu, sigma, obs)
    a = np.random.normal(hist_mu[0], hist_sigma[0], 60) + np.random.normal(0, hist_sigma[0], 60)
    b = np.random.normal(hist_mu[1], hist_sigma[1], 60) + np.random.normal(0, hist_sigma[1], 60)
    c = np.random.normal(hist_mu[2], hist_sigma[2], 60) + np.random.normal(0, hist_sigma[2], 60)
    d = np.random.normal(hist_mu[3], hist_sigma[3], 60) + np.random.normal(0, hist_sigma[3], 60)
    
    df1 = pd.DataFrame(np.array([a, b, c, d]).T)
    
    cov_df1 = df1.cov()
    
    sim1.append([df1, cov_df1])
    
# create graph objects
np.random.seed(123)
samp = np.random.randint(1, 1000, 4)
graphs1 = []
for i in range(4):
    port, _, _, sharpe, _ = Port_sim.calc_sim(sim1[samp[i]][0], 1000, 4)
    graf = [port,sharpe]
    graphs1.append(graf)

# Graph sample portfolios
fig, axes = plt.subplots(2, 2, figsize=(12,6))

for i, ax in enumerate(fig.axes):
    ax.scatter(graphs1[i][0][:,1]*np.sqrt(12)*100, graphs1[i][0][:,0]*1200, marker='.', c=graphs1[i][1], cmap='Blues')

plt.show()

# create graph objects
np.random.seed(123)
graphs2 = []
for i in range(4):
    port, _, _, sharpe, _ = Port_sim.calc_sim_lv(sim1[samp[i]][0], 1000, 4)
    graf = [port,sharpe]
    graphs2.append(graf)

# Graph sample portfolios
fig, axes = plt.subplots(2, 2, figsize=(12,6))

for i, ax in enumerate(fig.axes):
    ax.scatter(graphs2[i][0][:,1]*np.sqrt(12)*100, graphs2[i][0][:,0]*1200, marker='.', c=graphs2[i][1], cmap='Blues')

plt.show()

# Calculate probability of hitting risk-return constraints based on sample portfolos
probs = []
for i in range(8):
    if i <= 3:
        out = round(np.mean((graphs1[i][0][:,0] >= 0.07/12) & (graphs1[i][0][:,1] <= 0.1/np.sqrt(12))),2)*100
        probs.append(out)
    else:
        out = round(np.mean((graphs2[i-4][0][:,0] >= 0.07/12) & (graphs2[i-4][0][:,1] <= 0.1/np.sqrt(12))),2)*100
        probs.append(out)

print(probs)

# Simulate portfolios from reteurn simulations
def wt_func(sims, cols):
    wts = np.zeros(((cols-1)*sims, cols))
    count=0
    
    for i in range(1,cols):
        for j in range(sims):
            a = np.random.uniform(0,1,(cols-i+1))
            b = a/np.sum(a)
            c = np.random.choice(np.concatenate((b, np.zeros(i))),cols, replace=False)
            wts[count,] = c
            count+=1
    
    return wts
    
# Note this takes over 4min to run, substantially worse than the R version, which runs in under a minute. Not sure what I'm missng.
    
np.random.seed(123)
portfolios = np.zeros((1000, 3000, 2))
weights = np.zeros((1000,3000,4))
for i in range(1000):
    wt_mat = wt_func(1000,4)
    port_ret = sim1[i][0].mean(axis=0)
    cov_dat = sim1[i][0].cov()
    returns = np.dot(wt_mat, port_ret)
    risk = [np.sqrt(np.dot(np.dot(wt.T,cov_dat), wt)) for wt in wt_mat]
    portfolios[i][:,0] = returns
    portfolios[i][:,1] = risk
    weights[i][:,:] = wt_mat
    
port_1m = portfolios.reshape((3000000,2))
wt_1m = weights.reshape((3000000,4))

# Find probability of hitting risk-return constraints on simulated portfolios
port_1m_prob = round(np.mean((port_1m[:][:,0] > 0.07/12) & (port_1m[:][:,1] <= 0.1/np.sqrt(12))),2)*100
print(f"The probability of meeting our portfolio constraints is:{port_1m_prob: 0.0f}%")

# Plot sample portfolios
np.random.seed(123)
port_samp = port_1m[np.random.choice(1000000, 10000),:]
sharpe = port_samp[:,0]/port_samp[:,1]

plt.figure(figsize=(14,6))
plt.scatter(port_samp[:,1]*np.sqrt(12)*100, port_samp[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Ten thousand samples from three million simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

# Graph histograms
fig, axes = plt.subplots(1,2, figsize = (12,6))

for idx,ax in enumerate(fig.axes):
    if idx == 1:
        ax.hist(port_1m[:][:,1], bins = 100)
    else:
        ax.hist(port_1m[:][:,0], bins = 100)

plt.show()

## Create buckets for analysis and graphin
df_port = pd.DataFrame(port_1m, columns = ['returns', 'risk'])
port_bins = np.arange(-35,65,10)
df_port['dig_ret'] = pd.cut(df_port['returns']*1200, port_bins)

xs = ["(-35, -25]", "(-25, -15]", "(-15, -5]","(-5, 5]", "(5, 15]", "(15, 25]", "(25, 35]", "(35, 45]", "(45, 55]"]
ys = df_port.groupby('dig_ret').size().values/len(df_port)*100

# Graph buckets with frequency
fig,ax = plt.subplots(figsize = (12,6))
ax.bar(xs[2:7], ys[2:7])
ax.set(xlabel = "Return bucket (%)",
       ylabel = "Frequency (%)",
       title = "Frequency of occurrrence for return bucket ")
plt.show()

# Calculate frequency of occurence for mid range of returns
good_range = np.sum(df_port.groupby('dig_ret').size()[4:6])/len(df_port)
good_range

## Graph buckets with median return and risk
med_ret = df_port.groupby('dig_ret').agg({'returns':'median'})*1200
med_risk = df_port.groupby('dig_ret').agg({'risk':'median'})*np.sqrt(12)*100

labs_ret = np.round(med_ret['returns'].to_list()[2:7])
labs_risk = np.round(med_risk['risk'].to_list()[2:7])

fig, ax = plt.subplots(figsize = (12,6))
ax.bar(xs[2:7], ys[2:7])

for i in range(len(xs[2:7])):
    ax.annotate(str('Returns: ' + str(labs_ret[i])), xy = (xs[2:7][i], ys[2:7][i]+2), xycoords = 'data')
    ax.annotate(str('Risk: ' + str(labs_risk[i])), xy = (xs[2:7][i], ys[2:7][i]+5), xycoords = 'data')
    
ax.set(xlabel = "Return bucket (%)",
       ylabel = "Frequency (%)",
       title = "Frequency of occurrrence for return bucket ",
      ylim = (0,60))


plt.show()

# Find frequency of high return buckets
hi_range = np.sum(df_port.groupby('dig_ret').size()[6:])/len(df_port)
hi_range

## Identify weights for different buckets for graphing

wt_1m = pd.DataFrame(wt_1m, columns = ['Stocks', 'Bonds', 'Gold', 'Real estate'])

port_ids_mid = df_port.loc[(df_port['returns'] >= 0.05/12) & (df_port['returns'] <= 0.25/12)].index
mid_ports = wt_1m.loc[port_ids_mid,:].mean(axis=0)

port_ids_hi = df_port.loc[(df_port['returns'] >= 0.35/12)].index
hi_ports = wt_1m.loc[port_ids_hi,:].mean(axis=0)

port_ids_lo = df_port.loc[(df_port['returns'] <= -0.05/12)].index
lo_ports = wt_1m.loc[port_ids_lo,:].mean(axis=0)

# Sharpe portfolios
df_port['sharpe'] = df_port['returns']/df_port['risk']*np.sqrt(12)
port_ids_sharpe = df_port[(df_port['sharpe'] > 0.7)].index
sharpe_ports = wt_1m.loc[port_ids_sharpe,:].mean(axis=0)

# Create graph function
def wt_graph(ports, title):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(ports.index.values, ports*100)
    for i in range(len(mid_ports)):
        ax.annotate(str(np.round(ports[i],2)*100), xy=(ports.index.values[i], ports[i]*100+2), xycoords = 'data')
    
    ax.set(xlabel = '', ylabel = 'Weigths (%)', title = title, ylim = (0,max(ports)*100+5))
    plt.show()

# Graph weights
wt_graph(mid_ports, "Average asset weights for mid-range portfolios")
wt_graph(mid_ports, "Average asset weights for high return portfolios")
wt_graph(mid_ports, "Average asset weights for negative return portfolios")
wt_graph(mid_ports, "Average asset weights for Sharpe portfolios")
