# Built using Python 3.7.4

# Load libraries
#%%
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('ggplot')
sns.set()

# Load data
start_date = '1970-01-01'
end_date = '2019-12-31'
symbols = ["WILL5000INDFC", "BAMLCC0A0CMTRIV", "GOLDPMGBD228NLBM", "CSUSHPINSA", "DGS5"]
sym_names = ["stock", "bond", "gold", "realt", 'rfr']
filename = 'port_const.pkl'#'data_port_const.pkl'

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
df.to_pickle(filename) # If you haven't saved the file  
    
  
df = data_mon.pct_change()['1971':'2019']
pd.to_pickle(df,filename) # if you haven't saved the file

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

# Plot
np.random.seed(123)
port_sim_1, wts_1, _, sharpe_1, _ = Port_sim.calc_sim(df.iloc[1:60,0:4],1000,4)

Port_sim.graph_sim(port_sim_1, sharpe_1)

## Selection function

# Constraint function
def port_select_func(port, wts, return_min, risk_max):
    port_select = pd.DataFrame(np.concatenate((port, wts), axis=1))
    port_select.columns = ['returns', 'risk', 1, 2, 3, 4]
    
    port_wts = port_select[(port_select['returns']*12 >= return_min) & (port_select['risk']*np.sqrt(12) <= risk_max)]
    port_wts = port_wts.iloc[:,2:6]
    port_wts = port_wts.mean(axis=0)
    
    def graph():
        plt.figure(figsize=(12,6))
        key_names = {1:"Stocks", 2:"Bonds", 3:"Gold", 4:"Real estate"}
        lab_names = []
        graf_wts = port_wts.sort_values()*100
        
        for i in range(len(graf_wts)):
            name = key_names[graf_wts.index[i]]
            lab_names.append(name)

        plt.bar(lab_names, graf_wts)
        plt.ylabel("Weight (%)")
        plt.title("Average weights for risk-return constraint", fontsize=15)
        
        for i in range(len(graf_wts)):
            plt.annotate(str(round(graf_wts.values[i])), xy=(lab_names[i], graf_wts.values[i]+0.5))
    
        
        plt.show()
    
    return port_wts, graph()
    
    
# Graph
results_1_wts,_ = port_select_func(port_sim_1, wts_1, 0.07, 0.1)

# Return function with no rebalancing
def rebal_func(act_ret, weights):
    ret_vec = np.zeros(len(act_ret))
    wt_mat = np.zeros((len(act_ret), len(act_ret.columns)))
    for i in range(len(act_ret)):
        wt_ret = act_ret.iloc[i,:].values*weights
        ret = np.sum(wt_ret)
        ret_vec[i] = ret
        weights = (weights + wt_ret)/(np.sum(weights) + ret)
        wt_mat[i,] = weights
    
    return ret_vec, wt_mat
    
## Run rebalance function using desired weights
port_1_act, wt_mat = rebal_func(df.iloc[61:121,0:4], results_1_wts)
port_act = {'returns': np.mean(port_1_act), 
           'risk': np.std(port_1_act), 
           'sharpe': np.mean(port_1_act)/np.std(port_1_act)*np.sqrt(12)}

# Run simulation on first five-year period
np.random.seed(123)
port_sim_2, wts_2, _, sharpe_2, _ = Port_sim.calc_sim(df.iloc[61:121,0:4],1000,4)

# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_sim_2[:,1]*np.sqrt(12)*100, port_sim_2[:,0]*1200, marker='.', c=sharpe_2, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='red', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

# Calculate returns and risk for longer period
# Call prior file for longer period
df = data_mon.pct_change()['1971':'2019']

hist_mu = df['1971':'1991'].mean(axis=0)
hist_sigma = df['1971':'1991'].std(axis=0)

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
graphs = []
for i in range(4):
    samp = np.random.randint(1,1000)
    port, _, _, sharpe, _ = Port_sim.calc_sim(sim1[samp][0], 1000, 4)
    graf = [port,sharpe]
    graphs.append(graf)

# Graph sample portfolios
fig, axes = plt.subplots(2, 2, figsize=(12,6))

for i, ax in enumerate(fig.axes):
    ax.scatter(graphs[i][0][:,1]*np.sqrt(12)*100, graphs[i][0][:,0]*1200, marker='.', c=graphs[i][1], cmap='Blues')

plt.show()

# Calculate probability of hitting risk-return constraints based on sample portfolos
probs = []
for i in range(4):
    out = round(np.mean((graphs[i][0][:,0] >= 0.07/12) & (graphs[i][0][:,1] <= 0.1/np.sqrt(12))),2)*100
    probs.append(out)
    

# Simulate portfolios from reteurn simulations
def wt_func():
    a = np.random.uniform(0,1,4)
    return a/np.sum(a)
    
# Note this should run relatively quickly: less than a minute. 
np.random.seed(123)
portfolios = np.zeros((1000, 1000, 2))
for i in range(1000):
    wt_mat = np.array([wt_func() for _ in range(1000)])
    port_ret = sim1[i][0].mean(axis=0)
    cov_dat = sim1[i][0].cov()
    returns = np.dot(wt_mat, port_ret)
    risk = [np.sqrt(np.dot(np.dot(wt.T,cov_dat), wt)) for wt in wt_mat]
    portfolios[i][:,0] = returns
    portfolios[i][:,1] = risk

port_1m = portfolios.reshape((1000000,2))

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
plt.title('Ten thousand samples from one million simulated portfolios', fontsize=20)
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
