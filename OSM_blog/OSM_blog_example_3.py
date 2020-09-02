# Built using Python 3.7.4

#%%
# Load libraries
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

#import Port_sim # for class and methods see prior posts
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

plt.style.use('ggplot')

# Load data
#df = pd.read_pickle('port_const.pkl')
#dat = pd.read_pickle('data_port_const.pkl')

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
    

# Create portfolio simulation
np.random.seed(123)
port_sim_1, wts_1, _, sharpe_1, _ = Port_sim.calc_sim(df.iloc[1:60,0:4],1000,4)

# Create efficient frontier function
from scipy.optimize import minimize

def eff_frontier(df_returns, min_ret, max_ret):
    
    n = len(df_returns.columns)
    
    def get_data(weights):
        weights = np.array(weights)
        returns = np.sum(df_returns.mean() * weights)
        risk = np.sqrt(np.dot(weights.T, np.dot(df_returns.cov(), weights)))
        sharpe = returns/risk
        return np.array([returns,risk,sharpe])

    # Contraints
    def check_sum(weights):
        return np.sum(weights) - 1

    # Rante of returns
    mus = np.linspace(min_ret,max_ret,21) 

    # Function to minimize
    def minimize_volatility(weights):
        return  get_data(weights)[1] 

    # Inputs
    init_guess = np.repeat(1/n,n)
    bounds = ((0.0,1.0),) * n

    eff_risk = []
    port_weights = []

    for mu in mus:
        # function for return
        cons = ({'type':'eq','fun': check_sum},
                {'type':'eq','fun': lambda w: get_data(w)[0] - mu})

        result = minimize(minimize_volatility,init_guess,method='SLSQP',bounds=bounds,constraints=cons)

        eff_risk.append(result['fun'])
        port_weights.append(result.x)
    
    eff_risk = np.array(eff_risk)
    
    return mus, eff_risk, port_weights

# Create returns and min/max ranges
df_returns = df.iloc[1:60, 0:4]
min_ret = min(port_sim_1[:,0])
max_ret = max(port_sim_1[:,0])

# Find efficient portfolio
eff_ret, eff_risk, eff_weights = eff_frontier(df_returns, min_ret, max_ret)
eff_sharpe = eff_ret/eff_risk


### Test results of different weighting schemes on simulated returns
## Create weight schemes
satisfice_wts = np.array([0.32, 0.4, 0.08, 0.2]) # Calculated in previous post using port_select_func
simple_wts = np.repeat(0.25, 4)
eff_sharp_wts = eff_weights[np.argmax(eff_sharpe)]
eff_max_wts = eff_weights[np.argmax(eff_ret)]

## Create portfolio metric function to iterate
def port_func(df, wts):
    mean_ret = df.mean()
    returns = np.sum(mean_ret * wts)
    risk = np.sqrt(np.dot(wts, np.dot(df.cov(), wts)))
    return returns, risk
    
# Run portfolio returns for return simulations
from datetime import datetime
start_time = datetime.now()

list_df = [np.zeros((1000,2)) for _ in range(4)]
wt_list = [satisfice_wts, simple_wts, eff_sharp_wts, eff_max_wts]

for i in range(4):
    arr = list_df[i]
    for j in range(1000):
        arr[j] = port_func(sim1[j][0], wt_list[i])
    
    sharpe_calc = arr[:,0]/arr[:,1]
    list_df[i] = np.c_[arr, sharpe_calc]

satis_df = list_df[0]
simple_df = list_df[1]
eff_sharp_df = list_df[2]
eff_max_df = list_df[3]

end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))

# Note python produces this much faster than R. Duration: 0:00:03.226398. Our R code must not be optimized.

# Create portfolio means and names for graphing

port_means = []

for df in list_df:
    port_means.append(df[:][:,0].mean()*1200)
    
port_names = ['Satisfactory','Naive', 'Sharpe', 'Max']

# Create graphing function

def pf_graf(names, values, rnd, nudge, ylabs, graf_title):
    df = pd.DataFrame(zip(names, values), columns = ['key', 'value'])
    sorted = df.sort_values(by = 'value')
    plt.figure(figsize = (12,6))
    plt.bar('key', 'value', data = sorted, color='darkblue')

    for i in range(len(names)):
        plt.annotate(str(round(sorted['value'][i], rnd)), xy = (sorted['key'][i], sorted['value'][i]+nudge))
    
    plt.ylabel(ylabs)
    plt.title('{} performance by portfolio'.format(graf_title))
    plt.show()

# Graph return performance by portfolio
pf_graf(port_names, port_means, 2, 0.5, 'Returns (%)', 'Return')

# Build names for comparison chart
comp_names= []
for i in range(4):
    for j in range(i+1,4):
        comp_names.append('{} vs. {}'.format(port_names[i], port_names[j]))

# Calculate comparison values
comp_values = []

for i in range(4):
    for j in range(i+1, 4):
        comps =np.mean(list_df[i][:][:,0] > list_df[j][:][:,0])
        comp_values.append(comps)
        

# Graph comparisons
pf_graf(comp_names[:-1], comp_values[:-1], 2, 0.025, 'Frequency (%)', 'Frequency of')

# Build Sharpe portfolio comparisons 

sharp_means = []
for df in list_df:
    sharp_means.append(df[:][:,2].mean()*np.sqrt(12))
    
sharp_comp = []
for i in range(4):
    for j in range(i+1, 4):
        comp = np.mean(list_df[i][:][:,2] > list_df[j][:][:,2])
        sharp_comp.append(comp)
        
# Graph mean return comparsions for sharpe porfolio 
pf_graf(port_names, sharp_means, 2, 0.005, "Sharpe ratio", "Sharpe ratio")

# Graph sharpe results for sharpe portoflio
pf_graf(comp_names[:-1], sharp_comp[:-1], 2, 0.005, "Frequency(%)", "Frequency")

# Bring in port simulation to compare results across million portfolios
port_1m = pd.read_pickle("port_3m.pkl")
sharpe_1m = port_1m[:,0]/port_1m[:,1]

# Create mean and sharpe outperformance results lists
sim_mean = []
sim_sharp = []

for i in range(4):
    mean = np.mean(np.mean(list_df[i][:,0]) > port_1m[:,0])
    sim_mean.append(mean)
    sharp = np.mean(np.mean(list_df[i][:,2]) > sharpe_1m[:])
    sim_sharp.append(sharp)
    
# Graph return outperformance
pf_graf(port_names, sim_mean, 2, 0.005, "Frequency(%)", "Frequency")

# Graph sharpe outperformance
pf_graf(port_names, sim_sharp, 2, 0.005, 'Frequency (%)', 'Frequency')
