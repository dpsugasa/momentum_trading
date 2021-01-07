
#%%
# Load libraries
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.style.use('ggplot')

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
# df.to_pickle(filename) # If you haven't saved the file  
  
dat = data_mon.pct_change()['1971':'2019']
# pd.to_pickle(df,filename) # if you haven't saved the file
  
# Portfolio simulation functions

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
        
# Constraint function
def port_select_func(port, wts, return_min, risk_max):
    port_select = pd.DataFrame(np.concatenate((port, wts), axis=1))
    port_select.columns = ['returns', 'risk', 1, 2, 3, 4]
    
    port_wts = port_select[(port_select['returns']*12 >= return_min) & (port_select['risk']*np.sqrt(12) <= risk_max)]
    port_wts = port_wts.iloc[:,2:6]
    port_wts = port_wts.mean(axis=0)
    return port_wts
    
def port_select_graph(port_wts):
    plt.figure(figsize=(12,6))
    key_names = {1:"Stocks", 2:"Bonds", 3:"Gold", 4:"Real estate"}
    lab_names = []
    graf_wts = port_wts.sort_values()*100
        
    for i in range(len(graf_wts)):
        name = key_names[graf_wts.index[i]]
        lab_names.append(name)

    plt.bar(lab_names, graf_wts, color='blue')
    plt.ylabel("Weight (%)")
    plt.title("Average weights for risk-return constraint", fontsize=15)
        
    for i in range(len(graf_wts)):
        plt.annotate(str(round(graf_wts.values[i])), xy=(lab_names[i], graf_wts.values[i]+0.5))
    
        
    plt.show()

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
    
## Rum simulation and graph
np.random.seed(123)
port_sim_1, wts_1, _, sharpe_1, _ = Port_sim.calc_sim(df.iloc[1:60,0:4],1000,4)

Port_sim.graph_sim(port_sim_1, sharpe_1)


# Weight choice
results_1_wts = port_select_func(port_sim_1, wts_1, 0.07, 0.1)
port_select_graph(results_1_wts)

# Compute satisfactory portfolio
satis_ret = np.sum(results_1_wts * df.iloc[1:60,0:4].mean(axis=0).values)
satis_risk = np.sqrt(np.dot(np.dot(results_1_wts.T, df.iloc[1:60,0:4].cov()),results_1_wts))

# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_sim_1[:,1]*np.sqrt(12)*100, port_sim_1[:,0]*1200, marker='.', c=sharpe_1, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(satis_risk*np.sqrt(12)*100, satis_ret*1200, c='red', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()


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
    mus = np.linspace(min_ret,max_ret,20) 

    # Function to minimize
    def minimize_volatility(weights):
        return  get_data(weights)[1] 

    # Inputs
    init_guess = np.repeat(1/n,n)
    bounds = tuple([(0,1) for _ in range(n)])

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
    
## Create variables for froniter function
df_returns = df.iloc[1:60, 0:4]
min_ret = min(port_sim_1[:,0])
max_ret = max(port_sim_1[:,0])

eff_ret, eff_risk, eff_weights = eff_frontier(df_returns, min_ret, max_ret)

## Graph efficient frontier

plt.figure(figsize=(12,6))
plt.scatter(port_sim_1[:,1]*np.sqrt(12)*100, port_sim_1[:,0]*1200, marker='.', c=sharpe_1, cmap='Blues')
plt.plot(eff_risk*np.sqrt(12)*100,eff_ret*1200,'b--',linewidth=2)
plt.scatter(satis_risk*np.sqrt(12)*100, satis_ret*1200, c='red', s=50)

plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

## Graph with unconstrained weights
np.random.seed(123)
port_sim_1lv, wts_1lv, _, sharpe_1lv, _ = Port_sim.calc_sim_lv(df.iloc[1:60,0:4],1000,4)

Port_sim.graph_sim(port_sim_1lv, sharpe_1lv)

# Weight choice
results_1lv_wts = port_select_func(port_sim_1lv, wts_1lv, 0.07, 0.1)
port_select_graph(results_1lv_wts)

# Satisfactory portfolio unconstrained weights
satis_ret1 = np.sum(results_1lv_wts * df.iloc[1:60,0:4].mean(axis=0).values)
satis_risk1 = np.sqrt(np.dot(np.dot(results_1lv_wts.T, df.iloc[1:60,0:4].cov()),results_1lv_wts))

# Graph with efficient frontier
plt.figure(figsize=(12,6))
plt.scatter(port_sim_1lv[:,1]*np.sqrt(12)*100, port_sim_1lv[:,0]*1200, marker='.', c=sharpe_1lv, cmap='Blues')
plt.plot(eff_risk*np.sqrt(12)*100,eff_ret*1200,'b--',linewidth=2)
plt.scatter(satis_risk1*np.sqrt(12)*100, satis_ret1*1200, c='red', s=50)

plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

# Five year forward with unconstrained satisfactory portfolio
# Returns
## Run rebalance function using desired weights
port_1_act, wt_mat = rebal_func(df.iloc[61:121,0:4], results_1lv_wts)
port_act = {'returns': np.mean(port_1_act), 
           'risk': np.std(port_1_act), 
           'sharpe': np.mean(port_1_act)/np.std(port_1_act)*np.sqrt(12)}

# Run simulation on recent five-years
np.random.seed(123)
port_sim_2lv, wts_2lv, _, sharpe_2lv, _ = Port_sim.calc_sim_lv(df.iloc[61:121,0:4],1000,4)


# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_sim_2lv[:,1]*np.sqrt(12)*100, port_sim_2lv[:,0]*1200, marker='.', c=sharpe_2lv, cmap='Blues')
plt.plot(eff_risk*np.sqrt(12)*100,eff_ret*1200,'b--',linewidth=2)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='red', s=50)

plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

# Eficient frontier on long term data
df_returns_l = dat.iloc[1:254, 0:4]
min_ret_l = min(port_sim_1[:,0])
max_ret_l = max(port_sim_1[:,0])

eff_ret_l, eff_risk_l, eff_weightsl = eff_frontier(df_returns_l, min_ret_l, max_ret_l)

## Graph with original
plt.figure(figsize=(12,6))
plt.scatter(port_sim_1lv[:,1]*np.sqrt(12)*100, port_sim_1lv[:,0]*1200, marker='.', c=sharpe_1lv, cmap='Blues')
plt.plot(eff_risk_l*np.sqrt(12)*100,eff_ret_l*1200,'b--',linewidth=2)
plt.scatter(satis_risk1*np.sqrt(12)*100, satis_ret1*1200, c='red', s=50)

plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

## Graph with five-year forward
# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_sim_2lv[:,1]*np.sqrt(12)*100, port_sim_2lv[:,0]*1200, marker='.', c=sharpe_2lv, cmap='Blues')
plt.plot(eff_risk_l*np.sqrt(12)*100,eff_ret_l*1200,'b--',linewidth=2)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='red', s=50)

plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()