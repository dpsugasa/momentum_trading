# Coded in Python 3.7.4

#%%
# Load libraries
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

import seaborn as sns
plt.style.use('ggplot')
sns.set()

# Load data
start_date = '1970-01-01'
end_date = '2019-12-31'
symbols = ["WILL5000INDFC", "BAMLCC0A0CMTRIV", "GOLDPMGBD228NLBM", "CSUSHPINSA", "DGS5"]
sym_names = ["stock", "bond", "gold", "realt", 'rfr']
filename = 'port_const.pkl'

try:
    df = pd.read_pickle(filename)
    print('Data loaded')
except FileNotFoundError:
    print("File not found")
    print("Loading data", 30*"-")
    data = web.DataReader(symbols, 'fred', start_date, end_date)
    data.columns = sym_names
    #df = data.copy()
    
data_mon = data.resample('M').last()
df = data_mon.pct_change()['1987':'2019']
df.to_pickle(filename)

# Exploratory data analysis
sns.pairplot(df.iloc[1:61,0:4])
plt.show()

# Create function
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
    
    def graph_sim(port,sharpe):
        plt.figure(figsize=(14,6))
        plt.scatter(port[:,1]*np.sqrt(12)*100, port[:,0]*1200, marker='.',c=sharpe, cmap='Blues')
        plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
        plt.title('Simulated porfolios', fontsize=20)
        plt.xlabel('Risk (%)')
        plt.ylabel('Return (%)')
        plt.show() 
        
# Create simulation
np.random.seed(123)
port, wts, _, sharpe, _ = Port_sim.calc_sim(df.iloc[1:60,0:4],1000,4)

# Graph simulation
Port_sim.graph_sim(port, sharpe)

#### Portfolio constraint function
def port_select_func(port, wts, return_min, risk_max):
    port_select = pd.DataFrame(np.concatenate((port, wts), axis=1))
    port_select.columns = ['returns', 'risk', 1, 2, 3, 4]
    
    port_wts = port_select[(port_select['returns']*12 >= return_min) & (port_select['risk']*np.sqrt(12) <= risk_max)]
    port_wts = port_wts.iloc[:,2:6]
#     port_wts.columns = ["Stocks", "Bonds", "Gold", "Real estate"]
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
    
## Run function
results_wts, results_graph = port_select_func(port, wts, 0.05, 0.14)

## Create rebalancing function
def rebal_func(act_ret, weights):
    tot_ret = 1
    ret_vec = np.zeros(60)
    for i in range(60):
        wt_ret = act_ret.iloc[i,:].values*weights
        ret = np.sum(wt_ret)
        tot_ret = tot_ret * (1+ret)
        ret_vec[i] = ret
        weights = (weights + wt_ret)/(np.sum(weights) + ret)
    
    return ret_vec

# Run rebalancing function and dictionary
ret_vec = rebal_func(df.iloc[61:121,0:4], results_wts)
port_act = {'returns': np.mean(ret_vec), 
           'risk': np.std(ret_vec), 
           'sharpe': np.mean(ret_vec)/np.std(ret_vec)*np.sqrt(12)}

#### Run simulation on next group
# Run simulation on recent five-years
np.random.seed(123)
port_2, wts_2, _, sharpe_2, _ = Port_sim.calc_sim(df.iloc[61:121,0:4],1000,4)

# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_2[:,1]*np.sqrt(12)*100, port_2[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='blue', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

#### Run selection function n next five years implied weight
results_wts_2, results_graph_2 = port_select_func(port_2, wts_2, 0.07, 0.1)

# Run simulation on recent five-years
np.random.seed(123)
port_2l, wts_2l, _, _, _ = Port_sim.calc_sim(df.iloc[1:121,0:4],1000,4)

# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_2l[:,1]*np.sqrt(12)*100, port_2l[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='red', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()

# Run function on next five years implied weight
results_wts_2l, _ = port_select_func(port_2l, wts_2l, 0.06, 0.12)

# Run simulation on next five years with two different weightings
np.random.seed(123)
ret_old_wt = rebal_func(df.iloc[121:181, 0:4], fut_wt)
ret_new_wt = rebal_func(df.iloc[121:181, 0:4], results_wts_2l)

port_act_1_old = {'returns' : np.mean(ret_old_wt),
                  'risk' : np.std(ret_old_wt),
                  'sharpe' : np.mean(ret_old_wt)/np.std(ret_old_wt)*np.sqrt(12)}

port_act_1_new = {'returns' : np.mean(ret_new_wt),
                  'risk' : np.std(ret_new_wt),
                  'sharpe' : np.mean(ret_new_wt)/np.std(ret_new_wt)*np.sqrt(12)}

port_3, wts_3, _, _, _ = Port_sim.calc_sim(df.iloc[121:181, 0:4], 1000, 4)

plt.figure(figsize=(14,6))
plt.scatter(port_3[:,1]*np.sqrt(12)*100, port_3[:,0]*1200, marker='.', c=sharpe, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.5)
plt.scatter(port_act_1_old['risk']*np.sqrt(12)*100, port_act_1_old['returns']*1200, c='red', s=50)
plt.scatter(port_act_1_new['risk']*np.sqrt(12)*100, port_act_1_new['returns']*1200, c='purple', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()
