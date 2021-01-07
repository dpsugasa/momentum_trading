
#load libraries
#%%
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
plt.style.use('ggplot')
sns.set()

## Load data
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

# Run simulation on recent five-years
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

## Show performance attribution

calc = df.iloc[61:121,0:4]*np.concatenate((np.array([results_1_wts]).reshape(1,4),wt_mat))[0:60]
calc = calc.apply(lambda x: (np.prod(x+1)**(1/5)-1)*100)
contribution = round(calc/sum(calc),1)*100

plt.figure(figsize=(12,6))
key_names = ["Stocks", "Bonds", "Gold", "Real estate"]
graf_hts = calc.values

plt.bar(key_names, graf_hts)
plt.ylabel("Return (%)")
plt.ylim(0,7)
plt.title("Compound annual asset return weighted by allocation", fontsize=15)

for i in range(len(graf_hts)):
    plt.annotate("Return: " + str(round(graf_hts[i]))+"%", xy=(i-0.2, graf_hts[i]+0.35))
    plt.annotate("Contributions: " + str(contribution[i])+"%", xy=(i-0.2, graf_hts[i]+0.15))


plt.show()

## Show volatility contribution

port_1_vol = np.sqrt(np.dot(np.dot(results_1_wts.T, df.iloc[61:121,0:4].cov()), results_1_wts))
vol_cont = np.dot(results_1_wts.T, df.iloc[61:121,0:4].cov()/port_1_vol) * results_1_wts
contribution = round(vol_cont/sum(vol_cont),1).values*100

plt.figure(figsize=(12,6))
key_names = ["Stocks", "Bonds", "Gold", "Real estate"]
graf_hts = vol_cont.values * np.sqrt(12) * 100

plt.bar(key_names, graf_hts)
plt.ylabel("Risk (%)")
plt.ylim(0,4)
plt.title("Asset risk weighted by allocation", fontsize=15)

for i in range(len(graf_hts)):
    plt.annotate("Risk: " + str(round(graf_hts[i]))+"%", xy=(i-0.2, graf_hts[i]+0.35))
    plt.annotate("Contribution: " + str(contribution[i])+"%", xy=(i-0.2, graf_hts[i]+0.15))

plt.show()

## Show beginning and ending portfolio weights

ey_names = ["Stocks", "Bonds", "Gold", "Real estate"]
beg = results_1_wts.values*100
end = wt_mat[-1]*100
ind = np.arange(len(beg))
width = 0.4

fig,ax = plt.subplots(figsize=(12,6))
rects1 = ax.bar(ind - width/2, beg, width, label = "Beg", color="slategrey")
rects2 = ax.bar(ind + width/2, end, width, label = "End", color="darkblue")

for i in range(len(beg)):
    ax.annotate(str(round(beg[i])), xy=(ind[i] - width/2, beg[i]))
    ax.annotate(str(round(end[i])), xy=(ind[i] + width/2, end[i]))


ax.set_ylabel("Weight (%)")
ax.set_title("Asset weights at beginning and end of period\n", fontsize=16)
ax.set_xticks(ind)
ax.set_xticklabels(key_names)
ax.legend(loc='upper center', ncol=2)

plt.show()

## Run simulation on second five year period
np.random.seed(123)
port_sim_3, wts_3, _, sharpe_3, _ = Port_sim.calc_sim(df.iloc[121:181,0:4],1000,4)

ret_old_wt, _ = rebal_func(df.iloc[121:181, 0:4], results_1_wts)
ret_old = {'returns': np.mean(ret_old_wt), 
           'risk': np.std(ret_old_wt), 
           'sharpe': np.mean(ret_old_wt)/np.std(ret_old_wt)*np.sqrt(12)}

ret_same_wt, _ = rebal_func(df.iloc[121:181, 0:4], wt_mat[-1])
ret_same = {'returns': np.mean(ret_same_wt), 
           'risk': np.std(ret_same_wt), 
           'sharpe': np.mean(ret_same_wt)/np.std(ret_same_wt)*np.sqrt(12)}


# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(port_sim_3[:,1]*np.sqrt(12)*100, port_sim_3[:,0]*1200, marker='.', c=sharpe_3, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(ret_old['risk']*np.sqrt(12)*100, ret_old['returns']*1200, c='red', s=50)
plt.scatter(ret_same['risk']*np.sqrt(12)*100, ret_same['returns']*1200, c='purple', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()


np.random.seed(123)
test_port,_ ,_ ,sharpe_test, _ = Port_sim.calc_sim_lv(df.iloc[61:121, 0:4], 1000, 4)
# test_port,_ ,_ ,sharpe_test, _ = calc_sim_lv(df.iloc[61:121, 0:4], 1000, 4)

# port_sim_2, wts_2, _, sharpe_2, _ = Port_sim.calc_sim(df.iloc[61:121,0:4],1000,4)0

# Graph simulation with actual portfolio return
plt.figure(figsize=(14,6))
plt.scatter(test_port[:,1]*np.sqrt(12)*100, test_port[:,0]*1200, marker='.', c=sharpe_test, cmap='Blues')
plt.colorbar(label='Sharpe ratio', orientation = 'vertical', shrink = 0.25)
plt.scatter(port_act['risk']*np.sqrt(12)*100, port_act['returns']*1200, c='red', s=50)
plt.title('Simulated portfolios', fontsize=20)
plt.xlabel('Risk (%)')
plt.ylabel('Return (%)')
plt.show()
