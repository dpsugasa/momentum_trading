# [Python]

#%%

## Load libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import os
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = 'C:/Users/user_name/Anaconda3/Library/plugins/platforms'
plt.style.use('ggplot')

## For saving images to png. 
## AS noted in the previous post reticulate does not seem to handle plt.annotate() so 
## we've taken to saving the graphs as png files and then loading them within Rmarkdown.

import os
DIR = "~/your_directory/" 

def save_fig_blog(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(DIR, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dip=resolution)

## Load asset return and portfolio data
df = pd.read_pickle('port_const.pkl') # see previous posts for how this frame was constructed 
df.iloc[0,3] = 0.006

port1, wts1, sharpe1  = Port_sim.calc_sim_lv(df.iloc[:60, :4], 10000,4) # see previous posts for Port_sim class and methods

## Load economic data
# Quandl datar
import quandl
quandl.ApiConfig.api_key = 'your_key'
start_date = '1970-01-01'
end_date = '2019-12-31'

aaii = quandl.get("AAII/AAII_SENTIMENT", start_date = start_date, end_date = end_date)
aaii_mon = aaii.resample('M').last()
aaii_sen = aaii_mon.loc[:,'Bull-Bear Spread']

con_sent = quandl.get("UMICH/SOC1", start_date=start_date, end_date=end_date)
pmi = quandl.get("ISM/MAN_PMI", start_date=start_date, end_date=end_date)
pmi = pmi.resample('M').last() # PMI is released on first of month. But most of the other data is on the last of the month 

# FRED data
start_date = '1970-01-01'
end_date = '2019-12-31'
indicators = ['UNRATE', 'PERMIT','PCE', 'T10Y2Y', 'DAAA', 'DBAA']
fred_macros = {}

for indicator in indicators:
    fred_macros[indicator] = dr.DataReader(indicator, 'fred', start_date, end_date)
    fred_macros[indicator] = fred_macros[indicator].resample('M').last() 

## Combine data series
from functools import reduce

risk_factors = pd.merge(con_sent, pmi, how = 'left', on = 'Date')

for indicator in indicators:
    risk_factors = pd.merge(risk_factors, fred_macros[indicator], how = 'left', left_index=True, right_index=True) 

# Create corporate yield spread
risk_factors['CORP'] = risk_factors['DBAA'] - risk_factors['DAAA']
risk_factors = risk_factors.drop(['DAAA', 'DBAA'], axis=1)

# Transform factors
macro_risk_chg_1 = risk_factors.copy()
macro_risk_chg_1.iloc[:,[0,1,3,4]] = macro_risk_chg_1.iloc[:,[0,1,3,4]].pct_change()
macro_risk_chg_1.iloc[:, [2,5,6]] = macro_risk_chg_1.iloc[:,[2,5,6]]*.01
macro_risk_chg_1 = macro_risk_chg_1.replace([np.inf, -np.inf], 0.0)

# Graph factors
(macro_risk_chg_1['1987':'1991']*100).plot(figsize=(12,6), cmap="twilight_shifted")
plt.legend(['Sentiment', 'PMI', 'Unemployment', 'Housing', 'PCE', 'Treasuries', 'Corporates'])
plt.ylabel('Percent (%)')
plt.title('Macroeconomic risk variables')
save_fig_blog('macro_risk_factors')
plt.show()

# Run regressions on chosen factors and asset classes
import statsmodels.api as sm
x = macro_risk_chg_1.loc['1987':'1991', ['Index', 'PMI', 'PERMIT', 'PCE']]

X = sm.add_constant(x.values)

rsq = []
for i in range(4):
    y = df.iloc[:60,i].values
    mod = sm.OLS(y, X).fit().rsquared*100
    rsq.append(mod)

asset_names = ['Stocks', 'Bonds', 'Gold', 'Real estate']

# Plot R-squareds
plt.figure(figsize=(12,6))
plt.bar(asset_names, rsq, color='blue')

for i in range(4):
    plt.annotate(str(round(rsq[i]),), xy = (asset_names[i], rsq[i]+0.5))

plt.ylim([0,22])    
plt.ylabel("$R^{2}$")
plt.title("$R^{2}$ for Macro Risk Factor Model")
save_fig_blog('macro_risk_r2')
plt.show()   

# Find factor exposures
assets = df.iloc[:60,:4]
betas = pd.DataFrame(index=assets.columns)
pvalues = pd.DataFrame(index=assets.columns)
error = pd.DataFrame(index=assets.index)

# Calculate factor risk exposures and pvalues
x = macro_risk_chg_1.loc['1987':'1991', ['Index', 'PMI', 'PERMIT', 'PCE']]
X = sm.add_constant(x.values)
factor_names = [j.lower() for j in x.columns.to_list()] 

# Iterate through asset classes
for i in assets.columns:    
    y = assets.loc[:,i].values
    result = sm.OLS(y, X).fit()
        
    for j in range(1, len(result.params)):
        betas.loc[i, factor_names[j-1]] = result.params[j]
        pvalues.loc[i, factor_names[j-1]] = result.pvalues[j]
         
    error.loc[:,i] = (y - X.dot(result.params))
    
# Graph betas    
betas.plot(kind='bar', width = 0.75, color=['darkblue', 'blue', 'grey', 'darkgrey'], figsize=(12,6))
plt.legend(['Sentiment', 'PMI', 'Housing','PCE'])
plt.xticks([0,1,2,3], ['Stock', 'Bond', 'Gold', 'Real estate'], rotation=0)
plt.ylabel(r'Factor $\beta$s')
plt.title(r'Factor $\beta$s by asset class')
save_fig_blog('factor_betas')
plt.show()

# Create function to calculate how much factors explaing portfolio variance
def factor_port_var(betas, factors, weights, error):
    
        B = np.array(betas)
        F = np.array(factors.cov())
        S = np.diag(np.array(error.var()))

        factor_var = weights.dot(B.dot(F).dot(B.T)).dot(weights.T)
        specific_var = weights.dot(S).dot(weights.T)
            
        return factor_var, specific_var

# Graph explained variance on original portfolios        
satis_wt = np.array([0.32, 0.4, 0.2, 0.08])
equal_wt = np.repeat(0.25,4)
max_sharp_wt = wts1[np.argmax(sharpe1)]
max_ret_wt = wts1[pd.DataFrame(np.c_[port1,sharpe1], columns = ['ret', 'risk', 'sharpe']).sort_values(['ret', 'sharpe'], ascending=False).index[0]]

factors = macro_risk_chg_1.loc['1987':'1991', ['Index', 'PMI', 'PERMIT', 'PCE']]

wt_list = [satis_wt, equal_wt, max_sharp_wt, max_ret_wt]
port_exp=[]

for wt in wt_list:
    out = factor_port_var(betas, factors, wt, error)
    port_exp.append(out[0]/(out[0] + out[1]))

port_exp = np.array(port_exp)


port_names = ['Satisfactory', 'Naive', 'Max Sharpe', 'Max Return']
plt.figure(figsize=(12,6))
plt.bar(port_names, port_exp*100, color='blue')

for i in range(4):
    plt.annotate(str(round(port_exp[i]*100)) + '%', xy = (i-0.05, port_exp[i]*100+0.5))
    
plt.title('Original four portfolios variance explained by Macro risk factor model')
plt.ylabel('Variance explained (%)')
plt.ylim([0,20])
save_fig_blog('port_var_exp_21')
plt.show()

## Forward returns

## R-squared function
# A bit long. If we had more time we would have broken it up into more helper functions
def rsq_func(ind_df, dep_df, look_forward = None, period = 60, start_date=0, plot=True, asset_names = True, print_rsq = True, save_fig = False, fig_name = None):
    """ Assumes ind_df starts from the same date as dep_df.
        Dep_df has only as many columns as interested for modeling. """
    
    X = sm.add_constant(ind_df[0:start_date+period].values)
    
    N = len(dep_df.columns)
    
    rsq = []
    
    if look_forward:
        start = start_date + look_forward
        end = start_date + look_forward + period
    else:
        start = start_date 
        end = start_date + period
    
    for i in range(N):
        y = dep_df.iloc[start:end,i].values
        mod = sm.OLS(y, X).fit().rsquared*100
        rsq.append(mod)
        if print_rsq:
            print(f'R-squared for {df.columns[i]} is {mod:0.03f}')
        

    if plot:
        if asset_names:
            x_labels = ['Stocks', 'Bonds', 'Gold', 'Real estate']
        else:
            x_labels = asset_names

        plt.figure(figsize=(12,6))
        plt.bar(x_labels, rsq, color='blue')

        for i in range(4):
            plt.annotate(str(round(rsq[i]),), xy = (x_labels[i], rsq[i]+0.5))

        # plt.ylim([80,100])    
        plt.ylabel("$R^{2}$")
        plt.title("$R^{2}$ for Macro Risk Factor Model")
        if save_fig:
            save_fig_blog(fig_name)
        else:
            plt.tight_layout()
        plt.show()
        
    return rsq

# Graph figure for 10 month forward, save_fig=False if don't want to save!
ind_df_1 = macro_risk_chg_1.loc['1987':, ['Index', 'PMI', 'PERMIT', 'PCE']]
dep_df_1 = df.iloc[:,:-1]

_ = rsq_func(ind_df_1, dep_df_1, look_forward = 10, save_fig = True, fig_name = 'macro_risk_r2_10m')

#Graph figure for 10 month forward, save_fig=False if don't want to save!
_ = rsq_func(ind_df_1, dep_df_1, look_forward = 12, y_lim = [0,15], save_fig = True, fig_name = 'macro_risk_r2_12m')

# Scaled factors

# Grid search for best params
scale_for = pd.DataFrame(np.c_[np.array([np.repeat(x,12) for x in range(3,13)]).flatten(),\
                         np.array([np.arange(1,13)]*10).flatten(),\
                         np.array([np.zeros(120)]*4).T],\
                        columns = ['Month lookback', 'Month Forward', 'Stocks', 'Bonds', 'Gold', 'Real estate'])
count = 0
for i in range(3, 13):
    risk_scale = risk_factors.apply(lambda x: (x - x.rolling(i).mean())/x.rolling(i).std(ddof=1))['1987':]
    risk_scale.replace([np.inf, -np.inf], 0.0, inplace=True)
    for j in range(1,13):
        out = rsq_func(risk_scale, dep_df_1, look_forward = j, plot=False, print_rsq=False)
        scale_for.iloc[count,2:] = np.array(out)
        count+=1
        
## Sort data
scaled_sort = scale_for.sort_values(['Stocks','Bonds'], ascending=False).round(1).reset_index()
