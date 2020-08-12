#%%
# Import libraries
import math
import time
import numpy as np
import pandas as pd
import datetime as dt

# import cufflinks as cf
# from pylab import plt
import matplotlib.pyplot as plt

# pd.options.plotting.backend = 'plotly'

np.random.seed(1)  # For reproducibility
plt.style.use("seaborn")  # I think this looks pretty
#%matplotlib inline # To get out plots
# Coin flip variable set up
p = 0.55  # Fixes the probability for heads.
f = p - (1 - p)  # Calculates the optimal fraction according to the Kelly criterion.
f

# Preparing our simulation of coin flips with variables
I = 50  # The number of series to be simulated.
n = 100  # The number of trials per series.


def run_simulation(f):
    c = np.zeros(
        (n, I)
    )  # Instantiates an ndarray object to store the simulation results.
    c[0] = 100  # Initializes the starting capital with 100.
    for i in range(I):  # Outer loop for the series simulations.
        for t in range(1, n):  # Inner loop for the series itself.
            o = np.random.binomial(1, p)  # Simulates the tossing of a coin.
            if o > 0:  # If 1, i.e., heads …
                c[t, i] = (1 + f) * c[t - 1, i]  # … then add the win to the capital.
            else:  # If 0, i.e., tails …
                c[t, i] = (1 - f) * c[
                    t - 1, i
                ]  # … then subtract the loss from the capital.
    return c


c_1 = run_simulation(f)  # Runs the simulation.
c_1.round(2)  # Looking at a simulation

plt.figure(figsize=(10, 6))
plt.plot(c_1, "b", lw=0.5)  # Plots all 50 series.
plt.plot(c_1.mean(axis=1), "r", lw=2.5)
# Plots the average over all 50 series.
plt.title("50 Simulations of Rigged Coin Flips")
plt.xlabel("Number of trials")
plt.ylabel("$ Amount Won/Lost")

c_2 = run_simulation(0.05)  # Simulation with f = 0.05.
c_3 = run_simulation(0.25)  # Simulation with f = 0.25.
c_4 = run_simulation(0.5)  # Simulation with f = 0.5.
plt.figure(figsize=(10, 6))
plt.plot(c_1.mean(axis=1), "r", label="$f^*=0.1$")
plt.plot(c_2.mean(axis=1), "b", label="$f=0.05$")
plt.plot(c_3.mean(axis=1), "y", label="$f=0.25$")
plt.plot(c_4.mean(axis=1), "m", label="$f=0.5$")
plt.legend(loc=0)
plt.title("Varied KC Simulations of Rigged Coin Flips")
plt.xlabel("Number of trials")
plt.ylabel("$ Amount Won/Lost")

# Loading SPY data
data = pd.read_csv("SPY Historical Data.csv", index_col=0, parse_dates=True)
# Light Feature Engineering on Returns
data["Change %"] = data["Change %"].map(lambda x: x.rstrip("%")).astype(float) / 100
data.dropna(inplace=True)
data.tail()

mu = data["Change %"].mean() * 252  # Calculates the annualized return.
sigma = data["Change %"].std() * 252 ** 0.5  # Calculates the annualized volatility.
r = 0.0179  # 1 year treasury rate
f = (
    mu - r
) / sigma ** 2  # Calculates the optimal Kelly fraction to be invested in the strategy.
f

equs = []  # preallocating space for our simulations

wtf = "is this"


def kelly_strategy(f):
    global equs
    equ = "equity_{:.2f}".format(f)
    equs.append(equ)
    cap = "capital_{:.2f}".format(f)
    data[equ] = 1  # Generates a new column for equity and sets the initial value to 1.
    data[cap] = (
        data[equ] * f
    )  # Generates a new column for capital and sets the initial value to 1·f∗.
    for i, t in enumerate(data.index[1:]):
        t_1 = data.index[
            i
        ]  # Picks the right DatetimeIndex value for the previous values.
        data.loc[t, cap] = data[cap].loc[t_1] * math.exp(data["Change %"].loc[t])
        data.loc[t, equ] = data[cap].loc[t] - data[cap].loc[t_1] + data[equ].loc[t_1]
        data.loc[t, cap] = data[equ].loc[t] * f


kelly_strategy(f * 0.5)  # Values for 1/2 KC
kelly_strategy(f * 0.66)  # Values for 2/3 KC
kelly_strategy(f)  # Values for optimal KC
ax = data["Change %"].cumsum().apply(np.exp).plot(legend=True, figsize=(10, 6))
data[equs].plot(legend=True)
plt.title("Varied KC Values on SPY, Starting from $1")
plt.xlabel("Years")
plt.ylabel("$ Return")

