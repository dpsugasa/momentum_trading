import tpqoa

oanda = tpqoa.tpqoa('oanda.cfg')

data = oanda.get_history(  
    instrument='EUR_USD',  
    start='2020-11-11',  
    end='2020-11-12',  
    granularity='M1',  
    price='M' 
) 
data.info()

import numpy as np
data['r'] = np.log(data['c'] / data['c'].shift(1))
cols = []
for momentum in [15, 30, 60, 120, 150]:
    col = f'p_{momentum}'
    data[col] = np.sign(data['r'].rolling(momentum).mean())
    cols.append(col)

from pylab import plt
plt.style.use('seaborn')
strats = ['r']
for col in cols:
    strat = f's_{col[2:]}'
    data[strat] = data[col].shift(1) * data['r']
    strats.append(strat)
data[strats].dropna().cumsum().apply(np.exp).plot()

import pandas as pd
class MomentumTrader(tpqoa.tpqoa):
    def __init__(self, config_file, momentum):
        super(MomentumTrader, self).__init__(config_file)  
        self.momentum = momentum  
        self.min_length = momentum + 1  
        self.position = 0  
        self.units = 10000  
        self.tick_data = pd.DataFrame()
    def on_success(self, time, bid, ask):
        trade = False  
        # print(self.ticks, end=' ')  
        self.tick_data = self.tick_data.append(
            pd.DataFrame({'b': bid, 'a': ask, 'm': (ask + bid) / 2},
               index=[pd.Timestamp(time).tz_localize(tz=None)])
        )
        self.data = self.tick_data.resample('5s', 
           label='right').last().ffill()  
        self.data['r'] = np.log(self.data['m'] / 
          self.data['m'].shift(1))  
        self.data['m'] = self.data['r'].rolling(self.momentum).mean()
        self.data.dropna(inplace=True)  
        if len(self.data) > self.min_length:
            self.min_length += 1
            if self.data['m'].iloc[-2] > 0 and self.position in [0, -1]:
                o = oanda.create_order(self.stream_instrument,
                             units=(1 - self.position) * self.units,
                             suppress=True, ret=True)
                print('\n*** GOING LONG ***')
                oanda.print_transactions(tid=int(o['id']) - 1)
                self.position = 1
        if self.data['m'].iloc[-2] < 0 and self.position in [0, 1]:
              o = oanda.create_order(self.stream_instrument,
                            units=-(1 + self.position) * self.units,
                            suppress=True, ret=True)
              print('\n*** GOING SHORT ***')
              self.print_transactions(tid=int(o['id']) - 1)
              self.position = -1

mt = MomentumTrader('oanda.cfg', momentum=5)
mt.stream_data('EUR_USD', stop=3)

class myOanda(tpqoa.tpqoa):
    def on_success(self, time, bid, ask):
        ''' Method called when new data is retrieved. '''
        print('BID: {:.5f} | ASK: {:.5f}'.format(bid, ask))

my_oanda = myOanda('oanda.cfg')

my_oanda.stream_data('EUR_USD', stop=5)

ins = oanda.get_instruments()

