from xbbg import blp
import json
import configparser
import pandas as pd

tickers= ["ESA Index"]
class MomoPy(object):
    ''' tpqoa is a Python wrapper class for streaming bbg '''

    def __init__(self):
        ''' 
        need to put a docstring
        
        '''
        self.config = configparser.ConfigParser()
        self.suffix = '.000000000Z'
        self.stop_stream = False

        self.bbg = blp



    def stream_data(self, tickers, flds, max_cnt=None, info=None):
        ''' Starts a real-time data stream.

        Parameters
        ==========
        instrument: string
            valid instrument name
        '''
        self.stream_tickers = tickers
        self.ticks = 0
        response = async for px in self.bbg.live(tickers=None, flds=None, info=None, max_cnt=0, **kwargs)
        msgs = []
        for msg_type, msg in response.parts():
            msgs.append(msg)
            print(msg_type, msg)

        return msgs

p = MomoPy.stream_data(tickers=tickers, flds="LAST PRICE")

            if msg_type == 'pricing.ClientPrice':
                self.ticks += 1
                self.time = msg.time
                #self.on_success(msg.time,
                 #               float(msg.bids[0].dict()['price']),
                  #              float(msg.asks[0].dict()['price']))
                if stop is not None:
                    if self.ticks >= stop:
                        if ret:
                            return msgs
                        break
            if self.stop_stream:
                if ret:
                    return msgs
                break

    def on_success(self, time, bid, ask):
        ''' Method called when new data is retrieved. '''
        print(time, bid, ask)

