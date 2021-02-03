import pandas as pd                         
from pytrends.request import TrendReq 


pytrend = TrendReq()
pytrend.build_payload(kw_list=['define', 'definition', 'dictionary', 'urban dictionary'], timeframe='2019-12-30 2020-12-29')
related_queries = pytrend.related_queries()

searches = pytrend.trending_searches(pn='united_states', timeframe='2019-12-30 2020-12-29')

pytrend = TrendReq()
pytrend.build_payload(kw_list=['inflation'], timeframe='2010-01-01 2021-01-01')
history = pytrend.interest_over_time()