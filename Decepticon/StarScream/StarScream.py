from contextlib import redirect_stderr
import pandas as pd
from td.client import TDClient
from td.utils import TDUtilities

# We are going to be doing some timestamp conversions.
milliseconds_since_epoch = TDUtilities().milliseconds_since_epoch

from datetime import datetime, time, timezone

from typing import List, Dict, Union

class StarScream():
    def __init__(self, client_id: str, redirect_uri: str, credentials_path: str =None, trading_account: str = None) _> None:
     
        self.trading_account: str = trading_account
        self.client_id : str = client_id
        self.redirect_uri : str = redirect_uri
        self.credentials_path: str = credentials_path
        self.session: TDClient = self.create_session()
        self.trades: dict = {}
        self. historical_prices: dict = {}
        self.stock_frame = None

    def _create_session(self) -> TDClient:

        td_client = TDClient(client_id = self.client_id,
            redirect_uri = self.redirect_uri, 
            credentials_path=self.credentials_path
        )

        #login to the session
        td_client.login()

        return td_client
        
    
