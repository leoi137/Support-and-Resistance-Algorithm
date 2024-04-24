from tqdm import tqdm
import yfinance as yf
import pandas as pd

def get_yahoo_multiple_historical_stock_data(symbol_list, period, interval = '1d', only_close = False):
    """
    Retrieve historical stock data from Yahoo Finance for multiple symbols.

    Parameters:
        - symbol_list: List of stock symbols (e.g., ['TSLA', 'SPY', ..., 'GME'])
        - period: Historical data period (e.g., '1d', '5mo', 'max')
        - interval: Time interval for data (default: '1d'; options: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        - only_close: Flag to indicate if only closing prices are needed (default: False)
        
    Returns:
        DataFrame: Concatenated historical data for all symbols
    """
    close_prices = {} # Dictionary to store close prices for each symbol
    for symbol in symbol_list:
        my_symbol = yf.Ticker(symbol) # Get Ticker object for the symbol
        hist = my_symbol.history(period=period, interval=interval) # Retrieve historical data
        if only_close:
            close_prices[symbol] = hist['Close'] # Store only close prices
            close_prices[symbol].columns = ['close'] # Rename column to 'close'
        else:
            close_prices[symbol] = hist[['Open', 'High', 'Low', 'Close', 'Volume']] # Store OHLCV data
            close_prices[symbol] = close_prices[symbol].reset_index()
            close_prices[symbol].columns = ['date', 'open', 'high', 'low', 'close', 'volume'] # Rename columns
    return pd.concat(close_prices, axis=1).dropna() # Concatenate data for all symbols and drop any NaN values