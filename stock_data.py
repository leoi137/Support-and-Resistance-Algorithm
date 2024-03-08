from tqdm import tqdm
import yfinance as yf
import pandas as pd

def get_yahoo_multiple_historical_stock_data(symbol_list, period, interval = '1d', only_close = False):
    """
    Parameters:
        - symbols_list: ['TSLA', 'SPY', ..., 'RKT']
        - period: {1+}d, {1+}mo, max
        - interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    """
    close_prices = {}
    for symbol in symbol_list:
        my_symbol = yf.Ticker(symbol)
        hist = my_symbol.history(period=period, interval=interval)
        if only_close:
            close_prices[symbol] = hist['Close']
            close_prices[symbol].columns = ['close']
        else:
            close_prices[symbol] = hist[['Open', 'High', 'Low', 'Close', 'Volume']]
            close_prices[symbol] = close_prices[symbol].reset_index()
            close_prices[symbol].columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    return pd.concat(close_prices, axis=1).dropna()