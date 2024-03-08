# Packages
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

def get_swing_high_lows(ohlc, stdv_multiple = 3):
    swing_highs = []
    swing_lows = []
    kind = None
    close_prices = ohlc['close']
    sequence_df = pd.DataFrame(columns=['date','close'])
    # Rolling window needs to be shorter than your total data points
    close_stdv = np.log(ohlc['close'] / ohlc['close'].shift(1)).rolling(window=150).std().mean()*100
    pct_change = close_stdv * stdv_multiple
    print(f"Swings = > {pct_change} %")
    # Loop through the list of close prices
    for i in tqdm(range(len(close_prices))):
        current_date = close_prices.index[i]
        current_price = close_prices[i]
        new_row_df = pd.DataFrame({'date': [current_date], 'close': [current_price]})
        sequence_df = pd.concat([sequence_df, new_row_df], ignore_index=True)

        # Enough data to view
        if i > 10:
            # High Swing
            price_maxidx = sequence_df['close'].idxmax()
            max_date = sequence_df['date'][price_maxidx]
            max_price = sequence_df['close'][price_maxidx]
            max_historical_prices = sequence_df['close'][:price_maxidx]
            max_future_prices = sequence_df['close'][price_maxidx:]

            if kind == None or kind == 'swing_low':
                historical_price_change = np.log(max_price / max_historical_prices.min()) * 100
                future_price_change = np.log(max_future_prices.min() / max_price) * 100
                if historical_price_change > pct_change and future_price_change < -pct_change:
                    # print(f"\n*********** Max $: {max_price} - {max_date}")
                    # print(f"Historical: {historical_price_change}% - Future: {future_price_change}%")
                    swing_highs.append(max_date)
                    kind = 'swing_high'
                    sequence_df = sequence_df.iloc[price_maxidx:]

            # Low Swing
            price_minidx = sequence_df['close'].idxmin()
            min_date = sequence_df['date'][price_minidx]
            min_price = sequence_df['close'][price_minidx]
            min_historical_prices =  sequence_df['close'][:price_minidx]
            min_future_prices = sequence_df['close'][price_minidx:]

            if kind == None or kind == 'swing_high':
                historical_price_change = np.log(min_price / min_historical_prices.max()) * 100
                future_price_change = np.log(min_future_prices.max() / min_price) * 100
                if historical_price_change < -pct_change and future_price_change > pct_change:
                    # print(f"\n*********** Min $: {max_price} - {max_date}")
                    # print(f"Historical: {historical_price_change}% - Future: {future_price_change}%")
                    swing_lows.append(min_date)
                    kind = 'swing_low'
                    sequence_df = sequence_df.iloc[price_minidx:]
                    
    return swing_highs, swing_lows

# Concatenates price and swing dates into a DataFrame
def ohlc_dates_to_prices_df(ohlc_df, dates_array):
    with_swing = []

    for index in ohlc_df.index:
        # print(index.to_datetime64() in support_resistance['Date'])
        if index in dates_array:
            with_swing.append(ohlc_df.loc[index]['close'])
        else:
            with_swing.append(np.nan)
    return pd.concat([ohlc_df, pd.DataFrame(with_swing, index=ohlc_df.index, columns=['swing'])], axis = 1)


# Define a small function to check if the levels are approximately the same
def approx_same_level(price1, price2, tolerance=0.02):  # 2% tolerance, adjust as needed
    return abs(price1 - price2) / price1 <= tolerance

def greater_than_level(price1, price2, tolerance=0.005):  # 0.5% tolerance, adjust as needed
    return abs(price1 - price2) / price1 >= tolerance

def get_head_and_shoulders(swings_df):
    hns_patterns = []
    for i in range(4, len(swings_df)):
        left_shoulder = swings_df.iloc[i - 4]
        left_peak_or_trough = swings_df.iloc[i - 3]
        head = swings_df.iloc[i - 2]
        right_peak_or_trough = swings_df.iloc[i - 1]
        right_shoulder = swings_df.iloc[i]

        # Check the distance between the head and the shoulders
        if (head.date - left_shoulder.date).days <= 7 and (right_shoulder.date - head.date).days <= 7:
            # Top
            if head.high > left_shoulder.high and head.high > right_shoulder.high and head.high > left_peak_or_trough.high and head.high > right_peak_or_trough.high:
                # if approx_same_level(left_shoulder.high, right_shoulder.low) and approx_same_level(left_peak_or_trough.low, right_peak_or_trough.low, tolerance=0.05):
                if greater_than_level(left_shoulder.high, head.high) and greater_than_level(right_shoulder.high, head.high) \
                    and approx_same_level(left_shoulder.high, right_shoulder.high):
                    hns_patterns.append((left_shoulder.name, left_peak_or_trough.name, head.name, right_peak_or_trough.name, right_shoulder.name))
                
            # Bottom
            if head.low < left_shoulder.low and head.low < right_shoulder.low and head.low < left_peak_or_trough.low and head.low < right_peak_or_trough.low:
                if greater_than_level(left_shoulder.low, head.low) and greater_than_level(right_shoulder.low, head.low) \
                    and approx_same_level(left_shoulder.low, right_shoulder.low):
                    hns_patterns.append((left_shoulder.name, left_peak_or_trough.name, head.name, right_peak_or_trough.name, right_shoulder.name))

    return hns_patterns

def plot_swing_close(data):
    """
    Plots 'swing' as scatter points and 'close' as a line plot.

    Parameters:
    data (DataFrame): A pandas DataFrame with 'swing' and 'close' columns.
    """
    # Ensure that 'swing' and 'close' are present in the data
    if 'swing' not in data.columns or 'close' not in data.columns:
        raise ValueError("Data must contain 'swing' and 'close' columns")
    
    data = data.reset_index()

    # Plotting
    plt.figure(figsize=(18, 9))
    plt.plot(data.index, data['close'], color='blue', label='Close')
    plt.scatter(data.index, data['swing'], color='red', label='Swing', s=100)

    # Adding titles and labels
    plt.title('Swing and Close Values Over Time')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()

    # Show the plot
    plt.show()

def plot_support_resistance_candlesticks(data, start, end):
    
    candle_data = data[['open', 'high', 'low', 'close']][start:end]
    candle_data['open'] = candle_data.open.astype(float)
    candle_data['high'] = candle_data.high.astype(float)
    candle_data['low'] = candle_data.low.astype(float)
    candle_data['close'] = candle_data.close.astype(float)

    # Set the index to the date for mplfinance
    candle_data.index = pd.to_datetime(candle_data.index)

    # Define additional plots for Support and Resistance
    apds = [mpf.make_addplot(data['swing'][start:end], type='scatter', markersize=175, marker='_', color='blue', linewidths=5)]

    # # Define additional plots for Support and Resistance
    # apds = [mpf.make_addplot(all_df['Support'][start:end], type='scatter', markersize=250, marker='_', color='green'),
    #         mpf.make_addplot(all_df['Resistance'][start:end], type='scatter', markersize=250, marker='_', color='red')]

    # Plot using mplfinance
    # mpf.plot(candle_data, type='candle', style='charles', figscale=1.5, xrotation=45, figsize=(18, 8))
    mpf.plot(candle_data, type='candle', style='charles', addplot=apds, figscale=1.5, xrotation=45, figsize=(18, 9))

def ohlc_swings_df(ohlc_df, high_dates, low_dates):
    # Initialize 'kind' as object type for mixed data types (strings and floats)
    ohlc_df = ohlc_df.copy()
    ohlc_df['kind'] = np.nan
    ohlc_df['kind'] = ohlc_df['kind'].astype('object')

    # Fill in the high and low swings
    for index in ohlc_df.index:
        if index in high_dates:
            ohlc_df.at[index, 'kind'] = 'high'
        elif index in low_dates:
            ohlc_df.at[index, 'kind'] = 'low'

    # Remove rows where 'kind' is NaN
    ohlc_swing_df_cleaned = ohlc_df.dropna(subset=['kind']).copy()

    # Calculate log returns
    ohlc_swing_df_cleaned['log_return'] = np.log(ohlc_swing_df_cleaned['close'] / ohlc_swing_df_cleaned['close'].shift(1))

    # Calculate time differences as timedelta objects
    ohlc_swing_df_cleaned['time_diff'] = ohlc_swing_df_cleaned['date'].diff()

    # Calculate velocity: log_return / time_diff in days
    ohlc_swing_df_cleaned['velocity'] = ohlc_swing_df_cleaned['log_return'].abs() / ohlc_swing_df_cleaned['time_diff'].dt.total_seconds().abs() * (24*3600)

    # Remove the first row as it will have NaN values for log return and time difference
    ohlc_swing_df_cleaned = ohlc_swing_df_cleaned[1:]

    return ohlc_swing_df_cleaned