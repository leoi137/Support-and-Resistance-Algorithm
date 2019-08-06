from datetime import timedelta
from collections import deque

from mpl_finance import candlestick_ohlc
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates

import pandas as pd
import numpy as np
import time


class SupportResistance():

    """
    Parameters:

    data: Csv data with open, high, low, close, prices (lower case)
    rolling_std: Period length to take a rolling stdv
    partial_std: Take more or part of the standard deviation
    partial_std_amount: How much you want to take part of the std so if the std is 2.0,
    and the amount is 0.5 then the std taken into acount is 1.0
    frame: What candle data are you using? (Daily or Hour only)
    """
    
    def __init__(self, data, rolling_std = 100, partial_std = True, partial_std_amount = 2, frame = 'Daily'):
        
        self.data = data
        self.rolling_std = rolling_std
        self.partial_std = partial_std
        self.partial_std_amount = partial_std_amount
        self.frame = frame
        
    def main(self, view_graph = False, start = -200, end = -1, candle_width = 0.5):

        """
        Parameters:

        view_graph: View a plotted candlestick graph
        start: The index of every candle to start viewing the data at
        end: The index of every candle to end viewing the data at
        """
        
        algo_start = time.perf_counter()
        
        df_h, df_l, df = self.get_prev_next_values()
        
        drop_min = self.locate_real_min(
            df_l['Prices'], df_l.index,
            df_l['Prices -1'], df_l['Prices -2'], df_l['Prices -3'], df_l['Prices -4'],
            df_l['Prices 1'], df_l['Prices 2'], df_l['Prices 3'], df_l['Prices 4'])

        drop_max = self.locate_real_max(
            df_h['Prices'], df_h.index,
            df_h['Prices -1'], df_h['Prices -2'], df_h['Prices -3'], df_h['Prices -4'],
            df_h['Prices 1'], df_h['Prices 2'], df_h['Prices 3'], df_h['Prices 4'])
        
        filtered_df = self.filter_to_dataframe(df_h, df_l, df, drop_min, drop_max)

        mima = self.get_min_max_values_with_stdv(filtered_df)

        filtered_mima = self.consecutive_min_max_filter(mima)

        # return filtered_mima
        filtered_mima = self.stdv_varying_support_resistance(filtered_mima)
        support_resistance, last_SR = self.consecutive_min_max_filter(filtered_mima, end = True) #last_SR[1] is the timestamp of the SR level
        
        if self.frame == 'Daily':
            if last_SR[2] == 'Support':
                support_resistance = self.last_SR_filter(
                    support_resistance, last_SR, self.data[['high']][last_SR[1] + timedelta(days = 1):], 'Support')
            else:
                support_resistance = self.last_SR_filter(
                    support_resistance, last_SR, self.data[['low']][last_SR[1] + timedelta(days = 1):],'Resistance')
        else:
            if last_SR[2] == 'Support':
                support_resistance = self.last_SR_filter(
                    support_resistance, last_SR, self.data[['high']][last_SR[1] + timedelta(hours = 1):], 'Support')
            else:
                support_resistance = self.last_SR_filter(
                    support_resistance, last_SR, self.data[['low']][last_SR[1] + timedelta(hours = 1):],'Resistance')
        
        print(f"Calculating S/R Took: {time.perf_counter() - algo_start: 0.4f} seconds")
        
        if view_graph:
            all_df = self.view_plot(support_resistance)
            self.plot_support_resistance(all_df, start, end, candle_width)
            
        return pd.DataFrame(support_resistance, columns = ['Price', 'Date', 'Kind', 'STDV', 'PCT'])#, filtered_mima
    
    def get_prev_next_values(self):
        
        df_h = self.data['high'].values
        df_l = self.data['low'].values
        df = self.data[['close']]
        
        df.columns = ['Prices']
        
        start = 4
        data_h = []
        data_l = []
        for i in range(start, len(df_h) - start):
            data_h.append(
                [df_h[i], df_h[i - 1], df_h[i - 2], df_h[i - 3], df_h[i - 4],
                 df_h[i + 1], df_h[i + 2], df_h[i + 3], df_h[i + 4]])
            
        for i in range(start, len(df_l) - start):
            data_l.append(
                [df_l[i], df_l[i - 1], df_l[i - 2], df_l[i - 3], df_l[i - 4],
                 df_l[i + 1], df_l[i + 2], df_l[i + 3], df_l[i + 4]])

        data_h_df = pd.DataFrame(data_h, index = self.data[['close']][4:-4].index)
        data_h_df.columns = ['Prices', 'Prices -1', 'Prices -2', 'Prices -3', 'Prices -4', 
                             'Prices 1', 'Prices 2', 'Prices 3', 'Prices 4']
        data_l_df = pd.DataFrame(data_l, index = self.data[['close']][4:-4].index)
        data_l_df.columns = ['Prices', 'Prices -1', 'Prices -2', 'Prices -3', 'Prices -4', 
                             'Prices 1', 'Prices 2', 'Prices 3', 'Prices 4']
        
        return data_h_df, data_l_df, df
    
    def locate_real_min(self, price, index, prev1, prev2, prev3, prev4, fut1, fut2, fut3, fut4):
        drop = []
        for p, ind, p1, p2, p3, p4, f1, f2, f3, f4 in zip(price, index, 
                                                          prev1, prev2, prev3, prev4, 
                                                          fut1, fut2, fut3, fut4):
            if (p1 > p < f1) and (p2 > p < f2) and (p3 > p < p3) and (p4 > p < p4):
                pass
            else:
                drop.append(ind)
        return drop

    def locate_real_max(self, price, index, prev1, prev2, prev3, prev4, fut1, fut2, fut3, fut4):
        drop = []
        for p, ind, p1, p2, p3, p4, f1, f2, f3, f4 in zip(price, index, 
                                                          prev1, prev2, prev3, prev4, 
                                                          fut1, fut2, fut3, fut4):
            if (p1 < p > f1) and (p2 < p > f2) and (p3 < p > f3) and (p4 < p > p4):
                pass
            else:
                drop.append(ind)
        return drop
    
    def filter_to_dataframe(self, df_h, df_l, df, drop_min, drop_max):  # Add STDV here
        
        min_df = []
        max_df = []
        drop_min = set(drop_min)
        drop_max = set(drop_max)
        df_w_std = df['Prices'].pct_change().rolling(self.rolling_std).std() * self.partial_std_amount

        for p, ind in zip(df_l['Prices'], df_h.index):
            if ind in drop_min:
                min_df.append(np.nan)
            else:
                min_df.append(p)

        for p, ind in zip(df_h['Prices'], df_l.index):
            if ind in drop_max:
                max_df.append(np.nan)
            else:
                max_df.append(p)
                
        new_flt_df = pd.concat([
            df['Prices'], 
            pd.DataFrame(min_df, index = df_h.index), 
            pd.DataFrame(max_df, index = df_l.index),
            df_w_std
        ], axis = 1)

        new_flt_df.columns = ['Prices', 'Support', 'Resistance', 'STDV']
        
        return new_flt_df
    
    def get_min_max_values_with_stdv(self, filtered_df):

        mima = []

        for mi, ma, ind, stdv in zip(filtered_df['Support'].values, filtered_df['Resistance'].values, filtered_df.index, filtered_df['STDV'].values):
            if mi > 0:
                mima.append([mi, ind, "Support", stdv])
            if ma > 0:
                mima.append([ma, ind, "Resistance", stdv])
                
        return mima
                
    def consecutive_min_max_filter(self, mima, end = False):

        min_count = 0
        max_count = 0
        min_q = []
        max_q = []
        counter = 0
        new_mima = []

        for e in mima:

            if e[2] == 'Support':
                min_count += 1
                min_q.append(e[0:5])

            if e[2] == 'Support' and max_count > 0:  # Previous max_count
                if max_count > 1:
                    new_mima.append(max(max_q))
                else:
                    new_mima.append(max_q[0])
                    
                max_count = 0
                max_q = []

            if e[2] == 'Resistance':
                max_count += 1
                max_q.append(e[0:5])

            if e[2] == 'Resistance' and min_count > 0:   # Previous min_count
                if min_count > 1:
                    new_mima.append(min(min_q))
                else:
                    new_mima.append(min_q[0])

                min_count = 0
                min_q = []
                
            counter += 1
            
        if counter == len(mima):
            if min_count > 1:
                new_mima.append(min(min_q))
            elif min_count == 1:
                new_mima.append(min_q[0])
            elif max_count > 1:
                new_mima.append(max(max_q))
            elif max_count == 1:
                new_mima.append(max_q[0])
            else:
                pass

        if end:
            # print(new_mima[-1]) this is the last SR level with all its data
            return new_mima, new_mima[-1]# + [mima[-1][4]]
        else:
            return new_mima

    def stdv_varying_support_resistance(self, filtered_mima):

        mima_df = pd.DataFrame(filtered_mima)
        mima_df.columns = ['Prices', 'Location', 'Kind', 'STDV']
        mima_df['PCT'] = mima_df['Prices'].pct_change()

        clean_mima = []
        que = deque(maxlen = 2)
        for e in mima_df.values.tolist():
            que.append(e)
            if len(que) > 1:
                if que[1][3] > 0:
                    if abs(que[1][4]) > que[1][3]:
                        clean_mima.append(que[1])
                else:
                    if abs(que[1][4]) > 0.005:
                        clean_mima.append(que[1])
        return clean_mima
    
    # def stdv_varying_support_resistance(self, filtered_mima):
        
    #     mima_df = pd.DataFrame(filtered_mima)
    #     mima_df.columns = ['Prices', 'Location', 'Kind']
    #     mima_df['PCT'] = mima_df['Prices'].pct_change()
        
    #     if self.partial_std:
    #         mima_df['STDV'] = mima_df['PCT'].abs().rolling(
    #             self.rolling_std).std()*self.partial_std_amount
    #     else:
    #         mima_df['STDV'] = mima_df['PCT'].abs().rolling(
    #             self.rolling_std).std()

    #     clean_mima = []
    #     que = deque(maxlen = 2)
    #     for e in mima_df.values.tolist():
    #         que.append(e)
    #         if len(que) > 1:
    #             if que[1][4] > 0:
    #                 if abs(que[1][3]) > que[1][4]:
    #                     clean_mima.append(que[1])
    #             else:
    #                 if abs(que[1][3]) > 0.005:
    #                     clean_mima.append(que[1])
    #     return clean_mima
    
    def last_SR_filter(self, minmax, last_val, prices, kind):
        
        if kind == 'Resistance':
            swing = float(min(prices['low']))
        else:
            swing = float(max(prices['high']))
        
        diff = abs((swing - last_val[0]) / last_val[0])
        # print(diff)
        # print(last_val)
        # print(last_val[-2] * self.partial_std_amount)
        
        minmax = pd.DataFrame(minmax, columns = ['Price', 'Date', 'Kind', 'STDV', 'PCT'])
        if diff < last_val[-2] * self.partial_std_amount:
            minmax = minmax.drop([len(minmax) - 1])
        
        return minmax.values
        
    
    def view_plot(self, view):

        start_all = time.perf_counter()

        new_df = self.data[['close']]
        min_ind = []
        max_ind = []

        for e in view:
            if e[2] == 'Support':
                min_ind.append(e[0:2])
            else:
                max_ind.append(e[0:2])

        min_ind_vals = [x[1] for x in min_ind]
        max_ind_vals = [x[1] for x in max_ind]

        new_df_min = []
        new_df_max = []

        min_ind_df = pd.DataFrame(min_ind)
        min_ind_df.index = min_ind_df[1]
        min_ind_df.drop([1], axis = 1, inplace = True)

        max_ind_df = pd.DataFrame(max_ind)
        max_ind_df.index = max_ind_df[1]
        max_ind_df.drop([1], axis = 1, inplace = True)

        for i in new_df.index:
            if i in min_ind_vals:
                new_df_min.append(float(min_ind_df.loc[i]))
            else:
                new_df_min.append(np.nan)

        for i in new_df.index:
            if i in max_ind_vals:
                new_df_max.append(float(max_ind_df.loc[i]))
            else:
                new_df_max.append(np.nan)

        all_df = pd.concat([
            new_df, 
            pd.DataFrame(new_df_min, index = new_df.index), 
            pd.DataFrame(new_df_max, index = new_df.index)
        ], axis = 1)

        all_df.columns = ['Prices', 'Support', 'Resistance']

        print(f"Getting all data took: {time.perf_counter() - start_all:0.4f} seconds")

        return all_df
    
    def plot_support_resistance(self, all_df, start, end, candle_width):
        
        candle_data = self.data[['open', 'high', 'low', 'close']][start:end]
        candle_data['open'] = candle_data.open.astype(float)
        candle_data['high'] = candle_data.high.astype(float)
        candle_data['low'] = candle_data.low.astype(float)
        candle_data['close'] = candle_data.close.astype(float)

        candle_data['time'] = candle_data.index
        candle_data['date_ax'] = candle_data['time'].apply(lambda date: date2num(date))
        candle_data.drop('time', axis = 1, inplace = True)
        columns = ['date_ax', 'open','high','low','close']
        fx_vals = [tuple(vals) for vals in candle_data[columns].values]
        
        fig, ax = plt.subplots(figsize = (9, 7))

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        candlestick_ohlc(ax, fx_vals, width = candle_width, colorup = 'g', colordown = 'r')
        ax.plot(all_df['Support'][start:end], '_', label = 'Support',
                 c = 'green', markersize = 25, markeredgewidth = 3)
        ax.plot(all_df['Resistance'][start:end], '_', label = 'Resistance',
                 c = 'red', markersize = 25, markeredgewidth = 3)
        
        plt.legend()
        plt.xticks(rotation = 45)
        plt.show()

    def get_all_data(self, SR_data, prices):

        """
        Parameters:

        SR_data: support and resistance data
        prices: open, high, low, close prices 

        """
        
        start_all_d = time.perf_counter()
        
        SR_data = pd.DataFrame(SR_data).set_index('Date')
        SR_index = set(SR_data.index.values)
        
        all_data = []
        
        for ind in prices.index.values:
            if ind in SR_index:
                all_data.append(SR_data.loc[ind].values[1:3])
            else:
                all_data.append([np.nan, np.nan])
                
        all_data = pd.concat([
            prices, 
            pd.DataFrame(all_data, index = prices.index, columns = ['Kind', 'PCT'])], axis = 1)
                
        print(f"Structuring All Data Took: {time.perf_counter() - start_all_d:0.4f} seconds") 
        
        return pd.DataFrame(all_data)