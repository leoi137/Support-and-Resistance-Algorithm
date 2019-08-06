import pandas as pd
import pandas_datareader as web
import time

class YahooData():

    def __init__(self, symbols, start, end):

        self.symbols = symbols
        self.start = start
        self.end = end

        self.data = {}

    def get_data(self):

        print("Gathering Data...\n")
        get_data = time.perf_counter()

        for s in self.symbols:
            self.data[s] = web.DataReader(s, 'yahoo', self.start, self.end)

        for s in self.symbols:
            self.data[s].drop('Close', axis = 1, inplace = True)
            self.data[s].columns = ['high', 'low', 'open', 'volume', 'close']

        print("Gathering the data took: {:0.4f} seconds\n".format(time.perf_counter() - get_data))

        return self.data