import csv
import ystockquote
#import yahoo_finance
import os
import time
import datetime

from urllib2 import urlopen

import numpy as np

# Format
# date,open,high,low,close,volume

def fetch_historical(symbol, date_start, date_end):
    #symbol = 'GOOG'
    #data = ystockquote.get_historical_prices(symbol, '2015-12-31', '2016-01-23')

    start = time.clock()
    print(symbol, date_start, date_end)
    data = ystockquote.get_historical_prices(symbol, date_start, date_end)
    elapsed = time.clock()
    elapsed = elapsed - start
    print('Fetched. ', elapsed, ' s.', 'Processing...')

    quote = ''
    np_string = ''
    quote_np = []
    dates = sorted(data.keys(), reverse=True)
    for i in range(0, len(data)):
        np_string = ''
        #if(data[i]['Volume'] == 0):
        #  continue
        date = dates[i]
        #date2 = date.replace("-", "")
        #np_string += (date2 + ',')
        np_string += (data[date]['Open'] + ',')
        np_string += (data[date]['High'] + ',')
        np_string += (data[date]['Low'] + ',')
        np_string += (data[date]['Close'] + ',')
        np_string += (data[date]['Volume'] + '\n')
        quote_np += [np.fromstring(np_string, sep=',')]

    quote_np = np.array(quote_np)
    elapsed = time.clock()
    elapsed = elapsed - start
    print('Complete', elapsed, ' s')
    return quote_np
    #from fetch import fetch_historical
    #data = fetch_historical('GOOGL', '2014-01-01', '2015-01-01')

def fetch_today(symbol):
    return np.fromstring(urlopen('http://finance.yahoo.com/d/quotes.csv?s=' + symbol + '&f=ohgl1v').read(), sep = ',')

def fetch_since(symbol, date_start):
    yesterday = datetime.date.fromordinal(datetime.date.today().toordinal()-1).strftime('%Y-%m-%d')
    data = np.flipud(fetch_historical(symbol, date_start , yesterday))
    #count = 0
    #while True:
        #try:
            #equity = yahoo_finance.Share(symbol)
        #except Exception, e:
            #count += 1
            #print e
            #print 'Failed ', count
            #time.sleep(0.5)
            #continue
        #break
    ##print [equity.get_open(), equity.get_days_high(), equity.get_days_low(), equity.get_price(), equity.get_volume()]
    #data_today = np.array([[float(equity.get_open()), float(equity.get_days_high()), float(equity.get_days_low()), float(equity.get_price()), float(equity.get_volume())]])
    data_today = fetch_today(symbol)
    result = np.vstack((data, data_today))
    return result

if __name__ == "__main__":
    date_start = '2000-01-01' # We use 2000 as first year because most stock data before then will be incomplete and missign volume
    directory = './data_npy/'
    full_update = False # Set to true to redownload all data since date_start, false to just concat today's data
    wait = 0.1 # So we don't end up DOSing Yahoo Finance servers

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open('symbols.csv','r') as symbols_file:
        symbols = symbols_file.readlines()
        for s in symbols:
            s_clean = s[:-2]
            fn = directory + s_clean + '.npy'
            if not os.path.exists(fn) or full_update == False:
                data = fetch_since(s_clean, date_start)
                np.save(fn, data)
            else:
                data = np.concat([fetch_today(s_clean), np.load(fn)], axis = 0) # Just load and add today's data
                np.save(fn, data)
            time.sleep(wait)
