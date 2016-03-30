import csv
import ystockquote
import os
import time
import datetime

import numpy as np

# Format
# date,open,high,low,close,volume

def fetch_historical(symbol, date_start, date_end):
  #symbol = 'GOOG'
  #data = ystockquote.get_historical_prices(symbol, '2015-12-31', '2016-01-23')

  print symbol
  start = time.clock()
  data = ystockquote.get_historical_prices(symbol, date_start, date_end)
  elapsed = time.clock()
  elapsed = elapsed - start
  print('Fetched. ', elapsed, ' s.', 'Processing...')

  quote = ''
  np_string = ''
  dates = sorted(data.keys(), reverse=True)
  for i in range(0, len(data)):
    #if(data[i]['Volume'] == 0):
    #  continue
    date = dates[i]
    date2 = date.replace("-", "")
    np_string += (date2 + ',')
    np_string += (data[date]['Open'] + ',')
    np_string += (data[date]['High'] + ',')
    np_string += (data[date]['Low'] + ',')
    np_string += (data[date]['Close'] + ',')
    np_string += (data[date]['Volume'] + '\n')
    quote = symbol + ',' + np_string

  fileName = symbol + ".csv"
  file = open(fileName, "w")
  #file.write(quote)
  file.write(np_string)
  file.close()
  elapsed = time.clock()
  elapsed = elapsed - start
  #quote_np = np.fromstring(np_string, sep=',') ### Can't get this to work
  quote_np = np.genfromtxt(fileName, delimiter=',') ### So this will have to suffice
  print('Complete', elapsed, ' s')
  return quote_np
#from fetch import fetch_historical
#data = fetch_historical('GOOGL', '2014-01-01', '2015-01-01')