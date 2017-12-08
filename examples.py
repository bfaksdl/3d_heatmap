from bin_count_cl import *
#from only_bin_test import *

from datetime import datetime
import pandas as pd
import os

#http://www.nyc.gov/html/tlc/html/about/trip_record_data.shtml
csvFilePath = 'D:\\Letöltés\\2017.10\\yellow_tripdata_2015-09.csv'

def readData(csvFilePath, columns = []):
    pwd = os.getcwd()
    os.chdir(os.path.dirname(csvFilePath))
    csv_data = pd.read_csv(os.path.basename(csvFilePath))
    os.chdir(pwd)


    dataPanda = []
    for col in columns:
        dataPanda += [csv_data[col]]
    return dataPanda

def toNumpyData(dataPanda):
    data = []
    for c in dataPanda:
        data += [np.asanyarray(c, dtype=np.float32)]
        c = []

    return data

def readDataHour(csvFilePath):
    csv_columns = ['pickup_latitude', 'pickup_longitude', 'tpep_pickup_datetime']

    dataPanda = readData(csvFilePath, csv_columns)

    for i in range(len(dataPanda[2])):
        tmp = datetime.strptime(dataPanda[2][i], '%Y-%m-%d %H:%M:%S')
        dataPanda[2][i] = float(tmp.hour*60.0 + tmp.minute)
    return dataPanda

def demo1():
    #mt.measure('Program begins')
    global csvFilePath

    #csv_columns = ['total_amount', 'pickup_longitude', 'pickup_latitude']
    #csv_columns = ['pickup_latitude', 'pickup_longitude', 'trip_distance']
    csv_columns = ['dropoff_latitude'
        , 'dropoff_longitude'
        , 'trip_distance']
    data_h = toNumpyData(readData(csvFilePath, csv_columns))

    #mt.measure('data creation')

    bin_limits = [[40.81937822876625, 40.606283], [-74.03377532958984, -73.753336], [0, 30]]
    bin_counts = [100, 100, 100]
    bins_h = bincount_cl(data_h, bin_limits, bin_counts)
    #show_visvis(bins_h)
    show_vispy(bins_h)

def demo2():
    global csvFilePath
    data_h = toNumpyData(readDataHour(csvFilePath))
    bin_limits = [[40.81937822876625, 40.68205263933122], [-74.03377532958984, -73.8442611694336], [0, 24*60]]
    bin_counts = [100, 100, 100]
    bins_h = bincount_cl(data_h, bin_limits, bin_counts)
    show_visvis(bins_h)
    show_vispy(bins_h)


def demo3():
    #mt.measure('Program begins')
    global csvFilePath

    #csv_columns = ['total_amount', 'pickup_longitude', 'pickup_latitude']
    csv_columns = ['pickup_latitude', 'pickup_longitude', 'trip_distance']
    #csv_columns = ['dropoff_latitude', 'dropoff_longitude', 'trip_distance']
    data_h = toNumpyData(readData(csvFilePath, csv_columns))

    #mt.measure('data creation')

    bin_limits = [[40.81937822876625, 40.606283], [-74.03377532958984, -73.753336], [0, 30]]
    bin_counts = [100, 107, 119]
    bins_h = bincount_cl(data_h, bin_limits, bin_counts)

    binned_data = bin_cpu(data_h, bin_limits, bin_counts)
    count_h = only_count_cl(binned_data)
    show_visvis(count_h)

    show_visvis(bins_h)
    #show_vispy(bins_h)


def demo4():
#mt.measure('Program begins')
    mt = mytimer()
    global csvFilePath
    data_h = toNumpyData(readDataHour(csvFilePath))
    mt.measure('Dataread: long, lat, hour')

    csv_columns = ['tip_amount']
    data_h2 = toNumpyData(readData(csvFilePath, csv_columns))

    data_h += data_h2
    mt.measure('Dataread: tip')

    bin_limits = [[40.81937822876625, 40.68205263933122], [-74.03377532958984, -73.8442611694336], [0, 24*60]]
    bin_counts = [100, 100, 100]

    binned_data = bin_cpu(data_h, bin_limits, bin_counts)
    mt.measure('binning only')
    volume = only_avg_cl(binned_data)
    mt.measure('avg_only')

    mt.print_times()
#    bins_h = bincount_cl(data_h, bin_limits, bin_counts)
    show_visvis(volume)
    show_vispy(volume)
    show_visvis(volume)


def demo5():
#mt.measure('Program begins')
    mt = mytimer()
    global csvFilePath

    csv_columns = ['pickup_latitude', 'pickup_longitude', 'passenger_count', 'tip_amount']
    data_h = toNumpyData(readData(csvFilePath, csv_columns))


    mt.measure('Dataread')

    bin_limits = [[40.81937822876625, 40.68205263933122], [-74.03377532958984, -73.8442611694336], [0, 15]]
    bin_counts = [100, 100, 100]

    binned_data = bin_cpu(data_h, bin_limits, bin_counts)
    mt.measure('binning only')
    volume = only_avg_cl(binned_data)
    mt.measure('avg_only')

    mt.print_times()
#    bins_h = bincount_cl(data_h, bin_limits, bin_counts)
    show_visvis(volume)
    show_vispy(volume)
    show_visvis(volume)

def demo6():
#mt.measure('Program begins')
    mt = mytimer()
    global csvFilePath

    #csv_columns = ['pickup_latitude', 'pickup_longitude', 'passenger_count', 'tip_amount']
    csv_columns = ['pickup_latitude'
                    , 'pickup_longitude'
                    , 'trip_distance'
                    , 'tip_amount']
    data_h = toNumpyData(readData(csvFilePath, csv_columns))


    mt.measure('Dataread')

    bin_limits = [[40.81937822876625, 40.68205263933122], [-74.03377532958984, -73.8442611694336], [0, 30]]
    bin_counts = [250, 250, 250]

    binned_data = bin_cpu(data_h, bin_limits, bin_counts)
    mt.measure('binning only')
    volume1 = only_avg_cl(binned_data, -10)
    mt.measure('avg_only')
    volume2 = only_count_cl(binned_data)
    mt.measure('count_only')
    volume3 = only_percentile_cl(binned_data, 40, -10)
    mt.measure('avg_only')
    volume4 = only_min_cl(binned_data, -10)
    mt.measure('min_only')
    volume5 = only_max_cl(binned_data, -10)
    mt.measure('max_only')

    mt.measure('should be 0')
    mt.print_times()
#    bins_h = bincount_cl(data_h, bin_limits, bin_counts)
    show_visvis(volume1, csv_columns)
    show_vispy(volume2)
    show_visvis(volume3, csv_columns)
    show_visvis(volume4, csv_columns)
    show_visvis(volume5, csv_columns)

demo1()



