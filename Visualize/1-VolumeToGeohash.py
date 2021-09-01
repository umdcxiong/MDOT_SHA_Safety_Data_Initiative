import numpy as np
import pandas as pd
from shapely.ops import nearest_points, Point, MultiPoint, MultiLineString, LineString
import time
import geopandas as gpd
import geopy
from geopy.distance import great_circle, geodesic
import geopandas
import pickle
from shapely.geometry import LineString
from functools import partial
import multiprocessing
from multiprocessing import Pool
import shapely.geometry as sg
import os
import gc
from datashader.utils import lnglat_to_meters as webm
import math
from shapely import wkt
import json
from pandas.io.json import json_normalize
import pygeohash as pgh
from time import perf_counter
from geographiclib.geodesic import Geodesic
import random

n = 1   # meter
# n = 0.149   # meter
version = '1.0'
measure = 'Vehicle Volume'
season = 'Spring'
DOW = 'Weekday'
TOD = 'AM'
mean_offset_rand = 0
std_offset_lb = 0.5    # lower bound
std_offset_hb = 0.75   # higher bound

link = pd.read_csv('popup_data/CountyFullList/link/Baltimore City/Spring/Weekday/AM/link.csv', index_col=0).reset_index().drop('index', axis=1)

def get_bearing(lat1, lat2, long1, long2):
    brng = Geodesic.WGS84.Inverse(lat1, long1, lat2, long2)['azi1']
    return brng


# i = 0
# j = 0
# t=0
# c=0

# comment this line if you are trying to 
# produce for the whole Baltimore city: 
# it'll take a long time in a local computer
link = link.head(10)

# df_volume = link

def node_gen(df_volume, n, measure = 'Vehicle Volume'):
    t1 = time.perf_counter()
    print('Run task (%s)...' % (os.getpid()))
    Node_Info = []
    N_tasks = len(df_volume)
    milestones = [15, 30, 45, 60, 75, 90, 100]
    trip_count_max = df_volume[measure].max()
    trip_count_min = df_volume[measure].min()
    df_volume = df_volume.dropna(subset=[measure])
    for i in range(N_tasks):
        Line = df_volume['geometry'][i]
        Line = list(wkt.loads(Line).coords)
        from_id = df_volume['from_osm_node_id'][i]
        to_id = df_volume['to_osm_node_id'][i]
        trip_count = df_volume[measure][i]

        # by ratio
        std_offset_rand = (trip_count-trip_count_min) / \
            (trip_count_max-trip_count_min) * \
            (std_offset_hb - std_offset_lb) + std_offset_lb

        int_count = int(round(trip_count))
        # If it is a multipolyline
        for j in range(len(Line)-1):
            linestring = LineString([Line[j], Line[j+1]])

            bearing = get_bearing(
                Line[j][1], Line[j+1][1], Line[j][0], Line[j+1][0])
            if bearing > 270:
                bearing = 360 - bearing
            elif bearing > 180:
                bearing = bearing - 180
            elif bearing > 90:
                bearing = 180 - bearing
                pass
            
            std_offset_rand = std_offset_rand + 1.5*(pow(math.exp(90-bearing),1/10)-1)/8102.08392757538

            linelength = geodesic(Line[j], Line[j+1]).meters
            # n meter interval
            i_num = int(linelength/n) + 1
            mt_point = MultiPoint([linestring.interpolate(
                (k/i_num), normalized=True) for k in range(1, i_num)])

            for t in range(i_num-1):
                for c in range(int_count):
                    rand_d = np.random.normal(
                        mean_offset_rand, std_offset_rand, 1)[0]
                    rand_angl = np.random.uniform(0, 360, 1)[0]
                    d = geodesic(meters=rand_d)
                    temp = d.destination(
                        point=[mt_point[t].x, mt_point[t].y], bearing=rand_angl)
                    centroid = [from_id, to_id, temp[0], temp[1], 1]
                    Node_Info.append(centroid)
                lon = mt_point[t].x
                lat = mt_point[t].y
                centroid = [from_id, to_id, lon, lat, 1]
                Node_Info.append(centroid)
        percentage_complete = (100.0 * (i+1) / N_tasks)
        while len(milestones) > 0 and percentage_complete >= milestones[0]:
            t1_1 = time.perf_counter()
            print("%s pct completed in %s seconds" %
                  (milestones[0], round(t1_1-t1, 2)))
            # remove that milestone from the list
            milestones = milestones[1:]

    Node_Info = geopandas.GeoDataFrame(
        Node_Info, columns=['from_osm_node_id', 'to_osm_node_id', 'longitude', 'latitude', 'count'])
    t2 = time.perf_counter()
    print('Task (%s) finished.' % (os.getpid()))
    print(
        f'Interpolation and Random Draw Task finished in {round(t2-t1, 2)} seconds')
    t3 = time.perf_counter()
    Node_Info['geohash11'] = [pgh.encode(y, x, precision=11)
                              for (x, y) in zip(Node_Info['longitude'], Node_Info['latitude'])]
    Node_Info = Node_Info[['geohash11', 'count']]
    print(f'Geohash Generated in {round(t3-t2, 2)} seconds')
    print(f'Task finished in {round(t3-t1, 2)} seconds')
    return Node_Info


df = node_gen(link, n)
# df = node_gen(link, n, measure = 'Pedestrian Bicycle Volume')

dataset_path = 'output_data/%s' % measure
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

df.to_csv(dataset_path + '/Geohash_%s_%s_%s_%s.csv' % (season, DOW, TOD, version), index=False)