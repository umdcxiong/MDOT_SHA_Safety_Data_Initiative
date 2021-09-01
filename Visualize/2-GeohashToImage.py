import gdal2tiles   # '0.1.9'
import pygeohash as pgh
import numpy as np  # '1.19.2'
import pandas as pd  # '1.0.5'
import time
import multiprocessing as mp
from multiprocessing import Pool
from osgeo import gdal  # '3.0.2'
from osgeo import osr
import os
import sys
from scipy import stats  # '1.5.2'
import matplotlib.pyplot as plt  # '3.3.2'
from palettable.cmocean.sequential import Solar_20  # '3.3.0'
import matplotlib.cm as cm
from pyproj import Proj, transform  # '2.6.1.post1'
from datashader.utils import lnglat_to_meters as webm   # '0.11.0'
from math import ceil, floor
from math import sqrt
import itertools
import yaml  # '5.3.1'
import s3fs
from time import perf_counter
import numpy.ma as ma
import boto3

version = '1.0'

# configuration #####################################
pixels_per_tile = [256, 256]
mapbox_level_list = [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]
opacity = 0.9   # max 1
'''rectangle bound of the area we want to save AS image'''
# whole world
# xmin, ymin, xmax, ymax = [-20026376.39, -20048966.10, 20026376.39, 20048966.10]
# Maryland
# xmin, ymin, xmax, ymax = [-8860000, 4520000, -8330000, 4840000]
# Baltimore Harbor
# xmin, ymin, xmax, ymax = [-8532000, 4760000, -8521000, 4765000]
# Baltimore City
xmin, ymin, xmax, ymax = [-8539800, 4756000, -8519000, 4775000]
# College Park
# xmin, ymin, xmax, ymax = [-8571247, 4715753, -8558252, 4726340]

'''rectangle bound of the area we want to show IN image. dataset filter.'''
# Baltimore Harbor
# xmin_filter, ymin_filter, xmax_filter, ymax_filter = [
#     -8530000, 4760000, -8523000, 4765000]
# Maryland
# xmin_filter, ymin_filter, xmax_filter, ymax_filter = [
#     -8860000, 4520000, -8330000, 4840000]
# Baltimore City
xmin_filter, ymin_filter, xmax_filter, ymax_filter = [
    -8539800, 4756000, -8519000, 4775000]
# College Park
# xmin_filter, ymin_filter, xmax_filter, ymax_filter = [
    # -8571247, 4715753, -8558252, 4726340]

# rectangle bound of the whole earth
xmin_w, ymin_w, xmax_w, ymax_w = [-20026376.39, -
                                  20048966.10, 20026376.39, 20048966.10]


def GenDataGroup(measure, season, DOW, TOD, xmin_filter, ymin_filter, xmax_filter, ymax_filter):
    dataset = pd.read_csv('output_data/%s/Geohash_%s_%s_%s_%s.csv' % (measure, season, DOW, TOD, version))

    # main function #####################
    t1 = time.perf_counter()
    # decode geohash to lat lon
    dataset['loc'] = [pgh.decode(gh)
                      for gh in dataset['geohash11']]  # (lat, lon)
    dataset['lat'] = [x[0] for x in dataset['loc']]
    dataset['lon'] = [x[1] for x in dataset['loc']]
    t2 = time.perf_counter()
    print(f'GeohashID decoded into Lat Lon in {round(t2-t1, 2)} seconds')
    # convert lat lon to epsg:3857, in meters
    dataset['lon_3857'], dataset['lat_3857'] = webm(
        dataset['lon'], dataset['lat'])
    t3 = time.perf_counter()
    print(f'Coordinates converted in {round(t3-t2, 2)} seconds')
    # filter dataset
    dataset = dataset.loc[(dataset['lon_3857'] > xmin_filter) & (dataset['lon_3857'] < xmax_filter) & (
        dataset['lat_3857'] > ymin_filter) & (dataset['lat_3857'] < ymax_filter)]

    # generate data for various zoom levels
    dataset['geohash10'] = dataset['geohash11'].str[: -1]
    dataset['geohash9'] = dataset['geohash11'].str[: -2]
    dataset['geohash8'] = dataset['geohash11'].str[: -3]
    dataset['geohash7'] = dataset['geohash11'].str[: -4]
    dataset['geohash6'] = dataset['geohash11'].str[: -5]
    dataset['geohash5'] = dataset['geohash11'].str[: -6]

    dataset_gh10 = dataset.groupby(['geohash10']).agg(
        {'lat': 'mean', 'lon': 'mean', 'lon_3857': 'mean', 'lat_3857': 'mean',
         'count': 'sum'}).reset_index()
    dataset_gh9 = dataset.groupby(['geohash9']).agg(
        {'lat': 'mean', 'lon': 'mean', 'lon_3857': 'mean', 'lat_3857': 'mean',
         'count': 'sum'}).reset_index()
    dataset_gh8 = dataset.groupby(['geohash8']).agg(
        {'lat': 'mean', 'lon': 'mean', 'lon_3857': 'mean', 'lat_3857': 'mean',
         'count': 'sum'}).reset_index()
    dataset_gh7 = dataset.groupby(['geohash7']).agg(
        {'lat': 'mean', 'lon': 'mean', 'lon_3857': 'mean', 'lat_3857': 'mean',
         'count': 'sum'}).reset_index()
    dataset_gh6 = dataset.groupby(['geohash6']).agg(
        {'lat': 'mean', 'lon': 'mean', 'lon_3857': 'mean', 'lat_3857': 'mean',
         'count': 'sum'}).reset_index()
    # corresponding to zoom level 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4
    data_group = [dataset, dataset_gh10, dataset_gh9,
                  dataset_gh8, dataset_gh7, dataset_gh6]
    return data_group


def w_geo_parameters(xmin_w, ymin_w, xmax_w, ymax_w, mapbox_level, pixels_per_tile):
    # WORLD parameter calculation ################
    image_size_w = (2**mapbox_level*pixels_per_tile[0],
                    2**mapbox_level*pixels_per_tile[1])
    nx_w = image_size_w[0]
    ny_w = image_size_w[1]
    xres_w = (xmax_w - xmin_w) / float(nx_w)
    yres_w = (ymax_w - ymin_w) / float(ny_w)
    return nx_w, ny_w, xres_w, yres_w, mapbox_level


def geo_parameters(xmin, ymin, xmax, ymax, nx_w, ny_w, xres_w, yres_w):
    #  Initialize the Image Size
    image_size = (ceil(nx_w*(xmax-xmin)/(xmax_w-xmin_w)),
                  ceil(ny_w*(ymax-ymin)/(ymax_w-ymin_w)))
    nx = image_size[0]
    ny = image_size[1]
    print('image is %s * %s size. ' % (nx, ny))
    # set geotransform
    geotransform = (xmin, xres_w, 0, ymax, 0, -yres_w)
    print('image size is %s * %s' % (image_size[0], image_size[1]))
    return nx, ny, geotransform, image_size


def imageSizeofEachZoomLevel(mapbox_level):
    nx_w, ny_w, xres_w, yres_w, mapbox_level = w_geo_parameters(
        xmin_w, ymin_w, xmax_w, ymax_w, mapbox_level, pixels_per_tile)
    nx, ny, geotransform, image_size = geo_parameters(
        xmin, ymin, xmax, ymax, nx_w, ny_w, xres_w, yres_w)
    print(mapbox_level, image_size)
    return mapbox_level, image_size


def norm_color(dataset, image_size, xres_w, yres_w, opacity):
    # normalize data and generate color bands
    t1 = time.perf_counter()
    dataset['x'] = [round((x-xmin)/xres_w) for x in dataset['lon_3857']]
    dataset['y'] = [round((y-ymax)/-yres_w) for y in dataset['lat_3857']]

    # aggregate counts by same x and y
    dataset = dataset[['x', 'y', 'count']].groupby(
        by=['x', 'y']).sum().reset_index()

    dataset['count_HE'] = stats.rankdata(
        dataset['count'], "average")/len(dataset['count'])
    t2 = time.perf_counter()
    print(f'Data Normalization finished in {round(t2-t1, 2)} seconds')

    #  Create Each Channel
    r_pixels = np.transpose(np.zeros((image_size), dtype=np.uint8))
    g_pixels = np.transpose(np.zeros((image_size), dtype=np.uint8))
    b_pixels = np.transpose(np.zeros((image_size), dtype=np.uint8))
    a_pixels = np.transpose(np.zeros((image_size), dtype=np.uint8))

    # define color map
    cmap = cm.get_cmap('afmhot')

    # convert 0-1 to 0-255
    dataset['red'] = cmap(dataset['count_HE'])[:, 0]*255
    dataset['green'] = cmap(dataset['count_HE'])[:, 1]*255
    dataset['blue'] = cmap(dataset['count_HE'])[:, 2]*255
    dataset['opacity'] = cmap(dataset['count_HE'])[:, 3]*255*opacity

    for row in dataset.itertuples(index=False):
        try:
            r_pixels[row.y, row.x] = row.red
            g_pixels[row.y, row.x] = row.green
            b_pixels[row.y, row.x] = row.blue
            a_pixels[row.y, row.x] = row.opacity
        except:
            r_pixels[row.y-1, row.x-1] = row.red
            g_pixels[row.y-1, row.x-1] = row.green
            b_pixels[row.y-1, row.x-1] = row.blue
            a_pixels[row.y-1, row.x-1] = row.opacity

    t3 = time.perf_counter()
    print(f'Spent {round(t3-t2, 2)} seconds to generate RGB bands')
    return r_pixels, g_pixels, b_pixels, a_pixels


def generate_tif(r_pixels, g_pixels, b_pixels, a_pixels, mapbox_level, nx, ny, geotransform, measure, season, DOW, TOD):
    # create the 3-band raster file
    t1 = time.perf_counter()
    if not os.path.exists('tiff_%s' % version):
        os.makedirs('tiff_%s' % version)
    file_name = 'tiff_%s/%s_%s_%s_%s_%s.tif' % (
        version, measure, season, DOW, TOD, mapbox_level)
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        file_name, xsize=nx, ysize=ny, bands=4, eType=gdal.GDT_Byte, options=["TILED=YES", "BLOCKXSIZE=256", "BLOCKYSIZE=256"])

    dst_ds.SetGeoTransform(geotransform)    # specify coords
    srs = osr.SpatialReference()            # establish encoding
    # srs.ImportFromEPSG(4326)                # WGS84 lat/long
    srs.ImportFromEPSG(3857)                # WGS84 lat/long
    dst_ds.SetProjection(srs.ExportToWkt())  # export coords to file
    dst_ds.GetRasterBand(1).WriteArray(r_pixels)   # write r-band to the raster
    dst_ds.GetRasterBand(2).WriteArray(g_pixels)   # write g-band to the raster
    dst_ds.GetRasterBand(3).WriteArray(b_pixels)   # write b-band to the raster
    dst_ds.GetRasterBand(4).WriteArray(a_pixels)   # write a-band to the raster

    dst_ds.FlushCache()                     # write to disk
    dst_ds = None

    t2 = time.perf_counter()
    print(
        f'Spent {round(t2-t1, 2)} seconds to generate level {mapbox_level} TIFF')

    return file_name


def gen_png(file_name, mapbox_level, measure, season, DOW, TOD):
    # create folder if not exist
    t1 = time.perf_counter()
    output_dir = 'PNGs_%s/%s/%s/%s/%s' % (version, measure, season, DOW, TOD)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    gdal2tiles.generate_tiles(file_name, output_dir,
                              nb_processes=1, zoom=mapbox_level, s_srs="EPSG:3857", resampling='bilinear')
    t2 = time.perf_counter()
    print(f'{round(t2-t1, 2)} seconds to generate level {mapbox_level} PNGs')


def main_task(m, s, d, t, mapbox_level_list, pixels_per_tile, opacity, xmin_w, ymin_w, xmax_w, ymax_w, xmin, ymin, xmax, ymax, xmin_filter, ymin_filter, xmax_filter, ymax_filter):
    t1 = time.perf_counter()
    results = []
    data_group = GenDataGroup(m, s, d, t, xmin_filter,
                              ymin_filter, xmax_filter, ymax_filter)
    for mapbox_level in mapbox_level_list:
        if mapbox_level >= 16:
            dataset = data_group[0]
        elif mapbox_level >= 14:
            dataset = data_group[1]
        elif mapbox_level >= 12:
            dataset = data_group[2]
        elif mapbox_level >= 10:
            dataset = data_group[3]
        elif mapbox_level >= 7:
            dataset = data_group[4]
        elif mapbox_level >= 4:
            dataset = data_group[5]
        nx_w, ny_w, xres_w, yres_w, mapbox_level = w_geo_parameters(
            xmin_w, ymin_w, xmax_w, ymax_w, mapbox_level, pixels_per_tile)
        nx, ny, geotransform, image_size = geo_parameters(
            xmin, ymin, xmax, ymax, nx_w, ny_w, xres_w, yres_w)
        r_pixels, g_pixels, b_pixels, a_pixels = norm_color(
            dataset, image_size, xres_w, yres_w, opacity)
        file_name = generate_tif(
            r_pixels, g_pixels, b_pixels, a_pixels, mapbox_level, nx, ny, geotransform, m, s, d, t)
        gen_png(file_name, mapbox_level, m, s, d, t)
        results.append([file_name, mapbox_level, m, s, d, t])
    t2 = time.perf_counter()
    print(f'{round(t2-t1, 2)} seconds to complete ONE TASK! ')
    print("%s %s %s %s finished! " % (m, s, d, t))
    return results


result_list = []


if __name__ == "__main__":
    Measures = ['Vehicle Volume']
    # Measures = ['Pedestrian Bicycle Volume']
    Seasons = ['Spring']
    DOW = ['Weekday']
    TOD = ['AM']

    input_list = list(itertools.product(Measures, Seasons, DOW, TOD))
    print('In total %s tasks waiting...' % len(input_list))

    # Single task TEST
    main_task(*input_list[0], mapbox_level_list, pixels_per_tile, opacity, xmin_w, ymin_w,
              xmax_w, ymax_w, xmin, ymin, xmax, ymax, xmin_filter, ymin_filter, xmax_filter, ymax_filter)
