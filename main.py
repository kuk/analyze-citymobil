

import os
import re
from itertools import islice, groupby, chain
from collections import Counter, namedtuple, defaultdict
from subprocess import check_output, check_call
from datetime import datetime, time

from random import seed as random_seed
from random import sample, randint, random

from tqdm import tqdm_notebook as log_progress

import numpy as np
import pandas as pd
import xarray as xr
from scipy.ndimage.filters import convolve

from matplotlib import pyplot as plt
import seaborn as sns

import datashader as ds
import datashader.transfer_functions as tf
from datashader.colors import inferno, viridis, Hot, Greys9
from datashader.bokeh_ext import InteractiveImage
from datashader.utils import export_image

from bokeh import plotting as bk
from bokeh.tile_providers import WMTSTileSource as TileSource

import imageio


LIGHT_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'
)
DARK_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
)
# 'https://vec01.maps.yandex.net/tiles?l=map&v=17.06.26-1&x={x}&y={y}&z={z}&scale=2&lang=ru_RU'


import pyproj


DATA_DIR = 'data'
CITYMOBIL_DIR = os.path.join(DATA_DIR, 'citymobil')
RAW_CITYMOBIL_DIR = os.path.join(CITYMOBIL_DIR, 'raw')
RAW_CITYMOBIL1 = os.path.join(RAW_CITYMOBIL_DIR, 'data_export.sql')
RAW_CITYMOBIL2 = os.path.join(RAW_CITYMOBIL_DIR, 'data_export_part_2.sql')
RAW_CITYMOBIL3 = os.path.join(RAW_CITYMOBIL_DIR, 'data_export_part_3.sql')
CITYMOBIL = os.path.join(CITYMOBIL_DIR, 'data')

TABLES_DIR = os.path.join(DATA_DIR, 'tables')
SPLIT = 'split'
TMP_TABLES_DIR = os.path.join(TABLES_DIR, 'tmp')
MAP_CITYMOBIL_SQL = os.path.join(TABLES_DIR, 'map_citymobil_sql.tsv')

TRACK_POINTS = os.path.join(DATA_DIR, 'track_points.csv')
ORDERS_FEATURES = os.path.join(DATA_DIR, 'orders.tsv')

ANIMATION_DIR = 'i'
FRAMES_DIR = os.path.join(ANIMATION_DIR, 'frames')
ANIMATION = os.path.join(ANIMATION_DIR, 'animation.gif')

RECORD_PATTERN = re.compile(r"""\(
    \d+,
    '(?P<order_id>[a-z0-9]{32})',
    '(?P<client_id>[a-z0-9]{32})',
    '(?P<driver_id>[a-z0-9]{32})',
    (?P<lat>\d+\.\d+),
    (?P<lon>\d+\.\d+),
    (?P<time>\d+)
\)""", re.X)

WIDTH = 700
BLACK = 'black'
COLORS24 = [
    '#ff0000','#ff3f00','#ff7f00','#ffbf00','#ffff00','#bfff00','#7fff00','#3fff00',
    '#00ff00','#00ff3f','#00ff7f','#00ffbf','#00ffff','#00bfff','#007fff','#003fff',
    '#0000ff','#3f00ff','#7f00ff','#bf00ff','#ff00ff','#ff00bf','#ff007f','#ff003f'
]


CitymobilSqlRecord = namedtuple(
    'CitymobilSqlRecord',
    ['order', 'client', 'driver',
     'latitude', 'longitude', 'timestamp']
)
TrackRecord = namedtuple(
    'TrackRecord',
    ['timestamp', 'latitude', 'longitude']
)
CitymobilRecord = namedtuple(
    'CitymobilRecord',
    ['order', 'client', 'driver', 'track']
)
Point = namedtuple(
    'Point',
    ['latitude', 'longitude']
)
BoundingBox = namedtuple(
    'BoundingBox',
    ['lower_left', 'upper_right']
)
OrderFeatures = namedtuple(
    'OrderFeatures',
    ['id', 'client', 'driver',
     'timestamp', 'pickup', 'dropoff',
     'duration']
)
Frame = namedtuple(
    'Frame',
    ['timestamp', 'image']
)


MOSCOW_BOX = BoundingBox(
    Point(55.555959, 37.252433),
    Point(55.929357, 37.952812)
)
TTK_BOX = BoundingBox(
    Point(55.696607, 37.524263),
    Point(55.796919, 37.730943)
)
MOSCOW_BOX2 = BoundingBox(
    Point(55.387862, 37.189180),
    Point(56.000615, 37.980195)
)
MOSCOW_BOX3 = BoundingBox(
    Point(55.388223, 36.544598),
    Point(56.036352, 38.785809)
)
TTK_BOX2 = BoundingBox(
    Point(55.702747, 37.508507),
    Point(55.797628, 37.706604)
)
SADOVOE_BOX = BoundingBox(
    Point(55.727619, 37.578508),
    Point(55.773897, 37.663995)
)
SAVEL_BOX = BoundingBox(
    Point(55.774622, 37.581177),
    Point(55.778057, 37.587335)
)


KERNEL = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
])
KERNEL_3D = np.array([
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]],
    [[0, 1, 0],
     [0, 1, 0],
     [0, 1, 0]],
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
])


def jobs_manager():
    from IPython.lib.backgroundjobs import BackgroundJobManager
    from IPython.core.magic import register_line_magic
    from IPython import get_ipython

    jobs = BackgroundJobManager()

    @register_line_magic
    def job(line):
        ip = get_ipython()
        jobs.new(line, ip.user_global_ns)

    return jobs


def kill_thread(thread):
    import ctypes

    id = thread.ident
    code = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_long(id),
        ctypes.py_object(SystemError)
    )
    if code == 0:
        raise ValueError('invalid thread id')
    elif code != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(id),
            ctypes.c_long(0)
        )
        raise SystemError('PyThreadState_SetAsyncExc failed')


def read_chunks(path, size):
    with open(path) as file:
        while True:
            chunk = file.read(size)
            if chunk == '':
                break
            yield chunk


def parse_citymobil_sql(chunks):
    previous = ''
    for chunk in chunks:
        chunk = previous + chunk
        stop = 0
        for match in RECORD_PATTERN.finditer(chunk):
            items = match.groupdict()
            yield CitymobilSqlRecord(**items)
            stop = match.end()
        previous = chunk[stop:]


def map_citymobil_sql(records):
    for _ in records:
        yield '\t'.join([
            _.order, _.client, _.driver,
            _.timestamp, _.latitude, _.longitude
        ])

                
def dump_lines(lines, path):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line + '\n')


def get_lines_count(path):
    output = check_output(['wc', '-l', path])
    lines, _ = output.split(None, 1)
    return int(lines)


get_table_size = get_lines_count


def sort_table(table, by, chunks=20):
    if not isinstance(by, (list, tuple)):
        by = (by,)
    size = get_table_size(table) // chunks
    tmp = os.path.join(TMP_TABLES_DIR, SPLIT)
    try:
        print('Split in {} chunks, prefix: {}'.format(chunks, tmp))
        check_call(
            ['split', '-l', str(size), table, tmp],
            env={'LC_ALL': 'C'}
        )
        ks = ['-k{0},{0}'.format(_ + 1) for _ in by]
        tmps = [os.path.join(TMP_TABLES_DIR, _)
                for _ in os.listdir(TMP_TABLES_DIR)]
        for index, chunk in enumerate(tmps):
            print('Sort {}/{}: {}'.format(index + 1, chunks, chunk))
            check_call(
                ['sort', '-t', '\t'] + ks + ['-o', chunk, chunk],
                env={'LC_ALL': 'C'}
            )
        print('Merge into', table)
        check_call(
            ['sort', '-t', '\t'] + ks + ['-m'] + tmps + ['-o', table],
            env={'LC_ALL': 'C'}
        )
    finally:
        for name in os.listdir(TMP_TABLES_DIR):
            path = os.path.join(TMP_TABLES_DIR, name)
            os.remove(path)


def load_lines(path):
    with open(path) as file:
        for line in file:
            yield line.rstrip('\n')

            
def read_map_citymobil_sql():
    for line in load_lines(MAP_CITYMOBIL_SQL):
        order, client, driver, timestamp, latitude, longitude = line.split()
        yield CitymobilSqlRecord(order, client, driver, latitude, longitude, timestamp)
        
        
def format_citymobil_sql_groups(records):
    for (order, client, driver), group in groupby(
        records, key=lambda _: (_.order, _.client, _.driver)
    ):
        yield '\t'.join([order, client, driver])
        for _ in group:
            yield '\t' + '\t'.join([_.timestamp, _.latitude, _.longitude])


def load_citymobil():
    previous = None
    track = []
    for line in load_lines(CITYMOBIL):
        if line.startswith('\t'):
            timestamp, latitude, longitude = line.split()
            timestamp = int(timestamp)
            latitude = float(latitude)
            longitude = float(longitude)
            track.append(TrackRecord(
                timestamp,
                latitude,
                longitude
            ))
        else:
            if previous:
                # WRNING NOTE there seems to be a bug in column names
                # len(orders), len(clients), len(drivers)
                # (705904, 25233, 2078751)
                # len(data) == 2078751
                # Assume order is client, driver, order
                # not "order, client, driver"
                client, driver, order = previous
                yield CitymobilRecord(order, client, driver, track)
                track = []
            previous = line.split()
    order, client, driver = previous
    yield CitymobilRecord(order, client, driver, track) 


def is_inside(point, box):
    return (
        box.lower_left.latitude < point.latitude < box.upper_right.latitude
        and box.lower_left.longitude < point.longitude < box.upper_right.longitude
    )


WORLD = pyproj.Proj(init='epsg:4326')
MERCATOR = pyproj.Proj(init='epsg:3857')


def convert_point(point, source=WORLD, target=MERCATOR):
    try:
        longitude, latitude = pyproj.transform(
            source, target,
            point.longitude, point.latitude
        )
    except RuntimeError:
        raise ValueError(point)
    return Point(latitude, longitude)


def convert_box(box):
    return BoundingBox(
        convert_point(box.lower_left),
        convert_point(box.upper_right)
    )


def convert_order_features_table(table):
    point = Point(
        table.pickup_latitude.values,
        table.pickup_longitude.values
    )
    point = convert_point(point)
    table['pickup_latitude'] = point.latitude
    table['pickup_longitude'] = point.longitude

    point = Point(
        table.dropoff_latitude.values,
        table.dropoff_longitude.values
    )
    point = convert_point(point)
    table['dropoff_latitude'] = point.latitude
    table['dropoff_longitude'] = point.longitude
    return table


def format_tsv(*items):
    return '\t'.join(str(_) for _ in items)


def get_id_mapping():
    from itertools import count

    return defaultdict(count().__next__)


def format_track_points(records):
    orders = get_id_mapping()
    yield format_tsv('order', 'timestamp', 'latitude', 'longitude')
    for record in records:
        order = orders[record.order]
        for point in record.track:
            timestamp = point.timestamp
            yield format_tsv(
                order, timestamp,
                point.latitude, point.longitude
            )


def load_track_points():
    table = pd.read_csv(TRACK_POINTS, sep='\t')
    table['timestamp'] = pd.to_datetime(table.timestamp + 3 * 3600, unit='s')
    return table


def compute_order_features(records):
    orders = get_id_mapping()
    clients = get_id_mapping()
    drivers = get_id_mapping()
    for record in records:
        order = orders[record.order]
        client = clients[record.client]
        driver = drivers[record.driver]
        track = record.track
        pickup = track[0]
        dropoff = track[-1]
        timestamp = pickup.timestamp
        duration = dropoff.timestamp - timestamp
        yield OrderFeatures(
            order, client, driver,
            timestamp, pickup, dropoff,
            duration
        )


def format_order_features(records):
    yield format_tsv(
        'id', 'client', 'driver',
        'timestamp', 'pickup_latitude', 'pickup_longitude',
        'dropoff_latitude', 'dropoff_longitude',
        'duration'
    )
    for record in records:
        pickup = record.pickup
        dropoff = record.dropoff
        yield format_tsv(
            record.id, record.client, record.driver,
            record.timestamp, pickup.latitude, pickup.longitude,
            dropoff.latitude, dropoff.longitude,
            record.duration
        )


def load_order_features():
    lines = load_lines(ORDERS_FEATURES)
    next(lines)
    for line in lines:
        (id, client, driver,
         timestamp, pickup_latitude, pickup_longitude,
         dropoff_latitude, dropoff_longitude,
         duration) = line.split()
        yield OrderFeatures(
            int(id), int(client), int(driver),
            int(timestamp),
            Point(float(pickup_latitude), float(pickup_longitude)),
            Point(float(dropoff_latitude), float(dropoff_longitude)),
            int(duration)
        )


def make_order_features_table(records):
    data = []
    for _ in records:
        data.append([
            _.id, _.client, _.driver,
            _.timestamp,
            _.pickup.latitude, _.pickup.longitude,
            _.dropoff.latitude, _.dropoff.longitude,
            _.duration
        ])
    table = pd.DataFrame(
        data,
        columns=[
            'id', 'client', 'driver',
            'timestamp', 'pickup_latitude', 'pickup_longitude',
            'dropoff_latitude', 'dropoff_longitude',
            'duration'
        ]
    )
    table = table.set_index('id')
    # Need to use something like .dt.tz_localize but just add 3 hours
    table['timestamp'] = pd.to_datetime(table.timestamp + 3 * 3600, unit='s')
    return table


def get_canvas(box, width=WIDTH):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    height = int(width / (x_max - x_min) * (y_max - y_min))
    return ds.Canvas(
        plot_width=width, plot_height=height,
        x_range=x_range, y_range=y_range
    )


def get_figure(box, width=WIDTH):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    height = int(width / (x_max - x_min) * (y_max - y_min))
    
    figure = bk.figure(
        tools='pan, wheel_zoom',
        active_scroll='wheel_zoom',
        plot_width=width, plot_height=height,
        x_range=x_range, y_range=y_range,
    )
    figure.axis.visible = False
    return figure


def show_orders_by_time(records):
    counts = Counter()
    for record in records:
        timestamp = datetime.fromtimestamp(record.timestamp)
        if 2015 <= timestamp.year <= 2017:
            counts[timestamp] += 1
    table = pd.Series(counts)
    table = table.resample('d').sum()
    table.plot()


def show_orders_by_week(records):
    counts = Counter()
    for record in records:
        timestamp = datetime.fromtimestamp(record.timestamp)
        timestamp = datetime(
            2017, 1, timestamp.weekday() + 1,
            timestamp.hour, timestamp.minute, timestamp.second
        )
        counts[timestamp] += 1
    table = pd.Series(counts)
    table = table.resample('600s').sum()
    table.plot()


def show_orders_by_weekday(records):
    counts = Counter()
    for record in records:
        timestamp = datetime.fromtimestamp(record.timestamp)
        if timestamp.weekday() + 1 < 5:
            timestamp = datetime(
                2017, 1, 1,
                timestamp.hour, timestamp.minute, timestamp.second
            )
            counts[timestamp] += 1
    table = pd.Series(counts)
    table = table.resample('3600s').sum()
    table.plot()


def get_map(box, tiles, width=WIDTH):
    figure = get_figure(box, width=width)
    figure.add_tile(tiles)
    return figure


def decorate_callback(callback):
    def function(x_range, y_range, width, height, **kwargs):
        canvas = ds.Canvas(
            plot_width=width, plot_height=height,
            x_range=x_range, y_range=y_range
        )
        return callback(canvas, **kwargs)
    return function


def overlay_image(map, callback, **kwargs):
    callback = decorate_callback(callback)
    return InteractiveImage(map, callback, **kwargs)


def convert_track_points(table):
    point = Point(
        table.latitude.values,
        table.longitude.values
    )
    point = convert_point(point)
    table['latitude'] = point.latitude
    table['longitude'] = point.longitude
    return table


def convolve_aggregate(aggregate, kernel=KERNEL):
    coords = aggregate.coords
    aggregate = convolve(aggregate, kernel)
    return xr.DataArray(aggregate, coords)


def draw_time(image, timestamp):
    from PIL import ImageFont, ImageDraw
    from matplotlib import font_manager
    
    width, height = image.size
    draw = ImageDraw.Draw(image)
    path = font_manager.findfont('sans')
    font = ImageFont.truetype(path, 50)
    text = '{hour:02d}:{minute:02d}'.format(
        hour=timestamp.hour,
        minute=0
    )
    draw.text((width - 160, height - 60), text, (255, 255, 255), font=font)
    return image


def make_animation_frames(table):
    box = BoundingBox(
        Point(55.616249, 37.342988),
        Point(55.840990, 37.855226)
    )
    canvas = get_canvas(box, width=1000)
    step = 30
    for hour in range(24):
        for period in range(60 // step):
            minute = period * step
            timestamp = time(hour, minute)
            
            view = table[(table.hour == hour) & (table.minute // step == period)]

            aggregate = canvas.points(
                view,
                'pickup_longitude', 'pickup_latitude',
                ds.count()
            )
            aggregate = convolve_aggregate(aggregate)
            image = tf.shade(aggregate, cmap=viridis, how='eq_hist')
            image = tf.set_background(image, 'black')
            image = image.to_pil()
            # image = draw_time(image, timestamp)

            yield Frame(timestamp, image)


def get_frame_path(frame):
    timestamp = frame.timestamp
    filename = '{hour}_{minute}.png'.format(
        hour=timestamp.hour,
        minute=timestamp.minute
    )
    return os.path.join(FRAMES_DIR, filename)


def save_frame(frame):
    path = get_frame_path(frame)
    frame.image.save(path)

    
def cache_frames(frames):
    for frame in frames:
        save_frame(frame)
        yield frame


def build_animation(frames):
    images = []
    for frame in frames:
        path = get_frame_path(frame)
        image = imageio.imread(path)
        images.append(image)
    imageio.mimsave(ANIMATION, images)
