#!/usr/bin/env python
import rosbag
import pandas as pd
import pickle
import numpy as np


SAVE_DIR = '/home/pedge/experiment/pd/'


def mocap_to_row(msg):
    return [[np.array((p.x, p.y, p.z)) for p in msg.points]]


def bag_to_data_frame(name, save_dir=SAVE_DIR):
    bag = rosbag.Bag('/home/pedge/experiment/mixed_results/%s.bag' % name)

    df = pd.DataFrame()
    for topic, msg, t in bag.read_messages(topics='/mocap_point_cloud'):
        df = df.append(pd.DataFrame(mocap_to_row(msg), index=[t.to_sec()]))
    bag.close()

    dfs = {'':df}
    df_with_time = df.copy()
    df_with_time['dt'] = df.index
    dfs['displacement'] = df_with_time.diff()
    dfs['velocity'] = dfs['displacement'].div(dfs['displacement'].pop('dt'), axis=0)
    dfs['speed'] = dfs['velocity'].applymap(np.linalg.norm)
    for prefix, data_frame in dfs.items():
        data_frame.to_pickle(save_dir + prefix + '_' + name + '.df')

    return dfs


def snip(name, save_dir=SAVE_DIR):
    bag = rosbag.Bag('/home/pedge/experiment/mixed_results/%s.bag' % name)

    ends = []
    starts = []
    for topic, msg, t in bag.read_messages(topics=['/iiwa/state/DestinationReached', '/iiwa/command/CartesianPoseLin']):
        if topic == '/iiwa/state/DestinationReached':
            starts.append(t.to_sec())
        else:
            ends.append(t.to_sec())

    bag.close()
    ends.append(np.inf)

    runs = zip(starts[-16:], ends[-16:])
    data = {'runs': runs, 'starts': starts, 'ends': ends}

    pickle.dump(data, open(save_dir + name + '_snip.df', 'wb'))


names = [
    'andrea_0',
    'andrea_1',
    'andrea_2',
    'andrea_3',
    'rob_0',
    'rob_1',
    'rob_2',
    'rob_3',
    'sarah_0',
    'sarah_good_0',
    'sarah_good_1',
    'sarah_good_2',
]

names2 = [
    '2017-10-31-17-06-02',
    '2017-10-31-17-12-24',
    '2017-10-31-17-30-42'
]

save_dir2 = SAVE_DIR + '2'

for name in names:
    print('Converting ' + name)
    snip(name)


for name in names2:
    print('Converting ' + name)
    snip(name)
