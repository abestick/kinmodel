#!/usr/bin/env python
import pandas as pd
import pickle

dfs = [
    # '_andrea_0.bag.df',
    # '_andrea_1.bag.df',
    # '_andrea_2.bag.df',
    # '_andrea_3.bag.df',
    # '_rob_0.bag.df',
    # '_rob_1.bag.df',
    # '_rob_2.bag.df',
    '_rob_3.bag.df',
    '_sarah_0.bag.df',
    '_sarah_good_0.bag.df',
    '_sarah_good_1.bag.df',
    '_sarah_good_2.bag.df',
    # '_2017-10-31-17-06-02.bag.df',
    # '_2017-10-31-17-12-24.bag.df'
    # '_2017-10-31-17-30-42.bag.df'
]


names = [
    # 'participant_0_0',
    # 'participant_0_1',
    # 'participant_0_2',
    # 'participant_0_3',
    # 'participant_1_0',
    # 'participant_1_1',
    # 'participant_1_2',
    'participant_1_3',
    'participant_2_0',
    'participant_2_1',
    'participant_2_2',
    'participant_2_3',
    # 'participant_0_4',
    # 'participant_1_4',
    # 'participant_2_4'
]


SAVE_DIR = '/home/pedge/experiment/pd/'
human_order = ['shoulder_0', 'shoulder_1', 'shoulder_2', 'elbow']
obj_order = ['joint_0', 'joint_1', 'joint_2', 'joint_3']
order = human_order + obj_order

assert len(names) == len(dfs)
for df_name, name in zip(dfs, names):
    joints = pd.read_pickle(SAVE_DIR + 'joints2/' + df_name)
    obj_joints = pd.read_pickle(SAVE_DIR + 'joints2/obj' + df_name)
    print('Starting: %s' % df_name)
    both = joints[human_order].join(obj_joints[obj_order])[order]
    assert isinstance(both, pd.DataFrame)
    snip_data = pickle.load(open(SAVE_DIR + df_name[1:-7]+'_snip.df', 'rb'))
    runs = snip_data['runs']
    writer = pd.ExcelWriter(SAVE_DIR + 'export2/' + name + '_snipped.xlsx')
    for i, (s, e) in enumerate(runs):
        mask = both.index.to_series().between(s, e)
        df = both.loc[mask]

        df.to_excel(writer, sheet_name='run_%d' % i)

    writer.save()