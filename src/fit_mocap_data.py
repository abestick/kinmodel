#!/usr/bin/env python
import pandas as pd
import numpy as np
import kinmodel
from scipy.optimize import minimize
from phasespace.load_mocap import find_homog_trans


def string_labelled_mocap_df(df):
    """

    :param pd.DataFrame df:
    :return:
    """
    return df.rename(columns={k: 'mocap_%d'%k for k in df.columns})


def get_kin_tree_base_markers(kin_tree):
    # Get base marker names
    base_markers = []
    base_joint = kin_tree.get_root_joint()
    for child in base_joint.children:
        if not hasattr(child, 'children'):
            # This is a feature
            base_markers.append(child.name)

    # Get mapping of marker names -> marker idxs
    marker_indices = {}
    for feature_name in kin_tree.get_features():
        marker_indices[feature_name] = int(feature_name.split('_')[1])

    # Get the desired coordinates of each base marker
    base_frame_points = np.zeros((len(base_markers), 3, 1))
    all_features = kin_tree.get_features()
    for i, marker in enumerate(base_markers):
        base_frame_points[i, :, 0] = all_features[marker].q()

    base_idxs = [marker_indices[name] for name in base_markers]
    return base_idxs, base_frame_points.squeeze()


def get_transforms(df, kin_tree, name='T_bw', inv=False):
    """
    
    :param pd.DataFrame df: 
    :param kinmodel.KinematicTree kin_tree:
    :param str name:
    :return:
    """
    base_idx, base_points_base = get_kin_tree_base_markers(kin_tree)
    last_transform = [np.eye(4)]
    skipped = []

    def get_transform(row):
        try:
            base_points_world = np.vstack(row[base_idx])
            last_transform[0] = find_homog_trans(base_points_world, base_points_base)[0] if not inv else \
                find_homog_trans(base_points_base, base_points_world)[0]
        except np.linalg.LinAlgError:
            skipped.append(row.name)

        return pd.Series([last_transform[0]])

    df[name] = df.apply(get_transform, axis=1)
    print(skipped)
    return df


def transform_row(row, name='T_bw'):
    """

    :param pd.Series row:
    :param str name:
    :return:
    """
    transform = row[name]

    def transform_point(point):
        return transform.dot(np.append(point, [1])) if len(point) == 3 else point

    return row.apply(transform_point)


def transform_points(df):
    """

    :param pd.DataFrame df:
    :return:
    """
    return df.apply(transform_row, axis=1)


def kin_tree_friendly_df(df):
    """

    :param pd.DataFrame df:
    :return:
    """
    if 'T_bw' in df:
        df.pop('T_bw')

    df = string_labelled_mocap_df(df)
    return df.applymap(kinmodel.Point)


class KinTreeObjFunc(object):

    def __init__(self, kin_tree):
        """

        :param kinmodel.KinematicTree kin_tree:
        """
        self._kin_tree = kin_tree.to_1d_chain()
        self.joint_order = self._kin_tree.get_joints().keys()
        self.joint_order.remove('base')
        self.zeros = np.zeros(len(self.joint_order))
        self.bounds = zip(self.zeros - np.pi, self.zeros + np.pi)
        self.features = set(self._kin_tree.get_features())
        self.features

    def config_dict(self, theta):
        return {joint_name: joint_value for joint_name, joint_value in zip(self.joint_order, theta)}

    def observation_dict(self, row):
        obs = dict(row)
        assert self.features.issubset(obs), 'Columns have some features that are not in the KinematicTree: ' + \
                                                 str(set(obs) - self.features)
        return {k: obs[k] for k in self.features.intersection(obs) if not any(np.isnan(obs[k]))}

    def error(self, theta, row):
        configs = self.config_dict(theta)
        observations = self.observation_dict(row)
        return self._kin_tree.compute_error(configs, observations)

    def get_configs(self, row):

        result = minimize(self.error, x0=self.zeros, args=(row,), method='L-BFGS-B', bounds=self.bounds)
        cols = {'OptimizationResult':result}
        cols.update(self.config_dict(result.x))
        cols['Error'] = result.fun
        cols['Success'] = '%r: %s' % (result.success, result.message)
        return pd.Series(cols)


def get_joints(df, kin_tree):
    print('getting transforms')
    df = get_transforms(df, kin_tree)
    print('transforming points')
    df = transform_points(df)
    df = kin_tree_friendly_df(df)
    opt = KinTreeObjFunc(kin_tree)
    print('getting configs')
    return df.apply(opt.get_configs, axis=1)


dfs = [
    '_andrea_0.bag.df',
    '_andrea_1.bag.df',
    '_andrea_2.bag.df',
    '_andrea_3.bag.df',
    '_rob_0.bag.df',
    '_rob_1.bag.df',
    '_rob_2.bag.df',
    '_rob_3.bag.df',
    '_sarah_0.bag.df',
    '_sarah_good_0.bag.df',
    '_sarah_good_1.bag.df',
    '_sarah_good_2.bag.df',
    '_2017-10-31-17-06-02.bag.df',
    '_2017-10-31-17-12-24.bag.df',
    '_2017-10-31-17-30-42.bag.df'
]

jsons = [
    '/home/pedge/experiment/andrea/andrea.json',
    '/home/pedge/experiment/andrea/andrea.json',
    '/home/pedge/experiment/andrea/andrea.json',
    '/home/pedge/experiment/andrea/andrea.json',
    '/home/pedge/experiment/rob/rob.json',
    '/home/pedge/experiment/rob/rob.json',
    '/home/pedge/experiment/rob/rob.json',
    '/home/pedge/experiment/sarah/sarah.json',
    '/home/pedge/experiment/sarah/sarah.json',
    '/home/pedge/experiment/sarah/sarah.json',
    '/home/pedge/experiment/sarah/sarah.json',
    '/home/pedge/experiment/rob/rob.json',
    '/home/pedge/experiment/sarah/sarah.json',
    '/home/pedge/experiment/andrea/andrea.json',
]

obj_json = '/home/pedge/experiment/object/object.json'


SAVE_DIR = '/home/pedge/experiment/pd/'


if __name__ == '__main__':
    for df_name, json in zip(dfs, jsons):
        df = pd.read_pickle(SAVE_DIR + df_name)
        print('Starting: %s' % df_name)
        # try:
        #     human_joint_df = get_joints(df, kinmodel.KinematicTree(json_filename=json))
        #     object_joint_df = get_joints(df, kinmodel.KinematicTree(json_filename=obj_json))
        #     human_joint_df.to_pickle(SAVE_DIR + 'joints/' + df_name)
        #     object_joint_df.to_pickle(SAVE_DIR + 'joints/' + 'obj' + df_name)
        #     print('Done: %s' % df_name)
        # except Exception as e:
        #     print('Failed: %s' % df_name)
        #     print(e.message)

        object_joint_df = get_joints(df, kinmodel.KinematicTree(json_filename=obj_json))
        human_joint_df = get_joints(df, kinmodel.KinematicTree(json_filename=json))
        human_joint_df.to_pickle(SAVE_DIR + 'joints/' + df_name)
        object_joint_df.to_pickle(SAVE_DIR + 'joints/' + 'obj' + df_name)
        print('Done: %s' % df_name)
