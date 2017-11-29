#!/usr/bin/env python
import pandas as pd
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Header
from kinmodel import KinematicTree
from fit_mocap_data import get_transforms
import tf


def get_point_cloud(row, frame_id):
    stamp = rospy.Time(row.name)
    points = [Point32(*p) for p in row]
    return PointCloud(header=Header(frame_id=frame_id, stamp=stamp), points=points)


def get_point_clouds(df, frame_id):
    return df.apply(get_point_cloud, axis=1, args=(frame_id,))


def joints_to_points(df, kin_tree):
    """

    :param pd.DataFrame df:
    :param KinematicTree kin_tree:
    :return:
    """
    df = df[list(set(df.columns).intersection(kin_tree.get_joints()))]

    def row_to_points(row):
        config = dict(row)
        kin_tree.set_config(config)
        features = kin_tree.observe_features()
        points = {f: features[f].q() for f in features if 'mocap' in f}
        return pd.Series(points)

    return df.apply(row_to_points, axis=1)


def joints_to_point_clouds(df, kin_tree, frame_id):
    df = joints_to_points(df, kin_tree)
    return get_point_clouds(df, frame_id)


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


rospy.init_node('check_fit')
point_pub0 = rospy.Publisher('mocap_point_cloud', PointCloud, queue_size=100)
point_pub1 = rospy.Publisher('mocap_point_cloud1', PointCloud, queue_size=100)
point_pub2 = rospy.Publisher('mocap_point_cloud2', PointCloud, queue_size=100)
tf_pub = tf.TransformBroadcaster()


def pub_transform(transform, child, parent, t):
    tf_pub.sendTransform(transform[0:3, 3],
                              tf.transformations.quaternion_from_matrix(transform),
                              rospy.Time(t), child, parent)


for df_name, json in zip(dfs, jsons):
    orig_points = pd.read_pickle(SAVE_DIR + df_name)
    joints = pd.read_pickle(SAVE_DIR + 'joints/' + df_name)
    obj_joints = pd.read_pickle(SAVE_DIR + 'joints/obj' + df_name)
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

    obj = KinematicTree(json_filename=obj_json)
    human = KinematicTree(json_filename=json).to_1d_chain()

    new_points_human = joints_to_point_clouds(joints, human, 'human')
    new_points_obj = joints_to_point_clouds(obj_joints, obj, 'object')
    orig_points = get_transforms(orig_points, human, 'human', inv=True)
    orig_points = get_transforms(orig_points, obj, 'object', inv=True)
    human_trans = orig_points.pop('human')
    obj_trans = orig_points.pop('object')
    orig_points = get_point_clouds(orig_points, 'world')

    rate = rospy.Rate(100)

    raw_input('ready?')
    for pc0, pc1, pc2, th, to, t in zip(orig_points, new_points_human, new_points_obj,
                                     human_trans, obj_trans, orig_points.index):
        pub_transform(th, 'human', 'world', t)
        pub_transform(to, 'object', 'world', t)
        point_pub0.publish(pc0)
        point_pub1.publish(pc1)
        point_pub2.publish(pc2)
        rate.sleep()