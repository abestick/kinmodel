#!/usr/bin/env python
import pandas as pd
import rospy
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32, Transform, Pose, Quaternion, Point
from std_msgs.msg import Header
from kinmodel import KinematicTree, Transform
from kinmodel.track_mocap import KinematicTreeExternalFrameTracker
from fit_mocap_data import get_base_transforms
import tf
import pickle
import os.path


def attach_frame(kin_tree, frame_name, joint, flip=False):
    ktt = KinematicTreeExternalFrameTracker(kin_tree)
    ktt.attach_shared_frame(joint, frame_name, flip=flip)


def get_point_cloud(row, frame_id):
    stamp = rospy.Time(row.name)
    points = [Point32(*p) for p in row]
    return PointCloud(header=Header(frame_id=frame_id, stamp=stamp), points=points)


def get_pose(row, frame_id):
    stamp = rospy.Time(row.name)
    poses = [Pose(position=Point(*p[:3]), orientation=Quaternion(*p[3:])) for p in row]
    return PointCloud(header=Header(frame_id=frame_id, stamp=stamp), points=points)


def get_point_clouds(df, frame_id):
    return df.apply(get_point_cloud, axis=1, args=(frame_id,))


def get_poses(df, frame_id):
    return df.apply(get_pose, axis=1, args=(frame_id,))


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


def joints_to_transforms(df, kin_tree, transforms):
    """

    :param pd.DataFrame df:
    :param KinematicTree kin_tree:
    :return:
    """
    df = df[list(set(df.columns).intersection(kin_tree.get_joints()))]

    def row_to_transforms(row):
        config = dict(row)
        kin_tree.set_config(config)
        features = kin_tree.observe_features()
        transform = {f: features[f].homog() for f in features if f in transforms}
        return pd.Series(transform)

    return df.apply(row_to_transforms, axis=1)


def joints_to_point_clouds(joints_df, kin_tree, frame_id):
    joints_df = joints_to_points(joints_df, kin_tree)
    return get_point_clouds(joints_df, frame_id)


model_map = {
    'participant_0' : ('andrea', True),
    'participant_1' : ('rob', False),
    'participant_2' : ('sarah', False)
}

obj_json = '/home/pedge/experiment/object/object.json'


SAVE_DIR = '/home/pedge/experiment/correct/'


rospy.init_node('check_fit')
point_pub0 = rospy.Publisher('mocap_point_cloud', PointCloud, queue_size=100)
point_pub1 = rospy.Publisher('mocap_point_cloud1', PointCloud, queue_size=100)
point_pub2 = rospy.Publisher('mocap_point_cloud2', PointCloud, queue_size=100)
tf_pub = tf.TransformBroadcaster()


def pub_transform(transform, child, parent, t):
    tf_pub.sendTransform(transform[0:3, 3],
                              tf.transformations.quaternion_from_matrix(transform),
                              rospy.Time(t), child, parent)


processed = {}

if False and os.path.isfile('processed.pkl'):
    processed = pickle.load(open('processed.pkl', 'rb'))

else:

    for participant, (json, flip) in model_map.items():
        processed[participant] = []

        for run in range(1, 4):
            orig_points = pd.read_pickle(SAVE_DIR + 'points/' + '%s_%d.df' % (participant, run))
            joints = pd.read_pickle(SAVE_DIR + 'joints/' + '%s_%d.df' % (participant, run))
            obj_joints = pd.read_pickle(SAVE_DIR + 'joints/' + 'obj_%s_%d.df' % (participant, run))
            print('Processing: %s_%d.df' % (participant, run))

            obj = KinematicTree(json_filename=obj_json)
            human = KinematicTree(json_filename='/home/pedge/experiment/%s/%s.json' % (json, json)).to_1d_chain()

            attach_frame(obj, 'grip_obj', 'joint_3')
            attach_frame(human, 'grip_human', 'elbow', flip)

            new_points_human = joints_to_point_clouds(joints, human, 'human')
            new_points_obj = joints_to_point_clouds(obj_joints, obj, 'object')
            orig_points = get_base_transforms(orig_points, human, 'human', inv=True)
            orig_points = get_base_transforms(orig_points, obj, 'object', inv=True)
            human_trans = orig_points.pop('human')
            obj_trans = orig_points.pop('object')
            orig_points = get_point_clouds(orig_points, 'world')
            human_grip = joints_to_transforms(joints, human, ['grip_human'])['grip_human']
            obj_grip = joints_to_transforms(obj_joints, obj, ['grip_obj'])['grip_obj']

            processed[participant].append((orig_points, new_points_human, new_points_obj,
                                             human_trans, obj_trans, human_grip, obj_grip, orig_points.index))


    # pickle.dump(processed, open('processed.pkl', 'wb'))


for participant in model_map.keys():
    for i, run in enumerate(processed[participant]):
        orig_points, new_points_human, new_points_obj, human_trans, \
        obj_trans, human_grip, obj_grip, orig_points.index = run

        rate = rospy.Rate(50)

        raw_input('ready?')
        print('Starting: %s_%d.df' % (participant, i+1))

        for pc0, pc1, pc2, th, to, hg, og, t in zip(orig_points, new_points_human, new_points_obj,
                                         human_trans, obj_trans, human_grip, obj_grip, orig_points.index):
            pub_transform(th, 'human', 'world', t)
            pub_transform(to, 'object', 'world', t)
            pub_transform(hg, 'grip_human', 'human', t)
            pub_transform(og, 'grip_object', 'object', t)
            # point_pub0.publish(pc0)
            point_pub0.publish(PointCloud(header=pc0.header, points=pc0.points[18:19]))
            point_pub1.publish(pc1)
            point_pub2.publish(pc2)
            rate.sleep()
