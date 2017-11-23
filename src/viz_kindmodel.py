#!/usr/bin/env python
from __future__ import print_function
import numpy as np
from phasespace.load_mocap import find_homog_trans
from phasespace import load_mocap
import kinmodel
from extra_baxter_tools.conversions import matrix_to_pose_msg, array_to_point_msg, stamp
import tf
from tf.transformations import inverse_matrix
from geometry_msgs.msg import PoseStamped, PointStamped, Point, Quaternion, Pose, Point32
from sensor_msgs.msg import PointCloud
import rospy


FRAMERATE = 50


class KinTreeViz(object):

    def __init__(self, kin_tree, zero_points, ns=''):
        """

        :param kinmodel.KinematicTree kin_tree:
        """
        zero_points = zero_points.squeeze()
        self.kin_tree = kin_tree
        base_idxs, base_points_base = get_base_points_base(kin_tree)
        base_points_world = zero_points[base_idxs, :]
        self.transform = find_homog_trans(base_points_base, base_points_world)[0]
        self.points = PointCloud(points=[Point32(*p) for p in zero_points])
        self.points.header.frame_id='world'
        joints = self.kin_tree.get_params()
        joints.pop('base')

        self.axes = {joint_name: joint_to_msg(joint.params) for joint_name, joint in joints.items()}
        self.pubs = {joint_name: rospy.Publisher(joint_name, PoseStamped if isinstance(self.axes[joint_name], Pose) else PointStamped, queue_size=100) for joint_name in joints}
        self.tf_pub = tf.TransformBroadcaster()
        self.point_pub = rospy.Publisher(ns + '/mocap_point_cloud', PointCloud, queue_size=100)
        self.timer = rospy.Timer(rospy.Duration(0.25), self.publish)

    def publish(self, event):
        self.points.header.stamp = rospy.Time.now()
        self.point_pub.publish(self.points)
        self.tf_pub.sendTransform(self.transform[0:3, 3],
                             tf.transformations.quaternion_from_matrix(self.transform),
                             rospy.Time.now(), 'base_frame', 'world')
        for topic in self.pubs:
            self.pubs[topic].publish(stamp(self.axes[topic], frame_id='base_frame_ns'))


def joint_to_msg(params):
    if len(params) == 3:
        return Point(*params)
    elif len(params) == 6:
        return twist_to_pose(params)

    else:
        raise ValueError(str(params))


def twist_to_pose(xi):
    nu = xi[:3]
    w = xi[3:]
    q = np.cross(w, nu)
    quat_xyz = np.cross((1, 0, 0), w)
    quat = list(quat_xyz) + [1 + np.dot((1, 0, 0), w)]
    return Pose(position=Point(*q), orientation=Quaternion(*quat))


def get_base_points_base(kin_tree):
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
    base_frame_points = np.zeros((len(base_markers), 3))
    all_features = kin_tree.get_features()
    for i, marker in enumerate(base_markers):
        base_frame_points[i, :] = all_features[marker].q()

    base_idxs = [marker_indices[name] for name in base_markers]
    return base_idxs, base_frame_points



def track():
    # Load the calibration sequence
    rospy.init_node('kinmodel_viz')
    name = 'andrea'
    obj = 'obj1'
    human_json = '/home/pedge/experiment/%s/%s.json' % (name, name)
    obj_json = '/home/pedge/experiment/%s/%s_opt.json' % (obj,obj)

    calib_data = np.load('/home/pedge/experiment/%s/%s_rec.npz' % (obj,obj))
    mocap = load_mocap.ArrayMocapSource(calib_data['full_sequence'][:, :, :], FRAMERATE).get_stream()

    # Set the base frame coordinate transformation
    zero_points = mocap.read()[0][:,:,0]

    human_ktv = KinTreeViz(kinmodel.KinematicTree(json_filename=human_json), zero_points)
    kin_tree = kinmodel.KinematicTree(json_filename=obj_json)
    object_ktv = KinTreeViz(kin_tree, zero_points)
    rospy.spin()


if __name__ == '__main__':
    track()
