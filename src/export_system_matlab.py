#!/usr/bin/env python
from kinmodel.syms import get_sym_body_jacobian, left_pinv
from kinmodel import KinematicTree, Twist, Transform
from kinmodel.track_mocap import KinematicTreeExternalFrameTracker
import dill, sys
import numpy as np
from sympy import Matrix, eye
from os.path import expanduser
from sympy.printing.octave import octave_code


def grip_point(points):
    distances = [None] * len(points)
    midpoints = [None] * len(points)
    for i in range(len(points)):
        other_points = points[:]
        other_points.pop(i)
        diff = np.diff(other_points, axis=0).squeeze()
        midpoints[i] = np.mean(other_points, axis=0).squeeze()
        distances[i] = np.linalg.norm(diff)

    return midpoints[np.argmax(distances)]


HOME = expanduser("~")
names = ['rob', 'andrea', 'sarah']
load = False
sys.setrecursionlimit(sys.getrecursionlimit()*1000)


if load:
    print('Loading Kinematic Models')
    # object_kin_model = dill.load(open(HOME+'/object.d', 'rb'))
    # human_kin_model = dill.load(open(HOME+'/human.d', 'rb'))
    learning_model = dill.load(open(HOME+"/peter1.d", "rb"))

else:
    object_kin_tree = KinematicTree(json_filename=HOME + '/experiment/object/object.json')

    input_features = object_kin_tree.get_joints()['joint_3'].children
    input_points = [np.array(p.primitive) for p in input_features]
    grip_location = grip_point(input_points)

    object_frame_tracker = KinematicTreeExternalFrameTracker(object_kin_tree)
    robot_indices, robot_points = object_frame_tracker.attach_frame('base', 'robot')
    grip_indices, grip_points = object_frame_tracker.attach_frame('joint_3', 'grip', position=grip_location)

    for i, name in enumerate(names):
        human_kin_tree = KinematicTree(json_filename=HOME + '/experiment/%s/%s.json' % (name, name)).to_1d_chain()
        human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree)
        human_indices, human_points = human_frame_tracker.attach_frame('base', 'human')
        human_frame_tracker.attach_frame('elbow', 'grip')

        print('Creating Human Kinematic Model')
        human_jacobian = get_sym_body_jacobian(human_kin_tree, 'grip')

        print('Saving..')
        dill.dump(human_jacobian, open(HOME + "/human_jac_%d.d" % i, "wb"))

        print('Printing to Matlab')
        matlab_file = open(HOME + "/human_jac_%d.m" % i, "wb")
        matlab_string = octave_code(human_jacobian, assign_to='B')
        matlab_file.write(matlab_string)

    print('Creating Object Kinematic Model')
    object_jacobian = get_sym_body_jacobian(object_kin_tree, 'grip')

    print('Saving..')
    dill.dump(object_jacobian, open(HOME + "/obj_jac.d", "wb"))

    print('Printing to Matlab')
    matlab_file = open(HOME + "/obj_jac.m", "wb")
    matlab_string = octave_code(object_jacobian, assign_to='B')
    matlab_file.write(matlab_string)

    # print('Creating B(x) Matrix')
    print('Inverting')
    pinv_obj = left_pinv(object_jacobian)
    # print('Multiplying')
    # mult = pinv_obj*human_jacobian
    # input_matrix = Matrix.vstack(eye(4), mult)
    #
    print('Saving')
    dill.dump(pinv_obj, open(HOME + "/pinv_obj_jac.d", "wb"))
    #
    print('Printing to Matlab')
    matlab_file = open(HOME + "/input_matrix.m", "wb")
    matlab_string = octave_code(pinv_obj, assign_to='B')
    matlab_file.write(matlab_string)
    matlab_file.close()

    print('Done')
