#!/usr/bin/env python
from kinmodel.syms import get_sym_jacobian, left_pinv
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
name = 'peter1'
load = False
sys.setrecursionlimit(sys.getrecursionlimit()*1000)


if load:
    print('Loading Kinematic Models')
    # object_kin_model = dill.load(open(HOME+'/object.d', 'rb'))
    # human_kin_model = dill.load(open(HOME+'/human.d', 'rb'))
    learning_model = dill.load(open(HOME+"/peter1.d", "rb"))

else:
    object_kin_tree = KinematicTree(json_filename=HOME + '/experiment/box/new_box_opt.json')

    input_features = object_kin_tree.get_joints()['joint_3'].children
    input_points = [np.array(p.primitive) for p in input_features]
    grip_location = grip_point(input_points)

    object_frame_tracker = KinematicTreeExternalFrameTracker(object_kin_tree)
    robot_indices, robot_points = object_frame_tracker.attach_frame('joint_0', 'robot')
    grip_indices, grip_points = object_frame_tracker.attach_frame('joint_3', 'grip', position=grip_location)

    human_kin_tree = KinematicTree(json_filename=HOME + '/experiment/%s/new_%s_opt.json' % (name, name)).to_1d_chain()
    human_frame_tracker = KinematicTreeExternalFrameTracker(human_kin_tree)
    human_indices, human_points = human_frame_tracker.attach_frame('base', 'human')
    human_frame_tracker.attach_frame('elbow', 'hand')

    object_joints = ['joint_%d' % i for i in range(4)]
    human_joints = ['shoulder_0', 'shoulder_1', 'shoulder_2', 'elbow']
    joint_names = human_joints + object_joints

    print('Creating Human Kinematic Model')
    human_jacobian = get_sym_jacobian(human_kin_tree, 'human', 'hand')

    print('Creating Object Kinematic Model')
    object_jacobian = get_sym_jacobian(object_kin_tree, 'robot', 'grip')

    print('Saving..')
    dill.dump(human_jacobian, open(HOME + "/human_jac.d", "wb"))
    dill.dump(object_jacobian, open(HOME + "/obj_jac.d", "wb"))


    print('Printing to Matlab')
    matlab_file = open(HOME + "/human_jac.m", "wb")
    matlab_string = octave_code(human_jacobian, assign_to='B')
    matlab_file.write(matlab_string)
    
    matlab_file = open(HOME + "/obj_jac.m", "wb")
    matlab_string = octave_code(object_jacobian, assign_to='B')
    matlab_file.write(matlab_string)

    print('Creating B(x) Matrix')
    print('Inverting')
    pinv_obj = left_pinv(object_jacobian)
    print('Multiplying')
    mult = pinv_obj*human_jacobian
    input_matrix = Matrix.vstack(eye(4), mult)

    print('Saving')
    dill.dump(input_matrix, open(HOME + "/input_matrix.d", "wb"))

    print('Printing to Matlab')
    matlab_file = open(HOME + "/input_matrix.m", "wb")
    matlab_string = octave_code(input_matrix, assign_to='B')
    matlab_file.write(matlab_string)
    matlab_file.close()

    print('Done')
