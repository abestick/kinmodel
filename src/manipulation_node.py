#!/usr/bin/env python
from kinmodel.syms import CostLearningModel, ManipulationModel, CostModel, KinematicModel, CartesianTracker
from kinmodel import KinematicTree, Twist, Transform
from kinmodel.track_mocap import KinematicTreeExternalFrameTracker
from baxter_force_control.tools import grip_point
import dill, json, sys
import numpy as np
from os.path import expanduser


HOME = expanduser("~")
name = 'peter1'
load = False
sys.setrecursionlimit(sys.getrecursionlimit()*1000)


if load:
    print('Loading Kinematic Models')
    # object_kin_model = dill.load(open('/home/pedge/object.d', 'rb'))
    # human_kin_model = dill.load(open('/home/pedge/human.d', 'rb'))
    learning_model = dill.load(open("/home/pedge/peter1.d", "rb"))

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
    human_kin_model = KinematicModel(human_kin_tree, human_joints, 'human', 'hand')
    print('Saving...')
    dill.dump(human_kin_model, open('/home/pedge/human.d', 'wb'))
    print('Done.')

    print('Creating Object Kinematic Model')
    object_kin_model = KinematicModel(object_kin_tree, object_joints, 'robot', 'grip')
    print('Saving...')
    dill.dump(object_kin_model, open('/home/pedge/object.d', 'wb'))
    print('Done.')

    # object_kin_model = dill.load(open('/home/pedge/object.d', 'rb'))
    # human_kin_model = dill.load(open('/home/pedge/human.d', 'rb'))

    print('Creating Cartesian Tracker')
    cartesian_tracker = CartesianTracker(0.01875, Transform(), Twist())
    cartesian_tracker.track_frame('grip', grip_indices, grip_points)
    cartesian_tracker.track_frame('human', human_indices, human_points)
    cartesian_tracker.track_frame('robot', robot_indices, robot_points)

    print('Creating Manipulation Model')
    manip_model = ManipulationModel('robot', 'human', 'grip', cartesian_tracker, human_kin_model, object_kin_model)

    cost_names = ['ergonomic', 'configuration']
    with open(HOME + '/experiment/%s/%s_ergo_ref.json' % (name, name)) as f:
        ergo_reference = json.load(f)
    ergonomic_reference = [ergo_reference[k] for k in human_joints]
    configuration_reference = [0]*4
    references = [ergonomic_reference, configuration_reference]
    indices = [range(len(human_joints)), range(len(human_joints), len(joint_names))]

    print('Creating Cost Model')
    cost_model = CostModel(cost_names, references, indices)

    print('Creating Learning Model')
    learning_model = CostLearningModel(manip_model, cost_model)

    dill.dump(learning_model, open("/home/pedge/peter1.d", "wb"))


points = np.random.random((32, 3, 1))
learning_model.init(points)
print(learning_model.step(points))
