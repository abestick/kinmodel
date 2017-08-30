#!/usr/bin/env python
import numpy as np
import scipy.optimize
import se3
from math import pi
import kinmodel

class IKSolver(object):
    """Contains the kinematic tree, cost functions, and constraints associated
    with a given inverse kinematics problem.

    Methods allow the IK problem to be solved for different initial
    configurations
    """
    def __init__(self, tree):
        self.tree = tree

    def solve_ik(self, goal_feat_dict, init_config_dict=None,
            fixed_joint=None, fixed_joint_val=0.0):
        JAC_TOL = 1e-8
        SOLUTION_TOL = 1e-3

        if init_config_dict is not None:
            init_config_dict = [init_config_dict]

        #Define objective function with the one feature we'd like to use for our IK goal
        tree_obj_func = kinmodel.KinematicTreeObjectiveFunction(self.tree, [goal_feat_dict],
                init_config_dict, optimize={'configs':True, 'params':False, 'features':False},
                fix_zero_config=False)

        #Select initial config
        if fixed_joint is not None:
            print(fixed_joint_val)
            init_config_vec = tree_obj_func.get_current_param_vector()
            fixed_idx = tree_obj_func.get_vector_indices()[0][0][fixed_joint[0]][0] + fixed_joint[1]
            init_config_vec = np.delete(init_config_vec, fixed_idx)
            obj_func = lambda x: tree_obj_func.error(np.insert(x, fixed_idx, fixed_joint_val))
        else:
            init_config_vec = tree_obj_func.get_current_param_vector()
            obj_func = tree_obj_func.error


        #Run optimization
        result = scipy.optimize.minimize(obj_func, init_config_vec, method='L-BFGS-B', 
                options={'gtol':JAC_TOL})

        #Decide whether this is a solution and return as a config dict if so
        if obj_func(result.x) <= SOLUTION_TOL:
            if fixed_joint is not None:
                full_result = np.insert(result.x, fixed_idx, fixed_joint_val)
            else:
                full_result = result.x
            return tree_obj_func.unvectorize(full_result)[0][0], result
        else:
            return None

    def list_ik_sols(self, goal_feat_dict, init_config_dict=None, fixed_joint=None,
            fixed_joint_step_rad=0.1):
        # Generate an array of free joint displacements
        fixed_joint_disps = np.arange(0.0, pi*2, fixed_joint_step_rad)

        # Iterate over each free joint value
        solutions = []
        last_valid_solution = None
        for fixed_joint_val in fixed_joint_disps:
            result = self.solve_ik(goal_feat_dict, init_config_dict=last_valid_solution,
                fixed_joint=fixed_joint, fixed_joint_val=fixed_joint_val)

            # If solution found, add to list
            if result is not None:
                solutions.append(result[0])
                last_valid_solution = result[0]

        # Return list of solutions
        return solutions

# class KinematicCost(object):
#     def __init__(self, cost_func, jac_func):
#         self.cost_func = cost_func 
#         self.jac_func = jac_func

#     def get_cost(self, config):
#         #Takes a (N,) config and returns a scalar cost
#         return self.cost_func(config)

#     def get_jacobian(self, config):
#         #Takes a (N,) config and returns a (N,) gradient
#         return self.jac_func(config)


# class KinematicConstraint(KinematicCost):
#     def __init__(self, tree, constraint_type, frame, value):
#         #TODO: add orientation constraints
#         KinematicCost.__init__(self, self._constraint_cost, 
#                                self._constraint_jacobian)
#         self.tree = tree #The KinematicTree referenced by this constraint
#         self.type = constraint_type #Constraint type
#         self.frame = frame #Name of the constrained end effector frame
#         self.value = value #Desired value of the constrained frame (type depends on self.type)
#         #self.type=='position' -> self.value==Point, self.type=='orientation' -> self.value==Rotation
#         #Example types: 'position', 'orientation'

#     def _constraint_cost(self, config):
#         #Get the current value of the end effector transform
#         cur_trans = self.tree.get_transform(config, self.frame)

#         #Conmpute the value of the constraint depending on its type
#         if self.type is 'position':
#             diff = self.value.diff(cur_trans.position())
#             return diff.norm()**2
#         elif self.type is 'orientation':
#             raise NotImplementedError('Orientation constraints are not implemented')
#         else:
#             raise TypeError('Not a valid constraint type')

#     def _constraint_jacobian(self, config):
#         cur_trans = self.tree.get_transform(config, self.frame)
#         cur_jac = self.tree.get_jacobian(config, self.frame)

#         #Compute the velocity of the origin of the end effector frame,
#         #in spatial frame coordinates, for each joint in the manipulator
#         jac_hat = se3.hat(cur_jac) #4 x 4 x N ndarray
#         end_vel = np.zeros(jac_hat.shape)
#         for i in range(jac_hat.shape[2]):
#             end_vel[:,:,i] = jac_hat[:,:,i].dot(cur_trans.homog())
#         end_vel = se3.unhat(end_vel)

#         if self.type is 'position':
#             cost_jac = np.array(config)
#             cost_jac = 2 * cur_trans.position().x() - 2 * self.value.x()
#             return cost_jac.T.squeeze().dot(end_vel[3:,:])


# class QuadraticDisplacementCost(KinematicCost):
#     """Kinematic cost which penalizes movement away from a neutral pose.

#     The quadratic displacement cost is equal to the squared configuration space 
#     distance between the current kinematic configuration and a 
#     specified neutral configuration.

#     Args:
#     neutral_pos - (N,) ndarray: The neutral pose of the manipulator in c-space
#     """

#     def __init__(self, neutral_pos):
#         KinematicCost.__init__(self, self._cost, self._jacobian)
#         self.neutral_pos = neutral_pos

#     def _cost(self, config):
#         return la.norm(config - self.neutral_pos)**2

#     def _jacobian(self, config):
#         return 2 * config - 2 * self.neutral_pos

def main():
    from kinmodel import KinematicTree, Joint, Feature, Point, ThreeDofBallJoint, OneDofTwistJoint
    # Construct the kinematic chain
    j0 = Joint('joint0')
    j1 = Joint('joint1')
    j2 = Joint('joint2')

    j1.twist = ThreeDofBallJoint(np.ones(3))
    j2.twist = OneDofTwistJoint(np.array([1,0,0,0,0,0]))

    ft1 = Feature('feat1', Point(np.array([1,2,3,1])))
    ft2 = Feature('feat2', Point(np.array([3,2,1,1])))
    ft3 = Feature('feat3', Point(np.array([2,2,2,1])))
    ft4 = Feature('feat4', Point(np.array([5,1,2,1])))

    j0.children.append(j1)
    j1.children.append(ft1)
    j1.children.append(ft2)
    j1.children.append(j2)
    j2.children.append(ft3)
    j2.children.append(ft4)

    # Test JSON saving and loading
    tree = KinematicTree(j0)
    json_string_1 = tree.json()
    tree.json(filename='kinmodel_test_1.json')

    test_decode = KinematicTree(json_filename='kinmodel_test_1.json')
    json_string_2 = test_decode.json()
    assert json_string_1 == json_string_2, 'JSON saving/loading produces a different KinematicTree'

    #Test IKSolver
    EXAMPLE_IDX = 1
    tree.set_config({'joint1':[0, 0, 0], 'joint2':[0]})
    configs, feature_obs = kinmodel.generate_synthetic_observations(tree, EXAMPLE_IDX+2)
    tree.set_config({'joint1':[0, 0, 0], 'joint2':[0]})
    solver = IKSolver(tree)
    GOAL_FEATS = ['feat4']
    config = configs[EXAMPLE_IDX]
    goal = {feat_name:feature_obs[EXAMPLE_IDX][feat_name] for feat_name in GOAL_FEATS}

    goal_ft = Feature('feat4', Point(np.array([10,1,2,1])))
    tree.set_config({'joint1':[0, 0, 0], 'joint2':[0]})
    # test = solver.solve_ik(goal, fixed_joint=('joint1', 1), fixed_joint_val=config['joint1'][1])#{'feat4':goal_ft.primitive})
    test = solver.list_ik_sols(goal, fixed_joint=('joint1', 1), fixed_joint_step_rad=0.1)
    1/0

if __name__ == '__main__':
    main()