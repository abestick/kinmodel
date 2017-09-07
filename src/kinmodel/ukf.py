import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})


class UnscentedKalmanFilter(object):
    def __init__(self, proc_func, meas_func, x0, P0, Q=1e-1, R=1e-2):
        """Create an Unscented Kalman Filter instance with the specified process/measurement
        models and parameters.

        Args:
        proc_func - function handle: the process model function, which takes a single (N, k)
            ndarray as an argument and returns a (N, k) ndarray of transformed state vectors
        meas_func - function handle: the measurement model function, which takes a single (N, k)
            ndarray as an argument and returns a (M, k) ndarray of predicted measurement vectors
        x0 - (N, 1) ndarray: the initial state estimate of the system
        P0 - (N, N) ndarray: the initial state error covariance of the system
        Q - float: the additive process noise variance (assumed identical for each dimension)
        R - float: the additive measurement noise variance (assumed identical for each dimension)

        Note: N=state dimension, M=observation dimension
        """
        self.meas_func = meas_func
        self.proc_func = proc_func
        self.x = np.atleast_1d(x0.squeeze())[:,None]
        self.P = P0
        state_dim = self.x.shape[0]
        meas_dim = meas_func(x0).shape[0]
        self.Q = np.identity(state_dim) * Q
        self.R = R

    def filter(self, observation, plot_error=False):
        """Runs a single step of the UKF and returns the current state mean, covariance and sum of
        squared errors

        Args:
        observation - (meas_dim,) ndarray: the observation at the current time
        plot_error - bool: whether to plot the observed and predicted measurements for each
                frame (very slow right now)

        Returns:
        mean - (N,1) ndarray: the current state mean
        covariance - (N,N) ndarray: the current state covariance
        SSE - float: the sum of squared errors for this time step
        """
        # Generate a set of sigma points from x and P
        sigma_pts, mean_weights, covar_weights = generate_sigma_pts(self.x, self.P)

        # Propagate the sigma points through the process model
        process_sigma_pts = self.proc_func(sigma_pts)

        # Compute the new covariance and mean and add the process noise
        process_mean, process_covar = sigma_pts_to_mean_covariance(process_sigma_pts,
                process_sigma_pts, mean_weights, covar_weights)
        process_covar = process_covar + self.Q
        # process_sigma_pts = generate_sigma_pts(process_mean, process_covar)[0]

        # Propagate the updated sigma points through the measurement model, remove missing
        # measurements
        meas_sigma_pts = self.meas_func(process_sigma_pts)
        obs_missing = np.isnan(observation.squeeze())

        if obs_missing.all():
            # If we have no measurements simply propogate the model prediction
            self.x = process_mean[:, None]
            self.P = process_covar
            return self.x, self.P, np.nan

        observation = observation[np.logical_not(obs_missing)]
        meas_sigma_pts = meas_sigma_pts[np.logical_not(obs_missing),:]

        # Compute the mean and covariance of the predicted measurement values
        meas_mean, meas_covar = sigma_pts_to_mean_covariance(meas_sigma_pts, meas_sigma_pts,
                mean_weights, covar_weights)
        R = np.identity(meas_mean.shape[0]) * self.R
        meas_covar = meas_covar + R

        # Compute the cross-covariance of the measurement/state values
        cross_mean, cross_covar = sigma_pts_to_mean_covariance(process_sigma_pts, meas_sigma_pts,
                mean_weights, covar_weights)

        # Compute the error between the predicted and actual measurement values
        observation = observation.squeeze()
        error = observation - meas_mean
        if plot_error:
            self._plot_error(observation, meas_mean)

        # Compute the Kalman gain
        kalman_gain = cross_covar.dot(spla.inv(meas_covar))

        # Update the state mean and covariance values with the innovation
        self.x = process_mean[:,None] + kalman_gain.dot(error)[:,None]
        self.P = process_covar - kalman_gain.dot(meas_covar).dot(kalman_gain.T)
        return self.x, self.P, np.sum(error**2)

    def _plot_error(self, observation, meas_mean):
        observation = np.reshape(observation, (-1,3))
        meas_mean = np.reshape(meas_mean, (-1,3))

        if not hasattr(self, 'figure'):
            self.figure = plt.figure()
            self.axes = self.figure.add_subplot(111, projection='3d')
        else:
            self.axes.clear()

        for feature_idx in range(observation.shape[0]):
            # Predicted position
            predicted = meas_mean[feature_idx, :]
            self.axes.scatter(predicted[0], predicted[1], predicted[2], c='r', marker='o')

            # Observed position
            observed = observation[feature_idx, :]
            self.axes.scatter(observed[0], observed[1], observed[2], c='b', marker='o')
            endpoints = np.concatenate((observed[:,None], predicted[:,None]), axis=1)
            self.axes.plot(endpoints[0,:], endpoints[1,:], endpoints[2,:], 'k-')

        self.axes.set_xlabel('X Label')
        self.axes.set_ylabel('Y Label')
        self.axes.set_zlabel('Z Label')
        self.axes.auto_scale_xyz([-0.5,0.5], [-0.5,0.5], [-0.5,0.5])
        plt.ion()
        plt.pause(0.01)


def generate_sigma_pts(mean, covariance, alpha_squared=1.0e-3):
    # Make mean (N,)
    mean = np.atleast_1d(mean.squeeze())

    # Initialize (N,N) array to hold sigma pts and (N,) array of weights
    sigma_pts = np.zeros((mean.shape[0], (2 * mean.shape[0]) + 1))
    mean_weights = np.zeros((2 * mean.shape[0]) + 1)
    covar_weights = np.zeros((2 * mean.shape[0]) + 1)

    # Add two symmetric sigma points for each column of the covariance matrix
    scaled_covariance_sqrt = spla.sqrtm((covariance.shape[0] * alpha_squared) * covariance)
    for i in range(covariance.shape[0]):
        sigma_pts[:,2*i] = mean + scaled_covariance_sqrt[:,i]
        sigma_pts[:,2*i+1] = mean - scaled_covariance_sqrt[:,i]
        mean_weights[2*i:2*i+2] = 1.0 / (2 * covariance.shape[0] * alpha_squared)
        covar_weights[2*i:2*i+2] = 1.0 / (2 * covariance.shape[0] * alpha_squared)

    # Add the mean of the distribution
    sigma_pts[:,-1] = mean
    mean_weights[-1] = 1.0 - (1.0 / alpha_squared)
    covar_weights[-1] = 4.0 - (1.0 / alpha_squared) + alpha_squared

    return sigma_pts, mean_weights, covar_weights


def sigma_pts_to_mean_covariance(sigma_pts_x, sigma_pts_y, mean_weights, covar_weights):
    # Compute the means

    mean_x = np.sum(sigma_pts_x * mean_weights, axis=1)
    mean_y = np.sum(sigma_pts_y * mean_weights, axis=1)

    # Compute the covariance matrix
    zm_sigma_pts_x = sigma_pts_x - mean_x[:, None]
    zm_sigma_pts_y = sigma_pts_y - mean_y[:, None]
    covar = zm_sigma_pts_x.dot(np.diag(covar_weights)).dot(zm_sigma_pts_y.T)
    return mean_y, covar

if __name__ == "__main__":
    # mean = np.array([1,2,3]) 
    # covar = np.diag([0.25, 0.5, 0.75])
    # sigma_pts, mean_weights, covar_weights = generate_sigma_pts(mean, covar)

    A = np.array([[0.95, 0.05],[-0.05, 0.95]])
    C = np.zeros((4,2))
    C[0,0] = 2
    C[2,1] = 1
    C[1,1] = 3 
    C[3,0] = 4


    state = np.zeros((2,50))
    state[:,0] = np.array([[1.0, 0.0]])
    meas = np.zeros((4,50))
    meas[:,0] = C.dot(state[:,0])
    state_est = np.zeros((2,49))
    for i in range(1,state.shape[1]):
        state[:,i] = A.dot(state[:,i-1])
        meas[:,i] = C.dot(state[:,i])

    uk_filter = UnscentedKalmanFilter(lambda x:A.dot(x), lambda x:C.dot(x), np.zeros(2), np.identity(2)*0.75)
    for i in range(1,50):
        state_est[:,i-1] = uk_filter.filter(meas[:,i])[0].squeeze()
        print(state_est[:,i-1])
    1/0

    # trans = A.dot(sigma_pts)
    # mean_y, new_covar = sigma_pts_to_mean_covariance(sigma_pts, trans, mean_weights, covar_weights)