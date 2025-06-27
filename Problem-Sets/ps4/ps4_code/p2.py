import numpy as np
from .p1 import Q1_solution


class Q2_solution(Q1_solution):
    @staticmethod
    def observation(x):
        """Implement Q3A. Observation function without noise.
        Input:
          x: (6,) numpy array representing the state.
        Output:
          obs: (3,) numpy array representing the observation (u,v,d).
        Note:
          we define disparity to be possitive.
        """
        # Hint: this should be similar to your implemention in Q2, but with two cameras
        raise NotImplementedError()
        return obs

    @staticmethod
    def observation_state_jacobian(x):
        """Implement Q3B. The jacobian of observation function w.r.t state.
        Input:
          x: (6,) numpy array, the state to take jacobian with.
        Output:
          H: (3,6) numpy array, the jacobian H.
        """
        H = np.zeros((3, 6))
        raise NotImplementedError()
        return H

    @staticmethod
    def observation_noise_covariance():
        """Implement Q3C here.
        Output:
          R: (3,3) numpy array, the covariance matrix for observation noise.
        """
        raise NotImplementedError()
        return R


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from plot_helper import draw_2d, draw_3d

    np.random.seed(315)
    solution = Q2_solution()
    states, observations = solution.simulation()
    # plotting
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(states[:, 0], states[:, 1], states[:, 2], c=np.arange(states.shape[0]))
    plt.show()

    fig = plt.figure()
    plt.scatter(
        observations[:, 0], observations[:, 1], c=np.arange(states.shape[0]), s=4
    )
    plt.xlim([0, 640])
    plt.ylim([0, 480])
    plt.gca().invert_yaxis()
    plt.show()

    observations = np.load("./data/Q2D_measurement.npy")
    (
        filtered_state_mean,
        filtered_state_sigma,
        predicted_observation_mean,
        predicted_observation_sigma,
    ) = solution.EKF(observations)
    # plotting
    true_states = np.load("./data/Q2D_state.npy")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(true_states[:, 0], true_states[:, 1], true_states[:, 2], c="C0")
    for mean, cov in zip(filtered_state_mean, filtered_state_sigma):
        draw_3d(ax, cov[:3, :3], mean[:3])
    ax.view_init(elev=10.0, azim=45)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:, 0], observations[:, 1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        draw_2d(ax, cov[:2, :2], mean[:2])
    plt.xlim([0, 640])
    plt.ylim([0, 480])
    plt.gca().invert_yaxis()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(observations[:, 0] - observations[:, 2], observations[:, 1], s=4)
    for mean, cov in zip(predicted_observation_mean, predicted_observation_sigma):
        # TODO find out the mean and convariance for (u^R, v^R).
        raise NotImplementedError()
        draw_2d(ax, right_cov, right_mean)
    plt.xlim([0, 640])
    plt.ylim([0, 480])
    plt.gca().invert_yaxis()
    plt.show()
