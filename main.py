import numpy as np
from pr3_utils import *
from tqdm import tqdm
from a_IMU_local import *
from b_landmark_mapping import *
from c_slam import *


if __name__ == '__main__':

	# Load the measurements
	filename = "code/data/10.npz"
	t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)

	features = features[:,0::12,:]

	# Initialize trajectory
	mu_imu = np.zeros((4,4,t.shape[1]))
	mu_imu[:,:,0] = np.eye(4)

	# Initialize Covariance
	sigma = np.zeros((3*features.shape[1] + 6, 3*features.shape[1] + 6))
	sigma[-6:,-6:] = np.eye(6) * 0.001

	# Initializae timesteps
	tau = t[:,1::] - t[:,0:-1]

	for i in tqdm(range(0, tau.shape[1])):


		# (a) IMU Localization via EKF Prediction
		# mu_imu[:,:,i+1] = imu_local(i, mu_imu[:,:,i])

		# (b) Landmark Mapping via EKF Update
		# mu_stereo_mean, sigma[0:-6,0:-6] = landmark_mapping(i, mu_imu[:,:,i+1], sigma[0:-6,0:-6])

		# (c) Visual-Inertial SLAM
		mu_stereo_mean, mu_imu[:,:,i+1], sigma = slam(i,sigma)
		
	visualize_map_2d(mu_imu, mu_stereo_mean, show_ori=True)