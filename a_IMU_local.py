from pr3_utils import *
from scipy import linalg

filename = "code/data/03.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
w_hat = hat_map_3d(angular_velocity)
# v_hat = hat_map_3d(linear_velocity)
twist_hat4 = np.vstack([np.hstack([w_hat, linear_velocity[:,np.newaxis,:]]) ,np.zeros((1,4,t.shape[1]))])
# twist_hat6 = np.vstack([np.hstack([w_hat, v_hat]), np.hstack([np.zeros((3,3,t.shape[1])), w_hat])])

tau = t[:,1::] - t[:,0:-1]

def imu_local(i, mu_imu):

    mu_imu = np.matmul(mu_imu, linalg.expm(tau[:,i] * twist_hat4[:,:,i]))

    return mu_imu