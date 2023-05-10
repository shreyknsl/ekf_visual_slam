from pr3_utils import *
from scipy import linalg

############# Initialize Parameters ##############

#Load File
filename = "code/data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename)
features = features[:,0::12,:]

#timestep
tau = t[:,1::] - t[:,0:-1]

#IMU Pose parameters
Rotate_IMU = np.eye(4)
Rotate_IMU[1,1] = Rotate_IMU[2,2] = -1
imu_T_cam = np.matmul(Rotate_IMU, imu_T_cam)
mu_imu = np.zeros((4,4,t.shape[1]))
mu_imu[:,:,0] = np.eye(4)
delta_mu = np.zeros((t.shape[1],6,1))
delta_mu_hat = np.zeros((t.shape[1],4,4))
w_hat = hat_map_3d(angular_velocity)
v_hat = hat_map_3d(linear_velocity)
twist_hat4 = np.vstack([np.hstack([w_hat, linear_velocity[:,np.newaxis,:]]) ,np.zeros((1,4,t.shape[1]))])
twist_hat6 = np.vstack([np.hstack([w_hat, v_hat]), np.hstack([np.zeros((3,3,t.shape[1])), w_hat])])

# Initialize motion noise
W = np.zeros((6,6))
np.fill_diagonal(W[0:3,0:3], 0.001)
np.fill_diagonal(W[3:,3:], 0.001)

#Initialize Noise Covariance
V = np.eye(4) * 0.01

#Maintain visit frequency array
visit_freq = np.zeros((1,features.shape[1]))

#landmark position in world coordinates
m = np.zeros(features.shape)

#landmark position in world coordinates mean
mu_stereo = np.zeros((4, features.shape[1]))
mu_stereo_mean = np.zeros((4,features.shape[1])) + np.nan
mu_stereo[3,:] = 1

#Transforming K (3,3) to M (4,4)
M = generate_M(K,b)

#P (3,4)
P = np.hstack([np.eye(3), np.zeros((3,1))])

#Taking inverse of imu_T_cam
cam_T_imu = np.linalg.inv(imu_T_cam)


############### SLAM function ################

def slam(i,sigma):

    mu_imu[:,:,i+1] = np.matmul(mu_imu[:,:,i], linalg.expm(tau[:,i] * twist_hat4[:,:,i]))

    sigma[-6:,-6:] = np.matmul(np.matmul(linalg.expm(-tau[:,i] * twist_hat6[:,:,i]), sigma[-6:,-6:]), (linalg.expm((-tau[:,i] * twist_hat6[:,:,i])).transpose())) + W
      
    w_T_imu = mu_imu[:,:,i+1]


    ############ Landmark mapping #############
    
    #add noise to features
    v = np.random.normal(0,np.diag(V))
    
    # idx = np.array(np.where(features[0,:,i] != -1)).flatten()
    idx = np.where(features[0,:,i] != -1)
    features[:,idx[0],i] = features[:,idx[0],i] - v[:,np.newaxis]


    #Transform to Optical Coordinates
    coord_opt = np.zeros((4, features.shape[1]))
    coord_opt[2,:] = ((K[0,0] * b)/(features[0,:,i] - features[2,:,i]))
    coord_opt[0,:] = (((features[0,:,i] - K[0,2]) * coord_opt[2,:])/K[0,0])
    coord_opt[1,:] = ((features[1,:,i] - K[1,2]) * coord_opt[2,:])/K[1,1]
    coord_opt[3,:] = 1

    #Take inverse of IMU pose
    imu_T_w = np.linalg.inv(w_T_imu)

    #Transform to world coordinates
    m[:,:,i] = np.matmul(np.matmul(w_T_imu, imu_T_cam),coord_opt[:,:])

    #Check observed landmark

    #New code
    # idx = np.array(np.where((visit_freq[0,:] == 0) & (features[0,:,i] != -1))).flatten()

    not_seen_yet = np.where(visit_freq[0,:] == 0)
    first_time_seen = np.intersect1d(idx,not_seen_yet)
    mu_stereo_mean[:,first_time_seen] = m[:,first_time_seen,i]
    visit_freq[:,first_time_seen] += 1

    Nt_idx = np.array(idx).flatten()
    Nt = Nt_idx.shape[0]

    #Reshape mu_stereo_mean to (3m,1)
    mu_stereo_reshape = mu_stereo_mean[0:3,:].reshape(3*features.shape[1], order='F')
    mu_stereo_reshape_homo = mu_stereo_mean.reshape(4*features.shape[1], order='F')

    #Intialize Jacobian H
    H_imu = np.zeros((4*Nt, 6))
    H_stereo = np.zeros((4*Nt, 3*features.shape[1]))

    #Initialize z_tilda
    z_tilda = np.zeros((4*Nt, 1))

    k=0
    # print(idx.shape[0])
    for j in Nt_idx:

        #z_tilda
        pi = np.matmul(np.matmul(cam_T_imu, imu_T_w), mu_stereo_reshape_homo[4*j : 4*j + 4])
        pi = pi/pi[2]
        z_tilda[4*k : 4*k + 4] = np.matmul(M, pi[:,np.newaxis])
        
        #Jacobian of IMU
        H_imu[4*k : 4*k + 4,:] = -np.matmul(M, np.matmul(dcpi(np.matmul(np.matmul(cam_T_imu, imu_T_w), mu_stereo_reshape_homo[4*j : 4*j + 4])), np.matmul(cam_T_imu, circle_dot(np.matmul(imu_T_w, mu_stereo_reshape_homo[4*j : 4*j + 4])[:,np.newaxis]))))
        
        #Jacobian of Stereo
        H_stereo[4*k : 4*k + 4, 3*j : 3*j + 3] = np.matmul(np.matmul(M, dcpi(np.matmul(np.matmul(cam_T_imu, imu_T_w), mu_stereo_reshape_homo[4*j : 4*j + 4]))), np.matmul(np.matmul(cam_T_imu, imu_T_w), P.transpose()))

        #Sigma predict
        if (sigma[3*j,3*j] == 0):
            sigma[3*j : 3*(j+1), 3*j : 3*(j+1)] = np.eye(3) * 0.001

        k += 1
    #Stacking H for complete system
    H_system = np.hstack([H_stereo, H_imu])

    #Broadcasting observation noise
    IxV = np.eye(4*Nt) * 0.01

    #Innovation
    r = np.squeeze(features[:,idx,i], axis=1).reshape(z_tilda.shape, order = 'F') - z_tilda

    #Get Kalmam Gain for system
    K_system = np.matmul(sigma,np.matmul(H_system.transpose(), np.linalg.inv(np.matmul(np.matmul(H_system, sigma),H_system.transpose()) + IxV)))

    #Updating Covariance for system
    sigma = np.matmul(np.matmul((np.eye(sigma.shape[0]) - np.matmul(K_system, H_system)),sigma), (np.eye(sigma.shape[0]) - np.matmul(K_system, H_system)).transpose()) + np.matmul(np.matmul(K_system,IxV), K_system.transpose())

    #Update IMU pose
    mu_imu[:,:,i+1] = np.matmul(mu_imu[:,:,i+1], linalg.expm(hat4_2d(np.matmul(K_system[-6:,:], r))))		

    #Update Landmark positions
    mu_stereo_reshape[:,np.newaxis] = mu_stereo_reshape[:,np.newaxis] + np.matmul(K_system[0:-6,:], r)
    mu_stereo_mean[0:3] = mu_stereo_reshape.reshape(3, features.shape[1], order='F')

    return mu_stereo_mean, mu_imu[:,:,i+1], sigma
    # return mu_imu[:,:,i+1]