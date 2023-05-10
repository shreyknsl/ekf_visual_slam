import numpy as np
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
# import transform3d


def load_data(file_name):
    '''
    function to read visual features, IMU measurements and calibration parameters
    Input:
        file_name: the input data file. Should look like "XX.npz"
    Output:
        t: time stamp
            with shape 1*ts
        features: visual feature point coordinates in stereo images, 
            with shape 4*n*t, where n is number of features
        linear_velocity: velocity measurements in IMU frame
            with shape 3*t
        angular_velocity: angular velocity measurements in IMU frame
            with shape 3*t
        K: (left)camera intrinsic matrix
            with shape 3*3
        b: stereo camera baseline
            with shape 1
        imu_T_cam: extrinsic matrix from (left)camera to imu, in SE(3).
            with shape 4*4
    '''
    with np.load(file_name) as data:
    
        t = data["time_stamps"] # time_stamps
        features = data["features"] # 4 x num_features : pixel coordinates of features
        linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
        angular_velocity = data["angular_velocity"] # angular velocity measured in the body frame
        K = data["K"] # intrindic calibration matrix
        b = data["b"] # baseline
        imu_T_cam = data["imu_T_cam"] # Transformation from left camera to imu frame 
    
    return t,features,linear_velocity,angular_velocity,K,b,imu_T_cam

t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data('code/data/03.npz')
fs_u = K[0,0]
fs_v = K[1,1]
c_u = round(K[0,2])
c_v = round(K[1,2])

def visualize_map_2d(pose,m=0,path_name="Trajectory",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
    ax.scatter(m[0,:], m[1,:], s=4, label="landmarks")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    # plt.scatter(m[0,:], m[1,:], s=4, label="landmarks")
    plt.show(block=True)

    return fig, ax

def visualize_trajectory_2d(pose,path_name="Unknown",show_ori=False):
    '''
    function to visualize the trajectory in 2D
    Input:
        pose:   4*4*N matrix representing the camera pose, 
                where N is the number of poses, and each
                4*4 matrix is in SE(3)
    '''
    fig,ax = plt.subplots(figsize=(5,5))
    n_pose = pose.shape[2]
    ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
    ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
    ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  
    if show_ori:
        select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
        yaw_list = []
        
        for i in select_ori_index:
            _,_,yaw = mat2euler(pose[:3,:3,i])
            yaw_list.append(yaw)
    
        dx = np.cos(yaw_list)
        dy = np.sin(yaw_list)
        dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
        ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
            color="b",units="xy",width=1)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.axis('equal')
    ax.grid(False)
    ax.legend()
    plt.show(block=True)

    return fig, ax



def hat_map_3d(x):
    
    #x = (3,n)
    x_hat = np.zeros((3, 3, x.shape[1]))
    x_hat[0,1,:] = -x[2,:]
    x_hat[0,2,:] =  x[1,:]
    x_hat[1,0,:] =  x[2,:]
    x_hat[1,2,:] = -x[0,:]
    x_hat[2,0,:] = -x[1,:]
    x_hat[2,1,:] =  x[0,:]
    
    return x_hat    #(3,3,n)


def hat_map_2d(x):
    
    #x = (3,1)
    x_hat = np.zeros((3, 3))
    x_hat[0,1] = -x[2,:]
    x_hat[0,2] =  x[1,:]
    x_hat[1,0] =  x[2,:]
    x_hat[1,2] = -x[0,:]
    x_hat[2,0] = -x[1,:]
    x_hat[2,1] =  x[0,:]
    
    return x_hat    #(3,3)


def feature_transform(time):
    
    z = fs_u*b/(features[0,:,time] - features[2,:,time])
    # z[z>50] = 0
    x = (features[0,:,time] - c_u)*z/fs_u
    y = (features[1,:,time] - c_v)*z/fs_v 

    return x,y,z


# def dpi_dq(x):

#     # x (4,n)
#     # dpi_dx (4,4,n)

#     r1 = np.hstack([np.ones((1,1,x.shape[1])), np.zeros((1,1,x.shape[1])), -(x[0,:]/x[2,:])[np.newaxis, np.newaxis, :], np.zeros((1,1,x.shape[1]))])
#     r2 = np.hstack([np.zeros((1,1,x.shape[1])), np.ones((1,1,x.shape[1])), -(x[1,:]/x[2,:])[np.newaxis, np.newaxis, :], np.zeros((1,1,x.shape[1]))])
#     r3 = np.hstack([np.ones((1,1,x.shape[1])),np.ones((1,1,x.shape[1])),np.ones((1,1,x.shape[1])),np.ones((1,1,x.shape[1]))])
#     r4 = np.hstack([np.zeros((1,1,x.shape[1])), np.zeros((1,1,x.shape[1])), -(x[3,:]/x[2,:])[np.newaxis, np.newaxis, :], np.ones((1,1,x.shape[1]))])

#     dpi_dx = (np.vstack([r1,r2,r3,r4]))/x[2,:]

#     return dpi_dx


def dcpi(q):
    dq = np.zeros((4,4))
    dq[0,0] = 1/q[2]
    dq[0,2] = -q[0] / (q[2] * q[2])
    dq[1,1] = 1/q[2]
    dq[1,2] = -q[1] / (q[2] * q[2])
    dq[3,2] = -q[3] / (q[2] * q[2])
    dq[3,3] = 1/q[2]
    return dq


def circle_dot(x):

    # x(4,1)

    y = np.vstack([np.hstack([np.eye(3), -hat_map_2d(x[0:3])]), np.zeros((1,6))])

    # y (4,6)

    return y


def hat4_2d(x):

    # x (6,1)

    y = np.vstack([np.hstack([hat_map_2d(x[3:]), x[0:3]]) ,np.zeros((1,4))])

    # y (4,4)

    return y

def generate_M(K,b):
    Ks = np.zeros((4,4))
    Ks[0,0] = K[0,0]
    Ks[0,2] = K[0,2]
    Ks[1,1] = K[1,1]
    Ks[1,2] = K[1,2]
    Ks[2,0] = K[0,0]
    Ks[2,2] = K[0,2]
    Ks[2,3] = -1 * b * K[0,0]
    Ks[3,1] = K[1,1]
    Ks[3,2] = K[1,2]
    return Ks