# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 17:50:50 2019

@author: elif.ayvali

Motion regression implementation:
[1]Sittel, Florian, Joerg Mueller, and Wolfram Burgard.
Computing velocities and accelerations from a pose time sequence in
three-dimensional space. Technical Report 272, University of
Freiburg, Department of Computer Science, 2013.
*There are ambiguities with the quaternion notation used in this paper.
Thw implementation here follows [3]. 

[2]Quaternion averaging implementation:
Markley, F. Landis, et al. "Averaging quaternions."
Journal of Guidance, Control, and Dynamics 30.4 (2007): 1193-1197.

Conversion from body to world coordinates using quaternion algebra:
[3]Sola, Joan. "Quaternion kinematics for the error-state Kalman filter." 
arXiv preprint arXiv:1711.02508 (2017).

Notes:
-The external measurements are taken in world frame.
-The linear and angular velocities and accelerations are estimated in both body frame and world frame.
-Unit quaternion:4 element sequence: w, x, y, z 
-Checks for quaternion sign flip
"""

import numpy as np
import numpy.matlib as npm
import math
import matplotlib.pyplot as plt
import scipy.io as sio

PREPROCESSING=False
data_file_name="./test_data.mat"
median_win_size=11 
win_size=11#regression win_size should be an odd number
#Plot data between
sample_min=0
sample_max=8000

class GetPoseDerivatives(object):
    def __init__(self):
        pass

    def __motion_regression_1d(self, pnts, t):
        sx = 0.0
        stx = 0.0
        st2x = 0.0
        st = 0.0
        st2 = 0.0
        st3 = 0.0
        st4 = 0.0
        for pnt in pnts:
            ti = pnt[1] - t #ti-t
            sx += pnt[0] #sx+xi
            stx += pnt[0] * ti
            st2x += pnt[0] * ti**2
            st += ti
            st2 += ti**2
            st3 += ti**3
            st4 += ti**4
        
        n = len(pnts)
        A = n * (st3 * st3 - st2 * st4) + \
            st * (st * st4 - st2 * st3) + \
            st2 * (st2 * st2 - st * st3)
        
        if A == 0.0:
            return 0.0, 0.0
        
        v = (1.0 / A) * (sx * (st * st4 - st2 * st3) +
                         stx * (st2 * st2 - n * st4) +
                         st2x * (n * st3 - st * st2))
        
        a = (2.0 / A) * (sx * (st2 * st2 - st * st3) +
                         stx * (n * st3 - st * st2) +
                         st2x * (st * st - n * st2))
        return v, a

    def motion_regression_6d(self, pnts, qt, t):        

        world_lin_vel = np.zeros(3)
        world_lin_acc = np.zeros(3)
        
        q_d = np.zeros(4)
        q_dd = np.zeros(4)
        
        num_pts=len(pnts['pos'][:,1])
 
       #----------------------MotionRegression1D------------------------------#
 
        #Linear velocity and acceleration in world frame  
        for k in range(3):
            v, a = self.__motion_regression_1d([(pnts['pos'][j,k], pnts['t'][j]) for j in range(num_pts)], t)
            world_lin_vel[k] = v
            world_lin_acc[k] = a
        
        for i in range(4):
            w, alpha = self.__motion_regression_1d([(pnts['quat'][j,i], pnts['t'][j]) for j in range(num_pts)], t)
            q_d[i] = w
            q_dd[i] = alpha
            
       #------------------SpatialRotationDerivatives--------------------------#
        #Angular velocity and acceleration in world frame
        world_ang_vel = 2 * np.matmul(self.__quaternion_rate_matrix_world(qt),q_d)
        world_ang_acc = 2 * np.matmul(self.__quaternion_rate_matrix_world(qt),q_dd) 
        
       #-----------------------Body Derivatives-------------------------------# 
       #Angular velocity (3,1) and acceleration (3,1) in body frame   
        body_ang_vel = 2 * np.matmul(self.__quaternion_rate_matrix_body(qt),q_d)
        body_ang_acc = 2 * np.matmul(self.__quaternion_rate_matrix_body(qt),q_dd) 
        #Linear velocity and acceleration in body frame

        #Linear velocity and acceleration in body frame
        body_lin_vel=np.matmul(self.__quat_world_to_body(qt),world_lin_vel)
        body_lin_acc=np.matmul(self.__quat_world_to_body(qt),world_lin_acc)
    
        #Account for fictitious forces in body frame
        body_lin_acc_ff=body_lin_acc-np.cross(body_ang_vel,body_lin_vel)
        return np.hstack((body_lin_vel, body_ang_vel)), \
                np.hstack((body_lin_acc_ff, body_ang_acc)),\
                np.hstack((world_lin_vel, world_ang_vel)),\
                np.hstack((world_lin_acc, world_ang_acc)),\
                q_d, q_dd
       
            
    def __quaternion_rate_matrix_world(self,q):
        #2q_conj.q_dot (quaternion multiplication)
        #2W(q)*q_dot (matrix multiplication)
        W_w=np.array([[-q[1],q[0],-q[3],q[2]],
                   [-q[2],q[3],q[0],-q[1]],
                   [-q[3],-q[2],q[1],q[0]]])
        return W_w
    
    def __quaternion_rate_matrix_body(self,q):
        #2q_dot.q_conj (quaternion multiplication)
        #2W_b(q)*q_dot (matrix multiplication)
        W_b=np.array([[-q[1],q[0],q[3],-q[2]],
                   [-q[2],-q[3],q[0],q[1]],
                   [-q[3],q[2],-q[1],q[0]]])
        return W_b
               
    def __quat_world_to_body(self,q):
        #q measurements of body frame wrt world frame
        #q transforms a vector in world frame to body frame
        #[0, v_body]= q . [0,v_world] . qinv
        #v_body=R*v_world
        #q:unit quaternion
        R_b=np.array([
                [1-2*q[2]**2-2*q[3]**2,     2*q[1]*q[2]-2*q[0]*q[3],     2*q[1]*q[3]+2*q[0]*q[2]],
                [    2*q[1]*q[2]+2*q[0]*q[3], 1-2*q[1]**2-2*q[3]**2,     2*q[2]*q[3]-2*q[0]*q[1]],
                [    2*q[1]*q[3]-2*q[0]*q[2],     2*q[2]*q[3]+2*q[0]*q[1], 1-2*q[1]**2-2*q[2]**2]              
                ])
        return R_b
    
    def __quat_body_to_world(self,q):
        R_w=self.__quat_world_to_body(q).T
        return R_w

    def __quaternion_conjugate(self, quat):
        return np.array(quat) * np.array([1.0, -1, -1, -1])
        
class Tools:
    
    def averageQuaternions(Q):
        ''' Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
           The quaternions are arranged as (w,x,y,z), where w is the scalar'''
        M = Q.shape[0]
        A = npm.zeros(shape=(4,4))
    
        for i in range(0,M):
            q = Q[i,:]
            # multiply q with its transposed version q' and add A
            #Uses equal weights: W=Identity
            A = np.outer(q,q) + A
    
        # scale
        A = (1.0/M)*A
        # compute eigenvalues and -vectors
        eigenValues, eigenVectors = np.linalg.eig(A)
        # Sort by largest eigenvalue
        eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        q_avg=np.real(eigenVectors[:,0].A1)
        #for applications that need +q
#        if q_avg[0]<0:
#            q_avg=-q_avg
        return q_avg
    
    def filterPosition(pos):      
        return np.mean(pos, axis = 0)
    
    def getCurrentPose(quat,pos):
        """ Return homogeneous rotation matrix from quaternion.
            pos: 3 element position
            quaternion : 4 element quaternion
            output : (4,4) array
        """
        q = np.array(quat, dtype=np.float64, copy=True)
        n = np.dot(q, q)
        if n < np.finfo(np.float).eps:
            return np.identity(4)
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)
        return np.array([
            [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], pos[0]],
            [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], pos[1]],
            [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], pos[2]],
            [                0.0,                 0.0,                 0.0, 1.0]])


Data = sio.loadmat(data_file_name)
quat=Data['q_rk4'] #GT quaternion measurements
pos=100*Data['w']+np.array([20,30,40])
time=Data['time'].T
w=Data['w']#GT body angular velocity
time_seq=np.asarray(time)
w_GT=np.asarray(w)
#Calculate velocity and acceleration
GPD=GetPoseDerivatives()
vel_save=[]
vel_world_save=[]
acc_save=[]
acc_world_save=[]
qd_save=[]
qdd_save=[]
quat_filtered=[]

if PREPROCESSING:
    quat_filtered=[]
    pos_filtered=[]
    pos_regression=[]
    quat_regression=[]
    filter_time=[]
    regression_time=[]
    quat_consistency=quat[0,:]
    # -----------------Filtering---------------#
    for i in range(len(time_seq)-int(median_win_size)):
        pnts={}
        pnts['pos']=pos[i:(median_win_size+i),:]
        pnts['t']=time_seq[i:(median_win_size+i)]
        pnts['quat']=quat[i:(median_win_size+i),:]
        #check quaternion consistency: dot(q_t+1,q_t)>0
        quat_avg=Tools.averageQuaternions(pnts['quat'])
        if np.dot(quat_consistency,quat_avg)<0:
            quat_avg=-quat_avg        
        quat_filtered.append(quat_avg)
        pos_filtered.append(Tools.filterPosition(pnts['pos']))
        filter_time.append(pnts['t'][math.ceil(median_win_size/2)-1])  
        quat_consistency=quat_avg
    quat_filtered_np=np.asarray(quat_filtered)  
    pos_filtered_np=np.asarray(pos_filtered) 
    filter_time_np=np.asarray(filter_time) 
    # ------------Perform Regression------------#
    for i in range(len(filter_time_np)-int(win_size)):
        pnts={}
        pnts['quat']=quat_filtered_np[i:(win_size+i),:]
        pnts['pos']=pos_filtered_np[i:(win_size+i),:]
        pnts['t']=filter_time_np[i:(win_size+i)]
        tj=pnts['t'][math.ceil(win_size/2)-1]
        qj=pnts['quat'][math.ceil(win_size/2)-1,:]  
#       (body_lin_vel, body_ang_vel) (body_lin_acc_ff, body_ang_acc),(world_lin_vel, world_ang_vel),(world_lin_acc, world_ang_acc),q_d, q_dd
        vel,acc,vel_world,acc_world,qd,qdd=GPD.motion_regression_6d(pnts,qj,tj)   
        vel_save.append(vel)
        vel_world_save.append(vel_world)
        acc_save.append(acc)
        acc_world_save.append(acc_world)
        qd_save.append(qd)
        qdd_save.append(qdd)
        pos_regression.append(pnts['pos'][math.ceil(win_size/2)-1])#only used for plotting
        quat_regression.append(pnts['quat'][math.ceil(win_size/2)-1])#only used for plotting/ground truth
        regression_time.append(tj)
        
else:
    pos_regression=[]
    quat_regression=[]
    regression_time=[]
    # -----Perform Regression Without Filtering------------#
    for i in range(len(time_seq)-int(win_size)):
        pnts={}
        pnts['quat']=quat[i:(win_size+i),:]
        pnts['pos']=pos[i:(win_size+i),:]
        pnts['t']=time_seq[i:(win_size+i)]
        tj=pnts['t'][math.ceil(win_size/2)-1]
        qj=pnts['quat'][math.ceil(win_size/2)-1,:]  
        vel,acc,vel_world,acc_world,qd,qdd=GPD.motion_regression_6d(pnts,qj,tj)   
        vel_save.append(vel)
        vel_world_save.append(vel_world)
        acc_save.append(acc)
        acc_world_save.append(acc_world)
        qd_save.append(qd)
        qdd_save.append(qdd)
        pos_regression.append(pnts['pos'][math.ceil(win_size/2)-1])#only used for plotting/ground truth
        quat_regression.append(pnts['quat'][math.ceil(win_size/2)-1])#only used for plotting/ground truth
        regression_time.append(tj)
        
vel_np=np.asarray(vel_save)
vel_world_np=np.asarray(vel_world_save)
acc_np=np.asarray(acc_save)
acc_world_np=np.asarray(acc_world_save)
qd_np=np.asarray(qd_save)
qdd_np=np.asarray(qdd_save)
regression_time_np=np.asarray(regression_time)
pos_regression_np=np.asarray(pos_regression)
quat_regression_np=np.asarray(quat_regression)



#-------------Plot Results------------------------------#
plt.figure(figsize=(15,5))
plt.grid()
if PREPROCESSING:
    plt.plot(filter_time_np[sample_min:sample_max],pos_filtered_np[sample_min:sample_max,0], 'k+')
    plt.plot(filter_time_np[sample_min:sample_max],pos_filtered_np[sample_min:sample_max,1], 'k+')
    plt.plot(filter_time_np[sample_min:sample_max],pos_filtered_np[sample_min:sample_max,2], 'k+')
    plt.title('Position Filtered')
else:
    plt.title('Position')   
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,0], 'r-')
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,1], 'g-')
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,2], 'b-')



plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,0], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,1], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,2], 'b-')
plt.title('Estimated linear velocity in world frame')

plt.figure(figsize=(15,5))
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,0], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,1], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,2], 'b-')
plt.title('Estimated linear acceleration in world frame')




plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,0], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,1], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,2], 'b-')
plt.title('Estimated linear velocity in body frame')

plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,0], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,1], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,2], 'b-')
plt.title('Estimated linear acceleration in body frame')



plt.figure(figsize=(15,5))
plt.grid()
if PREPROCESSING:
    plt.plot(filter_time_np[sample_min:sample_max],quat_filtered_np[sample_min:sample_max,0], 'k+')
    plt.plot(filter_time_np[sample_min:sample_max],quat_filtered_np[sample_min:sample_max,1], 'k+')
    plt.plot(filter_time_np[sample_min:sample_max],quat_filtered_np[sample_min:sample_max,2], 'k+')
    plt.plot(filter_time_np[sample_min:sample_max],quat_filtered_np[sample_min:sample_max,3], 'k+')
    plt.title('Quaternion Filtered')
else:
    plt.title('Quaternion')
plt.plot(time_seq[sample_min:sample_max],quat[sample_min:sample_max,0], 'r-')
plt.plot(time_seq[sample_min:sample_max],quat[sample_min:sample_max,1], 'g-')
plt.plot(time_seq[sample_min:sample_max],quat[sample_min:sample_max,2], 'b-')
plt.plot(time_seq[sample_min:sample_max],quat[sample_min:sample_max,3], 'k-')


plt.figure(figsize=(15,5))
plt.grid()
plt.plot(time_seq[sample_min:sample_max],w_GT[sample_min:sample_max,0], 'k+')
plt.plot(time_seq[sample_min:sample_max],w_GT[sample_min:sample_max,1], 'k+')
plt.plot(time_seq[sample_min:sample_max],w_GT[sample_min:sample_max,2], 'k+')
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,3], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,4], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],vel_np[sample_min:sample_max,5], 'b-')
plt.title('Estimated angular velocity in body frame vs GT')

plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,3], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,4], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],vel_world_np[sample_min:sample_max,5], 'b-')
plt.title('Estimated angular velocity in world frame')


plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,3], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,4], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],acc_np[sample_min:sample_max,5], 'b-')
plt.title('Estimated angular acceleration in body frame')


plt.figure(figsize=(15,5))
plt.grid()
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,3], 'r-')
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,4], 'g-')
plt.plot(regression_time_np[sample_min:sample_max],acc_world_np[sample_min:sample_max,5], 'b-')
plt.title('Estimated angular acceleration in world frame')


#Reconstruct trajectory from incremental p_rtr in world
time_rtr_save=[]
pos_rtr_world_save=[]
pos_rtr_save=[]
del_pos_rtr=np.zeros(3)
pos_rtr=np.zeros(3)
pos_rtr_world=pos_regression_np[0,:]
for i in range(len(regression_time)-1):
    delt=(regression_time[i+1]-regression_time[i])
    #del_pos_rtr=v*delt+0.5(a+wxv)delt^2 #body velocities/accelerations
    del_pos_rtr= vel_np[i,:3]*delt + 0.5*(acc_np[i,:3] + np.cross(vel_np[i,4:],vel_np[i,:3]))*(delt**2)
    pos_rtr=pos_rtr+del_pos_rtr
    R_pose=Tools.getCurrentPose(quat_regression_np[i,:],pos_regression_np[i,:])[:3,:3]
    R_pose_inv=np.linalg.inv(R_pose)
    del_pos_rtr_world=np.dot(R_pose_inv,del_pos_rtr)
    pos_rtr_world=pos_rtr_world+del_pos_rtr_world
    pos_rtr_world_save.append(pos_rtr_world)
    pos_rtr_save.append(pos_rtr)
    time_rtr_save.append(regression_time[i])

pos_rtr_world_np=np.asarray(pos_rtr_world_save)
pos_rtr_np=np.asarray(pos_rtr_save)
time_rtr_np=np.asarray(time_rtr_save)

plt.figure(figsize=(15,5))
plt.grid()
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_world_np[sample_min:sample_max,0], 'k+')
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_world_np[sample_min:sample_max,1], 'k+')
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_world_np[sample_min:sample_max,2], 'k+')  
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,0], 'r-')
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,1], 'g-')
plt.plot(time_seq[sample_min:sample_max],pos[sample_min:sample_max,2], 'b-')
if PREPROCESSING:
    plt.title('Position Filtered vs Integrated ' + r'$\Delta p_{rtr}$'+' Position in World Frame')
else:
    plt.title('Position vs Integrated ' +r'$\Delta p_{rtr}$'+ ' Position in World Frame')


plt.figure(figsize=(15,5))
plt.grid()
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_np[sample_min:sample_max,0], 'r-')
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_np[sample_min:sample_max,1], 'g-')
plt.plot(time_rtr_np[sample_min:sample_max],pos_rtr_np[sample_min:sample_max,2], 'b-') 

if PREPROCESSING:
    plt.title('Integrated Filtered ' + r'$\Delta p_{rtr}$'+'in Body Frame')
else:
    plt.title('Integrated ' + r'$\Delta p_{rtr}$'+'in Body Frame')


