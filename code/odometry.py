from __future__ import print_function
import numpy as np
from core import FKinBody, MatrixLog6, Adjoint, se3ToVec, JacobianBody


def NextStep(wheel_and_arm_controls, current_config):

    """ chassis config """
    q = current_config[:3]

    """ arm joint angles """
    joint_angles = current_config[3:8]  

    """ wheel angles """
    wheel_angles = current_config[8:12] 
    gripper_state = current_config[12:]

    """wheel speeds"""
    wheel_speeds = u = np.array([[wheel_and_arm_controls[0]], [wheel_and_arm_controls[1]], \
         [wheel_and_arm_controls[2]], [wheel_and_arm_controls[3]]]) 


    """ joint speeds """
    joint_speeds = np.array([[wheel_and_arm_controls[4]], [wheel_and_arm_controls[5]], [wheel_and_arm_controls[6]], \
         [wheel_and_arm_controls[7]], [wheel_and_arm_controls[8]]]) 

    r = 0.0475
    l = 0.235
    w = 0.15
    F = (r/4) * np.array([[-1/(l + w),  1/(l + w),  1/(l + w), -1/(l + w)], \
                          [1,  1,  1, 1],[-1, 1,  -1, 1]])
    dt = 0.01
    
    delta_theta = wheel_speeds * 0.01
    V_b = F.dot(delta_theta)
    w_bz = V_b[0][0]
    v_bx = V_b[1][0]
    v_by = V_b[2][0]
   
   

    if w_bz == 0:
        delta_qb = np.array([[0], [v_bx], [v_by]])
    else:
        delta_qb = np.array([[w_bz], [( v_bx*np.sin(w_bz) + v_by*(np.cos(w_bz) - 1))/w_bz],\
                            [( v_by*np.sin(w_bz) + v_bx*(1 - np.cos(w_bz)))/w_bz] ])

       
    phi_k = q[0][0]
    R = np.array([[1, 0, 0], [0, np.cos(phi_k), -np.sin(phi_k)],\
                    [0, np.sin(phi_k), np.cos(phi_k)] ])
    delta_q = R.dot(delta_qb)

    q += delta_q
    wheel_angles += wheel_speeds * dt
    joint_angles += joint_speeds * dt

    with open('results/best/steps.csv', 'a', encoding='utf-8') as file:
        file.write('{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {}  \n'.format(q[0][0],\
                                        q[1][0], q[2][0], joint_angles[0][0], joint_angles[1][0], joint_angles[2][0], joint_angles[3][0],\
                                            joint_angles[4][0], wheel_angles[0][0], wheel_angles[1][0], wheel_angles[2][0],  wheel_angles[3][0],\
                                            int(gripper_state[0])))

    new_config = np.concatenate((q, joint_angles, wheel_angles, gripper_state), axis=0)

    return new_config



