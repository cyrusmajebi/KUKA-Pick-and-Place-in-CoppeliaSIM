import csv
import math
import numpy as np
from trajectory import TrajectoryGenerator
from odometry import NextStep
from control import FeedbackControl, WheelArmSpeeds
from core import FKinBody, IKinBody
from random_matrix import positive_definite_diagonal_ki_matrix, positive_definite_diagonal_kp_matrix
from plot import plot_error



def main_program():


    """ the  TrajectoryGenerator function is called which creates the desired reference trajectory for the YouBot """

    print("Generating desired reference trajectory")
    
    TrajectoryGenerator()

    with open('results/best/trajectory.csv', mode='r') as file:
        
        traj_list = list(csv.reader(file))
        traj_list = [[float(x) for x in p] for p in traj_list]

        
        # K_i = np.zeros(shape=(6, 6))
        # K_p = np.zeros(shape=(6, 6))

        """ the gain matrices K_p and K_i """
        K_i = positive_definite_diagonal_ki_matrix()
        K_p = positive_definite_diagonal_kp_matrix()

        """ initial chassis config """
        q = np.array([[0.0], [0.2], [0.0]])

        """ initial joint config """ 
        thetalist = np.array([[0.0], [0.0], [0.0], [0.377], [0.0]])

        """ initial wheel config """
        wheellist = np.array([[0.0], [0.0], [0.0], [0.0]])



        T_b0 = np.array([[1, 0, 0, 0.1662],
                        [0,  1, 0, 0],
                        [0,  0, 1,  0.0026],
                        [0,  0, 0,  1]])

        T_0e = np.array([[1, 0,  0, 0.033],
                        [0, 1,  0, 0],
                        [0, 0,  1, 0.6546],
                        [0, 0,  0, 1]])

        Blist = np.array([[0,  0,  1,  0, 0.033,  0],
                        [0, -1,  0, -0.5076, 0, 0],
                        [0, -1,  0, -0.3526, 0, 0],
                        [0, -1,  0, -0.2176, 0, 0],
                        [0,  0,  1, 0, 0, 0]]).T

        X = np.array([[1,  0,  0, 0.1992],
					  [0,  1,  0, 0],
					  [0,  0,  1,  0.7535],
					  [0,  0,  0,  1]])


        """ Loop that iterates through each reference trajectory configuration """
        print("Generating steps.csv file for animation")
        i = 0 
        while i < (len(traj_list) - 1):

            current = traj_list[i]
            next = traj_list[i+1]
           
            gripper_state = np.array([[current[12]]])
            current_config = np.concatenate((q, thetalist, wheellist, gripper_state), axis=0)
           
            """ current desired configuration """
            X_d = np.array([[current[0], current[1], current[2], current[9]], 
                            [current[3], current[4], current[5], current[10]],
                            [current[6], current[7], current[8], current[11]],
                            [0.0, 0.0, 0.0, 1.0]])

            """ next desired configuration """                
            X_d_next =  np.array([[next[0], next[1], next[2], next[9]], 
                            [next[3], next[4], next[5], next[10]],
                            [next[6], next[7], next[8], next[11]],
                            [0.0, 0.0, 0.0, 1.0]])


            """ calculate the end effector twist from FeedbackControl """
            end_effector_twist = FeedbackControl(X, X_d, X_d_next, K_i, K_p)


            """ calculate the wheel and arm controls from WheelArmSpeed  """
            wheel_and_arm_controls = WheelArmSpeeds(end_effector_twist, thetalist)


            """ calculate the next configuration from NextStep  """
            next_config = NextStep(wheel_and_arm_controls, current_config)

            """ extract q, thetalist and wheellist from next_config """
            q = next_config[:3]
            thetalist = next_config[3:8]
            wheellist = next_config[8:12]

            """ extract phi, x and y from q """
            phi = q[0][0]
            x = q[1][0]
            y = q[2][0]


            """ Calculate the actual configuration X for use in the next iteration of the loop """
            T_sb = np.array([[np.cos(phi), -np.sin(phi), 0, x],
                            [np.sin(phi),  np.cos(phi), 0, y],
                            [0,  0, 1,  0.0963],
                            [0,  0, 0,  1]])

            M_0e = FKinBody(T_0e, Blist, thetalist)
            X = T_se = T_sb.dot(T_b0).dot(M_0e)
            
            i += 1


        """ plot the error X_err once the loop is done """

        print("Writing error plot data.")
        plot_error()

        print("End of program")



main_program()

