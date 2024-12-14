from __future__ import print_function
import numpy as np
from core import CubicTimeScaling, MatrixExp6, QuinticTimeScaling, MatrixLog3, TransInv, MatrixLog6



"""
The TrajectoryGenerator function generates a reference trajectory for the end-effector frame {e}.
This trajectory consists of eight concatenated trajectory segments as described below.

1. A trajectory to move the gripper from its initial configuration to a "standoff" configuration a 
   few cm above the block.
2. A trajectory to move the gripper down to the grasp position.
3. Closing of the gripper.
4. A trajectory to move the gripper back up to the "standoff" configuration.
5. A trajectory to move the gripper to a "standoff" configuration above the final configuration.
6. A trajectory to move the gripper to the final configuration of the object.
7. Opening of the gripper.
8. A trajectory to move the gripper back to the "standoff" configuration.

This function makes use of the ScrewTrajectory function from the MR library to generate the individual 
trajectory segments.
"""
def TrajectoryGenerator():
	"""Configuration of the frame {b} of the mobile base, relative to the frame {s}"""
	T_sb = np.array([[1, 0, 0, 0],
					 [0, 1, 0, 0],
					 [0, 0, 1, 0.0963],
					 [0, 0, 0, 1]])


	"""Fixed offset from chassis frame {b} to base of arm {0}."""
	T_b0 = np.array([[1, 0, 0, 0.1662],
					[0,  1, 0, 0],
					[0,  0, 1, 0.0026],
					[0,  0, 0, 1]])

	"""End-effector frame {e} relative to the arm base frame {0} (at home configuration)"""
	M_0e = np.array([[1, 0, 0, 0.033],
					[0,  1, 0, 0],
					[0,  0, 1, 0.6546],
					[0,  0, 0, 1]])

	"""
	Transformation which rotates the end-effector by 90 degrees about the y_e axis.
	"""
	T_r = np.array([[0, 0, 1, 0],
					[0,  1, 0, 0],
					[-1,  0, 0, 0],
					[0,  0, 0,  1]])

	"""Initial configuration of the end-effector"""
	# T_se = T_sb.dot(T_b0).dot(M_0e)
	T_se = np.array([[1,  0, 0, 0.1992],
					 [0,  1, 0, 0],
					 [0,  0, 1, 0.7535],
					 [0,  0, 0, 1]])


	"""First standoff position of the end-effector in the {s} frame (above the cube's initial config)."""
	T_se_standoff_1 = np.array([[1,  0,  0, 1.068],
							    [0,  1,  0, 0],
							    [0,  0,  1, 0.2],
							    [0,  0,  0, 1]])
	T_se_standoff_1 = T_se_standoff_1.dot(T_r)

	"""Initial configuration of the cube in the {s} frame."""
	T_sc_initial = np.array([[1,  0,  0, 1],
							 [0,  1,  0, 0],
							 [0,  0,  1, 0.025],
						     [0,  0,  0, 1]])

	"""Grasp configuration of the end-effector in the {s} frame."""
	T_se_grasp_1 = np.array([[1,  0,  0, 1.068],
							 [0,  1,  0, 0],
							 [0,  0,  1, 0.025],
						     [0,  0,  0, 1]])

	T_se_grasp_1 = T_se_grasp_1.dot(T_r)
	
	"""Final configuration of the cube in the {s} frame."""
	T_sc_goal = np.array([[0, 1, 0, 0],
						  [-1,  0, 0, -1],
						  [0,  0, 1,  0.025],
						  [0,  0, 0,  1]])	

	"""Second standoff position of the end-effector in the {s} frame (above the cube's final config)."""
	T_se_standoff_2 = np.array([[0, 1, 0, 0],
								[0,  0, -1, -1.068],
								[-1,  0, 0,  0.2],
								[0,  0, 0,  1]])		

	"""Release configuration of the end-effector in the {s} frame."""
	T_se_grasp_2 = np.array([[0, 1, 0, 0],
							[0,  0, -1, -1.068],
							[-1,  0, 0,  0.025],
							[0,  0, 0,  1]])	

	method = 3

	def ScrewTrajectory(Xstart, Xend, Tf, N, method):
		N = int(N)
		timegap = Tf / (N - 1.0)
		traj = [[None]] * N
		
		for i in range(N):
			if method == 3:
				s = CubicTimeScaling(Tf, timegap * i)
			else:
				s = QuinticTimeScaling(Tf, timegap * i)
			traj[i] = np.dot(Xstart, MatrixExp6(MatrixLog6(np.dot(TransInv(Xstart), Xend)) * s))

			with open('trajectory.csv', 'a', encoding='utf-8') as file:
				file.write('{:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {:.6f}, {}  \n'.format(traj[i][0][0], \
										traj[i][0][1], traj[i][0][2], traj[i][1][0], traj[i][1][1], traj[i][1][2], traj[i][2][0], traj[i][2][1],\
										traj[i][2][2], traj[i][0][3], traj[i][1][3], traj[i][2][3], gripper_state))
		return traj

	Tf = 4
	N = 401
	Xstart = T_se
	Xend = T_se_standoff_1
	gripper_state = 0
	#print("Moving to first standoff position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 2
	N = 201
	Xstart = T_se_standoff_1
	Xend = T_se_grasp_1
	gripper_state = 0
	#print("Descending to grasp position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 1
	N = 101
	Xstart = T_se_grasp_1
	Xend = T_se_grasp_1
	gripper_state = 1
	#print("Grasping cube...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 2
	N = 201
	Xstart = T_se_grasp_1
	Xend = T_se_standoff_1
	gripper_state = 1
	#print("Ascending to standoff position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 5
	N = 501
	Xstart = T_se_standoff_1
	Xend = T_se_standoff_2
	gripper_state = 1
	#print("Moving from first standoff position to second standoff position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 2
	N = 201
	Xstart = T_se_standoff_2
	Xend = T_se_grasp_2
	gripper_state = 1
	#print("Descending to grasp position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 1
	N = 101
	Xstart = T_se_grasp_2
	Xend = T_se_grasp_2
	gripper_state = 0
	#print("Releasing cube...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)

	Tf = 2
	N = 201
	Xstart = T_se_grasp_2
	Xend = T_se_standoff_2
	gripper_state = 0
	#print("Ascending to standoff position...")
	ScrewTrajectory(Xstart, Xend, Tf, N, method)


