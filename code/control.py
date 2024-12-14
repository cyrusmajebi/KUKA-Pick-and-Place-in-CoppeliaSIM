from __future__ import print_function

import numpy as np
from core import FKinBody, MatrixLog6, Adjoint, se3ToVec, JacobianBody


def FeedbackControl(X, X_d, X_d_next, K_i, K_p):

    r = 0.0475
    l = 0.235
    w = 0.15
    F = (r/4) * np.array([[-1/(l + w),  1/(l + w),  1/(l + w), -1/(l + w)],
                        [1,  1,  1, 1],
                        [-1, 1,  -1, 1]])
    dt = 0.01
    
    """Calculate feedforward twist V_d."""
    X_d_next_inv = np.linalg.inv(X_d)
    T = X_d_next_inv.dot(X_d_next)
    V_d_br = (1/0.01) * MatrixLog6(T)
    V_d = se3ToVec(V_d_br)

    """Express twist V_d in current frame of end-effector at X."""
    L = Adjoint(np.linalg.inv(X).dot(X_d))
    L = L.dot(V_d)

    """Calculate error twist X_err that takes X to Xd in unit time."""
    X_err_br = MatrixLog6(np.linalg.inv(X).dot(X_d))
    X_err = se3ToVec(X_err_br)
    X_err = np.round(X_err, 3)

    with open('results/best/error.csv', 'a', encoding='utf-8') as file:
        file.write('{}, {}, {}, {}, {}, {}\n'.format(X_err[0], X_err[1], X_err[2], X_err[3], X_err[4], X_err[4]))

    """Return V, the commanded end-effector twist expressed in end-effector frame {e}"""
    V = L + K_p.dot(X_err) + K_i.dot(X_err).dot(dt)

    return V



def WheelArmSpeeds(V, thetalist):
    r = 0.0475
    l = 0.235
    w = 0.15
    F = (r/4) * np.array([[-1/(l + w),  1/(l + w),  1/(l + w), -1/(l + w)],
                        [1,  1,  1, 1],
                        [-1, 1,  -1, 1]])

    T_b0 = np.array([[1, 0, 0, 0.1662],
				     [0,  1, 0, 0],
				     [0,  0, 1,  0.0026],
				     [0,  0, 0,  1]])

    Blist = np.array([[0,  0,  1,  0, 0.033,  0],
                      [0, -1,  0, -0.5076, 0, 0],
                      [0, -1,  0, -0.3526, 0, 0],
                      [0, -1,  0, -0.2176, 0, 0],
                      [0,  0,  1, 0, 0, 0]]).T

    """Home configuration of the end-effector"""
    M = T_0e = np.array([[1, 0,  0, 0.033],
                         [0, 1,  0, 0],
                         [0, 0,  1, 0.6546],
                         [0, 0,  0, 1]])

    M_0e = FKinBody(M, Blist, thetalist)

    """Calculate J_base from M_0e, T_b0 and F6"""
    P = Adjoint(np.linalg.inv(M_0e).dot(np.linalg.inv(T_b0)))
    F6 = np.array([[0,  0,  0,  0],
                [0,  0,  0,  0],
                [F[0][0], F[0][1],  F[0][2], F[0][3]],
                [F[1][0], F[1][1],  F[1][2], F[1][3]],
                [F[2][0], F[2][1],  F[2][2], F[2][3]],
                [ 0, 0,  0, 0]])
    J_base = P.dot(F6)

    """Calculate J_arm from body Jacobian"""
    J_arm = JacobianBody(Blist, thetalist)
    J_e = np.concatenate((J_base, J_arm), axis=1)
    np.set_printoptions(suppress=True, formatter={'float_kind':'{:.6f}'.format})

    """Return wheel and arm controls from Jacobian pseudoinverse of J_e """
    J_e_pinv = np.linalg.pinv(J_e, rcond=1e-5, hermitian=False)
    controls = J_e_pinv.dot(V)

    return controls

