import numpy as np
import matplotlib.pyplot as plt

#CONSTANT
OMEGA = 0.1 # rad/sec
A = 10

FREQ_ACCL = 50 #Hz 50
FREQ_GPS = 2 #Hz 2
TP_ACCL = 1 / FREQ_ACCL
TP_GPS = 1 / FREQ_GPS
TP = TP_ACCL

WM_ACCL = 0
BM_ACCL = 0

X0_M = 0
V0_M = 100

ETA1M = 0
ETA1V = 1
ETA2M = 0
ETA2V = 4/10000

OPER_TIME = 30

T_ACCL = np.arange(0, OPER_TIME + TP_ACCL, TP_ACCL)
T_GPS = np.arange(0, OPER_TIME + TP_GPS, TP_GPS)
T = T_ACCL
TP_RATIO = TP_GPS/TP_ACCL

#VARIANCE
PVAR = 10 * 10 #meters^2
VVAR = 1 #(m/s)^2
BIASVAR = 0.01 #(m/s^2)^2
WVAR = 0.0001 #(m/s^2)^2

PLOT_KF = 0
PLOT_ORTHO = 0

def KalmanFilter(): #xhat, res, P, est, err, dxhat):
    #1. NOISE Process
    x0 = X0_M + np.sqrt(PVAR) * np.random.normal(loc=0, scale=1, size=1)
    v0 = V0_M + np.sqrt(VVAR) * np.random.normal(loc=0, scale=1, size=1)
    w = WM_ACCL + np.sqrt(WVAR) * np.random.normal(loc=0, scale=1, size=len(T_ACCL))
    b = BM_ACCL + np.sqrt(BIASVAR) * np.random.normal(loc=0, scale=1, size=1)
    eta_p = ETA1M + np.sqrt(ETA1V) * np.random.normal(loc=0, scale=1, size=len(T_GPS))
    eta_v = ETA2M + np.sqrt(ETA2V) * np.random.normal(loc=0, scale=1, size=len(T_GPS))

    #2. TRUE MODEL
    a_true = A * np.sin(OMEGA * T)
    v_true = v0 + A / OMEGA - (A / OMEGA) * np.cos(OMEGA * T)
    pos_true = x0 + (v0 + A/OMEGA) * T - (A / (OMEGA ** 2)) * np.sin(OMEGA * T)

    #3. ACCELEROMETER
    bias_vec = np.zeros((1, len(T_ACCL))).reshape((len(T_ACCL), ))
    for i in range(len(T_ACCL)):
        bias_vec[i] = b

    a_c = a_true + bias_vec + w
    v_c = np.zeros((1, len(T_ACCL))).reshape((len(T_ACCL), ))
    p_c = np.zeros((1, len(T_ACCL))).reshape((len(T_ACCL), ))

    v_c[0] = v0
    p_c[0] = x0

    for i in range(len(T_ACCL) - 1):
        i_next = i + 1
        v_c[i_next] = v_c[i] + a_c[i] * TP_ACCL
        p_c[i_next] = p_c[i] + v_c[i] * TP_ACCL + a_c[i] * TP_ACCL ** 2 / 2

    #4. GPS
    p_GPS = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))
    v_GPS = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))

    for i in range(len(T_GPS)):
        p_GPS[i] = pos_true[int(TP_RATIO*i)] + eta_p[i]
        v_GPS[i] = v_true[int(TP_RATIO * i)] + eta_v[i]


    #5. Measurements
    delta_p = pos_true - p_c
    delta_v = v_true - v_c
    z = np.zeros((2, len(T_GPS)))
    for i in range(len(T_GPS)):
        z[0, i] = delta_p[int(TP_RATIO*i)] + eta_p[i]
        z[1, i] = delta_v[int(TP_RATIO*i)] + eta_v[i]

    #6. Dynamics Model
    Phi = np.array([[1, TP, -(TP ** 2) / 2], [0, 1, -TP],[0, 0, 1]])
    PhiT = np.transpose(Phi)
    H = np.array([[1, 0, 0], [0, 1, 0]]) #2 by 3
    HT = np.transpose(H) #3 by 2
    R = np.array([[(TP ** 2)/2, TP, 0]]) #3 by 1
    RT = np.transpose(R) #1 by 3
    W = WVAR #1 by 1
    V = np.array([[ETA1V, 0], [0, ETA2V]]) #2 by 2
    invV = np.linalg.inv(V)
    M = np.array([[PVAR, 0, 0],[0, VVAR, 0], [0, 0, BIASVAR]]) #3 by 3
    xbar = np.array([0, 0, 0]) + 0.1 * np.random.normal(loc=0, scale=1, size=3)
    xbar = xbar.reshape((3, 1))

    #7. Kalman Filtering
    n = TP_RATIO
    X_bar = np.zeros((3, len(T))) #3 by length T
    dxhat = np.zeros((3, len(T_GPS)))
    estimate = np.zeros((3, len(T_GPS)))
    error = np.zeros((3, len(T_GPS)))
    xhatList = np.zeros((3, len(T_GPS)))
    Rmat = np.zeros((2, len(T_GPS)))
    Ks = np.zeros((3, 2, len(T_GPS)))
    KK = np.zeros((1, len(T_GPS)))
    Ps = np.zeros((3, 3, len(T_GPS)))
    Ms = np.zeros((3, 3, len(T_GPS)))
    RES = np.zeros((2, 2)) #2 by 2

    ii = 0

    for i in range(len(T)):
        if n == TP_RATIO:
            #8. UPDATE MEAN
            K_num = np.dot(M, HT)
            K_den = (np.dot(np.dot(H, M), HT) + V)
            K = np.dot(K_num, np.linalg.pinv(K_den)) #3 by 2
            Ks[:, :, ii] = K
            KK[:, ii] = K[0, 0]
            res = z[:, ii].reshape((2, 1)) - np.dot(H, xbar) #res, 2 by 1
            if ii == 72:
                RES[:, 0] = res.reshape((2, )) #RES 2 by 2
            elif ii == 35:
                RES[:, 1] = res.reshape((2, ))

            xhat = xbar + np.dot(K, res) #K 3 by 2, res 2 by 1
            Rmat[:, ii] = res.reshape((2, ))

            #9. Update Conditional Variance
            P = np.dot(np.dot((np.eye(3) - np.dot(K, H)), M), np.transpose(np.eye(3) - np.dot(K, H))) +\
                np.dot(np.dot(K, V), np.transpose(K))
            Ps[:, :, ii] = P# 3 by 3 by len T_GPS

            #10. Propagate the mean
            xbar = np.dot(Phi, xhat)
            X_bar[:, i] = xbar.reshape((3, ))

            #11. Propagate Variance
            M_1 = np.dot(np.dot(Phi, P), PhiT) + np.dot(np.dot(R, W), RT)
            M = M_1
            Ms[:, :, ii] = M

        else:
            xbar = np.dot(Phi, xbar)
            X_bar[:, i] = xbar.reshape((3, ))
            M_1 = np.dot(np.dot(Phi, P), PhiT) + np.dot(np.dot(R, W), RT)
            M = M_1
            Ms[:, :, ii] = M

        dxhat[:, ii] = xhat.reshape((3, ))

        if n == TP_RATIO:
            ii = ii + 1
            n = 0

        n = n + 1

    p_c_TP_RATIO = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))
    v_c_TP_RATIO = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))
    for i in range(len(T_GPS)):
        p_c_TP_RATIO[i] = p_c[int(TP_RATIO*i)]
        v_c_TP_RATIO[i] = v_c[int(TP_RATIO*i)]

    estimate[0, :] = dxhat[0, :] + p_c_TP_RATIO #3 by length T_GPS
    estimate[1, :] = dxhat[1, :] + v_c_TP_RATIO
    estimate[2, :] = dxhat[2, :]

    pos_true_TP_RATIO = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))
    v_true_TP_RATIO = np.zeros((1, len(T_GPS))).reshape((len(T_GPS), ))
    for i in range(len(T_GPS)):
        pos_true_TP_RATIO[i] = pos_true[int(TP_RATIO*i)]
        v_true_TP_RATIO[i] = v_true[int(TP_RATIO*i)]

    error[0, :] = estimate[0, :] - pos_true_TP_RATIO
    error[1, :] = estimate[1, :] - v_true_TP_RATIO
    error[2, :] = estimate[2, :] - b

    xhatList[0, :] = p_c_TP_RATIO + dxhat[0, :]
    xhatList[1, :] = v_c_TP_RATIO + dxhat[1, :]
    xhatList[2, :] = dxhat[2, :]

    if PLOT_KF == 1:
        plt.figure(1)
        plt.subplot(3, 1, 1)
        plt.plot(T, a_true)
        plt.title('Acceleration')
        # plt.xlabel('Time(s)')
        plt.ylabel('Acceleration (m/s^2)')
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(T, v_true)
        plt.title('Velocity')
        # plt.xlabel('Time(s)')
        plt.ylabel('Velocity (m/s)')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(T, pos_true)
        plt.title('Position')
        plt.xlabel('Time(s)')
        plt.ylabel('Position(m)')
        plt.grid(True)



        e1 = a_true - a_c
        e2 = v_true - v_c
        e3 = pos_true - p_c

        plt.figure(2)
        plt.subplot(3, 1, 1)
        plt.plot(T, e1)
        plt.title('Acceleration Error')
        # plt.xlabel('Time(s)')
        plt.ylabel('Acceleration Error(m/s^2)')
        plt.grid(True)


        plt.subplot(3, 1, 2)
        plt.plot(T, e2)
        plt.title('Velocity Error')
        # plt.xlabel('Time(s)')
        plt.ylabel('Velocity Error(m/s)')
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(T, e3)
        plt.title('Position Error')
        plt.xlabel('Time(s)')
        plt.ylabel('Position Error(m/s)')
        plt.grid(True)

        plt.figure(3)
        plt.subplot(3, 1, 1)
        plt.plot(T_GPS, error[0, :])
        plt.title('Filtered Position Error')
        # plt.xlabel('Time(s)')
        plt.ylabel('Position(m)')
        plt.hold(True)
        plt.plot(T_GPS, np.sqrt(Ps[0, 0, :]), 'r--')
        plt.plot(T_GPS, -np.sqrt(Ps[0, 0, :]), 'r--')
        plt.grid(True)
        plt.hold(False)

        plt.subplot(3, 1, 2)
        plt.plot(T_GPS, error[1, :])
        plt.title('Filtered Velocity Error')
        # plt.xlabel('Time(s)')
        plt.ylabel('Velocity(m/s)')
        plt.hold(True)
        plt.plot(T_GPS, np.sqrt(Ps[1, 1, :]), 'r--')
        plt.plot(T_GPS, -np.sqrt(Ps[1, 1, :]), 'r--')
        plt.grid(True)
        plt.hold(False)

        plt.subplot(3, 1, 3)
        plt.plot(T_GPS, error[2, :])
        plt.title('Filtered Bias Error')
        plt.xlabel('Time(s)')
        plt.ylabel('Bias Error')
        plt.hold(True)
        plt.plot(T_GPS, np.sqrt(Ps[2, 2, :]), 'r--')
        plt.plot(T_GPS, -np.sqrt(Ps[2, 2, :]), 'r--')
        plt.grid(True)
        plt.hold(False)

        plt.figure(4)
        plt.plot(T, [b] * len(T))
        plt.hold(True)
        plt.plot(T_GPS, estimate[2, :], 'x')
        plt.title('Bias')
        plt.ylabel('Bias')
        plt.xlabel('Time(s)')
        plt.hold(False)

        plt.figure(5)
        plt.title('K')
        plt.subplot(3, 2, 1)
        plt.plot(T_GPS, Ks[0, 0, :])

        plt.subplot(3, 2, 2)
        plt.plot(T_GPS, Ks[0, 1, :])

        plt.subplot(3, 2, 3)
        plt.plot(T_GPS, Ks[1, 0, :])

        plt.subplot(3, 2, 4)
        plt.plot(T_GPS, Ks[1, 1, :])

        plt.subplot(3, 2, 5)
        plt.plot(T_GPS, Ks[2, 0, :])

        plt.subplot(3, 2, 6)
        plt.plot(T_GPS, Ks[2, 1, :])


        plt.figure(8)
        plt.subplot(2, 1, 1)
        plt.plot(T_GPS, error[0])
        plt.title('Position Error in GPS')
        plt.ylabel('Position Error')
        plt.grid(True)


        plt.subplot(2, 1, 2)
        plt.plot(T_GPS, error[1])
        plt.title('Position Error in GPS')
        plt.ylabel('Position Error')
        plt.xlabel('Time (s)')
        plt.grid(True)

        plt.show()

    return xhat, RES, Ps, estimate, error, dxhat





def orthogonalcheck():
    N = 1000  # Monte Carlo 1000 Simulation
    T_ortho = np.arange(1, len(T_GPS) + 1)  # ??????????
    X_ortho = np.arange(0, N)
    Y_ortho = np.arange(0, len(T_GPS))
    ERR_ortho = np.zeros((3, len(T_GPS), N))
    xhat_ortho = np.zeros((3, len(T_GPS), N))
    estimate_ortho = np.zeros((3, len(T_GPS), N))
    dxhat_ortho = np.zeros((3, len(T_GPS), N))
    error_ave = np.zeros((3, len(T_GPS)))

    ortho_check = np.zeros((3, 3, len(T_GPS)))
    P_ortho = np.zeros((3, 3, len(T_GPS)))
    P_ave_ortho = np.zeros((3, 3, len(T_GPS)))
    P_diff_ortho = np.zeros((3, 3, len(T_GPS)))
    RES_ortho = np.zeros((2, 2, N))
    res_check_ortho = np.zeros((2, 2, N))

    #PERFORM MONTE CARLO SIMULATION
    #RUNNING THE KALMAN FILTER N TIMES and SAVING ALL DATA
    error_sum = np.zeros((3, len(T_GPS)))
    for i in range(N):
        xhat, RES, Ps, estimate, error, dxhat = KalmanFilter()
        P_ortho = Ps #3 by 3 by len(T_GPS)
        RES_ortho[:, :, i] = RES #2 by 2 by i
        ERR_ortho[:, :, i] = error
        xhat_ortho[:, :, i] = xhat
        estimate_ortho[:, :, i] = estimate
        dxhat_ortho[:, :, i] = dxhat
        error_sum = error_sum + error

    error_ave = error_sum / N

    for j in range(N):
        for k in range(len(T_GPS)):
            P_ave_ortho[:, :, k] = P_ave_ortho[:, :, k] + np.dot((ERR_ortho[:, k, j] - error_ave[:, k]), \
                                                           np.transpose(ERR_ortho[:, k, j] - error_ave[:, k]))
        res_check_ortho[:, :, j] = np.dot(RES_ortho[:, 0, j], np.transpose(RES_ortho[:, 1, j]))

    P_ave_ortho = P_ave_ortho / (N-1)
    P_diff_ortho = P_ave_ortho - P_ortho

    #Orthogonality CHECK
    for i in range(N):
        for j in range(len(T_GPS)):
            ortho_check[:, :, j] = ortho_check[:, :, j] + np.dot((ERR_ortho[:, j, i] - error_ave[:, j]), \
                                                           np.transpose(xhat_ortho[:, j, i]))#3 by 3 by len T_GPS

    ortho_check = ortho_check / N

    plt.figure(6)
    plt.title('Orthogonality Check')


    plt.subplot(3, 3, 1)
    plt.plot(T_GPS, ortho_check[0, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(T_GPS, ortho_check[0, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(T_GPS, ortho_check[0, 2, :])
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.plot(T_GPS, ortho_check[1, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.plot(T_GPS, ortho_check[1, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(T_GPS, ortho_check[1, 2, :])
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(T_GPS, ortho_check[2, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 8)
    plt.plot(T_GPS, ortho_check[2, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 9)
    plt.plot(T_GPS, ortho_check[2, 2, :])
    plt.grid(True)


    plt.figure(7)
    plt.title('Conditional Variance Difference')
    plt.subplot(3, 3, 1)
    plt.plot(T_GPS, P_diff_ortho[0, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(T_GPS, P_diff_ortho[0, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 3)
    plt.plot(T_GPS, P_diff_ortho[0, 2, :])
    plt.grid(True)

    plt.subplot(3, 3, 4)
    plt.plot(T_GPS, P_diff_ortho[1, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 5)
    plt.plot(T_GPS, P_diff_ortho[1, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 6)
    plt.plot(T_GPS, P_diff_ortho[1, 2, :])
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(T_GPS, P_diff_ortho[2, 0, :])
    plt.grid(True)

    plt.subplot(3, 3, 8)
    plt.plot(T_GPS, P_diff_ortho[2, 1, :])
    plt.grid(True)

    plt.subplot(3, 3, 9)
    plt.plot(T_GPS, P_diff_ortho[2, 2, :])
    plt.grid(True)



    plt.show()



if __name__ == '__main__':
    orthogonalcheck()









