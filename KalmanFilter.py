import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter(object):
    def __init__(self, F=None, B=None, H=None, Q=None, R=None, P=None, x0=None):

        if (F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u=0):
        self.x = np.matmul(self.F, self.x) + np.dot(self.B, u)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.matmul(self.H, self.x)
        S = self.R + np.matmul(self.H, np.matmul(self.P, self.H.T))
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(S))
        self.x = (self.x + np.matmul(K, y))
        I = np.eye(self.n)
        self.P = np.matmul(I - np.matmul(K, self.H), self.P)
        # (I - np.matmul(K, self.H)).T) + np.matmul(np.matmul(K, self.R), K.T)
        self.tr = np.trace(self.P)
        self.eig = np.linalg.eigvals(self.P)


def generator(F, H, Q, R, x0):
    # no control u and its responding sensivity B
    x = x0

    xx = np.array(np.zeros((2, 100)))
    yy = np.array(np.zeros((2, 100)))
    # xx = np.zeros((2,100))
    # yy = np.zeros((2,100))
    a = len(Q)
    b = len(R)
    for i in range(100):

        x = np.matmul(F, x) + \
            np.random.multivariate_normal(np.array([0, 0]), Q, 1).T
        xx[0][i] = x[0]
        xx[1][i] = x[1]
        yy[0][i] = (np.matmul(H, x) +
                    np.random.multivariate_normal(np.array([0, 0]), R, 1).T)[0]
        yy[1][i] = (np.matmul(H, x) +
                    np.random.multivariate_normal(np.array([0, 0]), R, 1).T)[1]

    return xx, yy


def example():

    dt = 0.1
    x0 = np.array([0, 0]).reshape(2, 1)
    P0 = np.array([[1, 0], [0, 1]])
    F = np.array([[1, dt], [0, 1]])
    # H = np.array([1, 0]).reshape(1, 2)
    H = np.array([[1, 0], [0, 1]])
    Q = np.array([[(1/4)*dt**4, (1/2)*dt**3], [(1/2)*dt**3, dt**2]])
    # R = np.array([0.1]).reshape(1, 1)
    R = np.array([[0.1, 0], [0, 0.01]])
    xx, measurements = generator(F, H, Q, R, x0)
    kf = KalmanFilter(F=F, H=H, Q=Q, R=R, P=P0, x0=x0)
    predictions = np.zeros((2, 100))
    update = np.zeros((2, 100))
    trace = np.zeros((1,100))
    eig = np.zeros((1,100))
    for i in range(100):
        # predictions.append(np.matmul(H,  kf.predict()))
        kf.predict()
        predictions[0][i] = kf.x[0]
        predictions[1][i] = kf.x[1]
        # zz = [measurements[0][i],measurements[1][i]].T

        kf.update((measurements[:, i]).reshape(-1, 1))
        update[0][i] = kf.x[0]
        update[1][i] = kf.x[1]
        trace[0][i] = kf.tr
        eig[0][i] = np.max(kf.eig)

    # plt.plot(range(len(measurements)), measurements, label = 'Measurements')
    plt.plot(range(len(update[0])), update[0],
             label='Kalman filter position estimates')
    plt.plot(range(len(xx[0])), xx[0], label='True position')
    
    plt.plot(range(len(measurements[0])),
             measurements[0], linestyle=':', label='Measured position')
    # plt.plot(range(np.shape(predictions)[0]), np.array(predictions)[0], label = 'Kalman Filter Prediction')
    plt.legend()
    # plt.show()
    plt.figure()
    plt.plot(range(len(update[1])), update[1],
             label='Kalman filter velocity estimates')
    plt.plot(range(len(xx[1])), xx[1], label='True velocity')
    plt.plot(range(len(measurements[1])),
             measurements[1], linestyle=':', label='Measured velocities')
    # plt.plot(range(np.shape(predictions)[0]), np.array(predictions)[0], label = 'Kalman Filter Prediction')
    plt.legend()

    plt.figure()
    plt.plot(range(len(trace[0])), trace[0], label='Covariance Trace')

    # plt.figure()
    # plt.legend()
    plt.plot(range(len(eig[0])), eig[0], linestyle=':', label='Spectral radius')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()
