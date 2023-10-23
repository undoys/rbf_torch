import numpy as np
from RBFNet import RBF
import scipy.io as sio
from sklearn.metrics import r2_score
import time
from sklearn.preprocessing import MinMaxScaler

def main():
    # load data
    data = sio.loadmat('./matlab.mat')
    # T, T1, U, U1 = np.transpose(np.array(data['T'])), np.transpose(np.array(data['T1'])),np.transpose(np.array(data['U'])), np.transpose(np.array(data['U1']))
    T = np.array(data['T'])
    T1 = np.array(data['T1'])
    U = np.array(data['U'])
    U1 = np.array(data['U1'])
    # normalize data
    scaler_t = MinMaxScaler(feature_range=(0, 1))
    train_t = scaler_t.fit_transform(T1)
    test_t = scaler_t.transform(T)
    scaler_u = MinMaxScaler(feature_range=(0, 1))
    train_u = scaler_u.fit_transform(U1)
    test_U = U
    # train
    # rbf = RBF(0.02, 10000, 10)
    rbf = RBF(train_t,train_u,10000)
    rbf.train(train_t,train_u)
    pre_u=rbf.predict(test_t)
    pre_U = scaler_u.inverse_transform(pre_u)
    # accuracy
    RR = [r2_score(test_U[i], pre_U[i]) for i in range(U.shape[1])]
    print(RR)
    print('Running time: %s Seconds' % (time.time() - start))

if __name__ == "__main__":
    start = time.time()
    main()
