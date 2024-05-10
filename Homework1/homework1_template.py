import numpy as np

def problem_1a (A, B):
    return A + B

def problem_1b (A, B, C):
    return (A * B) + np.transpose(C)

def problem_1c (x, y):
    return x.T.dot(y)

def problem_1d (A, j):
    return np.sum(A[0::2, j])

def problem_1e (A, c, d):
    return np.mean(A[np.nonzero((A >= c) & (A <=d ))])

def problem_1f (x, k, m, s):
    return np.random.multivariate_normal(x + m, s * np.eye(x.shape[0]), k).T

def problem_1g (A):
    return A[:, np.random.permutation(A.shape[0])]

def problem_1h (x):
    return (x - np.mean(x)) / np.std(x)

def problem_1i (x, k):
    return np.repeat(np.atleast_2d(x), k, axis=0)

def linear_regression (X_tr, y_tr):
    X_t = X_tr.T
    X_y = np.matmul(X_tr.T, y_tr)
    X_XT = np.matmul(X_t, X_tr)
    return np.linalg.solve(X_XT, X_y)

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    y_te = np.load("age_regression_yte.npy")

    w = linear_regression(X_tr, y_tr)
    yhat_tr = np.matmul(X_tr, w)
    yhat_te = np.matmul(X_te, w)

    # Calculate MSE for training data
    trainDifference = np.square(yhat_tr - y_tr)
    trainMSE = np.mean(trainDifference)

    # Calculate MSE for testing data
    testDifference = np.square(yhat_te - y_te)
    testMSE = np.mean(testDifference)

    return trainMSE, testMSE
    # Returns (80.83988427156137, 749.3051827446681)

import matplotlib.pyplot as plt
from scipy.stats import poisson

def poisson_distribution ():
    data = np.load('PoissonX.npy')
    plt.hist(data,density=True)
    plt.show

    rateParameters = [2.5, 3.1, 3.7, 4.3]
    figure, axis = plt.subplots(2,2)

    cdf = poisson.cdf(data,rateParameters[0])
    axis[0, 0].plot(data, cdf)
    axis[0, 0].set_title("Rate Parameter of 2.5") 
    
    cdf = poisson.cdf(data,rateParameters[1])
    axis[0, 1].plot(data, cdf)
    axis[0, 1].set_title("Rate Parameter of 3.1") 
    
    cdf = poisson.cdf(data,rateParameters[2]) 
    axis[1, 0].plot(data, cdf)
    axis[1, 0].set_title("Rate Parameter of 3.7") 
    
    cdf = poisson.cdf(data,rateParameters[3])
    axis[1, 1].plot(data, cdf)
    axis[1, 1].set_title("Rate Parameter of 4.3")

    plt.show