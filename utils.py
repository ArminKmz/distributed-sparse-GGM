import numpy as np
import scipy.io
import numpy.linalg as LA
from sklearn.covariance import graphical_lasso
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
glasso_lib = rpackages.importr('glasso')

def get_max_degree(graph):
    return int(np.max(np.sum(graph, axis=1)))

def glasso(cov, rho):
    '''
        USING R IMPLEMENTED [FASTER]
    '''
    numpy2ri.activate()
    ret = glasso_lib.glasso(cov, rho)
    numpy2ri.deactivate()
    return np.array(ret[1])

def _glasso(cov, rho):
    '''
        USING SKLEARN LIBRARY [SLOWER]
        note: diagonal coefficients do not penalize.
    '''
    ret = graphical_lasso(cov, rho)
    return np.array(ret[1])

def get_graph(J):
    '''
        J -> adjcancy matrix
        return adjcancy list
    '''
    p = J.shape[0]
    neighbors = []
    for i in range(p):
        tmp = []
        for j in range(p):
            if i != j and J[i, j] != 0:
                tmp.append(j)
        neighbors.append(tmp)
    return neighbors

def sparsity_pattern(A):
    '''
        return sparsity pattern of A.
    '''
    tmp = np.copy(A)
    np.fill_diagonal(tmp, 0)
    tmp[tmp != 0] = 1
    return tmp

def edges(J):
    '''
        J -> adj matrix
        return number of edges
    '''
    sp = sparsity_pattern(J)
    return np.sum(sp) // 2

def ddiff(real, estimated, log=True):
    '''
        real      -> real neighbors (adjcancy list)
        estimated -> estimated neighbors (adjcancy list)
        log       -> flag for printing log
        return: total edges, false negative, false positive
    '''
    p = len(real)
    e, not_all, wrong_all = 0, 0, 0
    for i in range(p):
        if log:
            not_detected = []
            wrong_detected = []
        for j in range(p):
            if j in real[i]:
                e += 1
            if (j in real[i]) and (j not in estimated[i]):
                not_all += 1
                if log:
                    not_detected.append(j+1)
            if (j in estimated[i]) and (j not in real[i]):
                wrong_all += 1
                if log:
                    wrong_detected.append(j+1)
        if log:
            print(i+1, ':', [x+1 for x in estimated[i]])
            print('not detected:'.ljust(16), not_detected)
            print('wrong detected:'.ljust(16), wrong_detected)
            print('-'*20)
    if log:
        print('edges:', e // 2)
        print('not detected all:', not_all // 2)
        print('wrong detected all:', wrong_all // 2)
    return e//2, not_all//2, wrong_all//2

def diff(ground_graph, predicted_graph):
    '''
        return false negative, false positive
    '''
    tmp = ground_graph - predicted_graph
    return int(np.sum(tmp[tmp==1])//2), int(abs(np.sum(tmp[tmp==-1]))//2)

def quantize(samples, R):
    '''
        return R-bit quantized of samples (according to codebook.mat)
    '''
    mat = scipy.io.loadmat('codebook.mat')
    codebook = mat['codebook'][R-1][0][0]
    intervals = mat['intervals'][R-1][0][0]
    distorted_samples = np.copy(samples)
    n, p = samples.shape
    for i in range(n):
        for j in range(p):
            k = 0
            while intervals[k] < distorted_samples[i][j]:
                k += 1
            distorted_samples[i][j] = codebook[k-1]
    return distorted_samples

def best_error(cov, ground_graph, return_precision_matrix=False):
    '''
        return fn+fp, fn, fp, (precision_matrix), rho for best
        found lambda.
    '''
    def error(cov, ground_graph, rho, return_precision_matrix):
        J = glasso(cov, rho)
        predicted_graph = sparsity_pattern(J)
        fn, fp = diff(ground_graph, predicted_graph)
        error = fn + fp
        if return_precision_matrix:
            return error, fn, fp, J
        return error, fn, fp
    _lambda = 0
    best_lambda = None
    best_error = 1e20
    for i in range(200):
        _lambda += 1e-3
        cur_error, _, __ = error(cov, ground_graph, _lambda, 0)
        if cur_error < best_error:
            best_lambda = _lambda
            best_error = cur_error
    _lambda = best_lambda
    for i in range(100):
        _lambda += 1e-5
        cur_error, _, __ = error(cov, ground_graph, _lambda, 0)
        if cur_error < best_error:
            best_lambda = _lambda
            best_error = cur_error
    if return_precision_matrix:
        error, fn, fp, J = error(cov, ground_graph, best_lambda, return_precision_matrix)
        return error, fn, fp, J, best_lambda
    error, fn, fp = error(cov, ground_graph, best_lambda, return_precision_matrix)
    return error, fn, fp, best_lambda

def error(cov, ground_graph, _lambda, return_precision_matrix=False):
    '''
        return fn+fp, fn, fp, (precision_matrix) for given lambda.
    '''
    J = glasso(cov, _lambda)
    predicted_graph = sparsity_pattern(J)
    fn, fp = diff(ground_graph, predicted_graph)
    if return_precision_matrix:
        return fn+fp, fn, fp, J
    return fn+fp, fn, fp

def original_data(samples, ground_graph, _lambda=None):
    N = samples.shape[0]
    cov = 1. / N * (samples.T @ samples)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def sign_method(samples, ground_graph, _lambda=None):
    sign_samples = np.sign(samples)
    assert(sign_samples[sign_samples==0].shape[0] == 0)
    N = samples.shape[0]
    cov = 1. / N * (sign_samples.T @ sign_samples)
    cov = np.sin(np.pi * cov / 2.)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def per_symbol_quantization_method(samples, ground_graph, r, _lambda=None):
    quantized_samples = quantize(samples, r)
    N = samples.shape[0]
    cov = 1. / N * (quantized_samples.T @ quantized_samples)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def joint_method(samples, ground_graph, Hr, Hi, snr, sigma2, _lambda=None):
    p = snr * sigma2
    samples = np.sqrt(p / 2.) * samples
    N, n = samples.shape
    x1_samples = samples[:N//2, :]
    x2_samples = samples[N//2:, :]
    H = np.zeros((2*n, 2*n))
    H[:n, :n] = Hr
    H[:n, n:] = -Hi
    H[n:, :n] = Hi
    H[n:, n:] = Hr
    H_inv = LA.inv(H)
    z1_samples = np.random.multivariate_normal(np.zeros(n), sigma2*np.eye(n), N // 2)
    z2_samples = np.random.multivariate_normal(np.zeros(n), sigma2*np.eye(n), N // 2)
    y_samples = []
    for i in range(N//2):
        y1 = Hr @ x1_samples[i, :] - Hi @ x2_samples[i, :] + z1_samples[i, :]
        y2 = Hr @ x2_samples[i, :] + Hi @ x1_samples[i, :] + z2_samples[i, :]
        y = np.zeros(2*n)
        y[:n] = y1
        y[n:] = y2
        y_samples.append(y)
    y_samples = np.array(y_samples)
    S_y = 2. / N * (y_samples.T @ y_samples)
    cov = H_inv @ (S_y - sigma2 * np.eye(2*n)) @ H_inv.T
    cov = (cov[:n, :n] + cov[n:, n:]) / 2.
    np.fill_diagonal(cov, p / 2.)
    w, v = LA.eig(cov)
    for i in range(w.shape[0]):
        w[i] = max(w[i], 1e-9)
    cov = v @ np.diag(w) @ LA.inv(v)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)
