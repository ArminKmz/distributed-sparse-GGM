import numpy as np
import scipy.io
import numpy.linalg as LA
from sklearn.covariance import graphical_lasso
import rpy2.robjects.packages as rpackages
from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from scipy import stats
glasso_lib = rpackages.importr('glasso')


def get_grid(n, theta):
    N = n * n
    Q_inv = np.zeros((N, N))
    rc = [0, 0, 1, -1]
    cc = [1, -1, 0, 0]
    def _in(_x, _y):
        return (0 <= _x and 0 <= _y and _x < n and _y < n)
    for i in range(N):
        Q_inv[i, i] = 1
        x = i // n
        y = i % n
        for k in range(4):
            xx = x + rc[k]
            yy = y + cc[k]
            if _in(xx, yy):
                j = xx * n + yy
                Q_inv[i, j]  = theta
    Q = LA.inv(Q_inv)
    for i in range(Q_inv.shape[0]):
        Q_inv[i, :] *= np.sqrt(Q[i, i])
        Q_inv[:, i] *= np.sqrt(Q[i, i])
    print('min_theta:', np.min(np.min(abs(Q_inv[Q_inv!=0]))))
    return Q_inv

def get_star(n, rho, delta=None, normalize=False):
    if delta == None:
        delta = n - 1
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = 1
        if 1 <= i and i <= delta:
            Q[i, 0] , Q[0, i] = rho, rho
            if normalize:
                Q[i, 0] /= (delta+.0)
                Q[0, i] /= (delta+.0)
    for i in range(1, delta+1):
        for j in range(i+1, delta+1):
            Q[i, j] = Q[0, i] * Q[0, j]
            Q[j, i] = Q[i, j]
    Q_inv = LA.inv(Q)
    Q_inv[abs(Q_inv) < 1e-12] = 0
    return Q_inv

def get_chain(n, rho):
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = 1
        for j in range(i+1, n):
            Q[i, j] = rho ** (j-i)
            Q[j, i] = Q[i, j]
    Q_inv = LA.inv(Q)
    Q_inv[abs(Q_inv) < 1e-12] = 0
    return Q_inv

def get_cycle(n, omega):
    Q_inv = np.zeros((n, n))
    for i in range(n):
        Q_inv[i, i] = 1
        Q_inv[i, (i+1)%n] = omega
        Q_inv[(i+1)%n, i] = omega
    Q = LA.inv(Q_inv)
    Q_inv = np.diag(np.diag(Q)) @ Q_inv
    return Q_inv

def get_model_complexity(Q_inv, delta):
    theta_min = np.min(abs(Q_inv[np.nonzero(Q_inv)]))

    Q = LA.inv(Q_inv)
    Q[Q<1e-12] = 0
    print(Q)
    kappa_sigma = LA.norm(Q, np.inf)

    gamma = np.kron(Q, Q)
    p = Q.shape[0]
    ei, ej = np.nonzero(Q_inv)
    Q_inv_c = np.copy(Q_inv)
    Q_inv_c[Q_inv==0] = 1
    Q_inv_c[Q_inv!=0] = 0
    cei, cej = np.nonzero(Q_inv_c)
    A = np.zeros((cei.shape[0], ei.shape[0]))
    for i in range(cei.shape[0]):
        a = cei[i]
        b = cej[i]
        for j in range(ei.shape[0]):
            c = ei[j]
            d = ej[j]
            A[i, j] = gamma[a*p+b, c*p+d]
    B = np.zeros((ei.shape[0], ej.shape[0]))
    for i in range(ei.shape[0]):
        a = ei[i]
        b = ej[i]
        for j in range(ej.shape[0]):
            c = ei[j]
            d = ej[j]
            B[i, j] = gamma[a*p+b, c*p+d]
    kappa_gamma = LA.norm(LA.inv(B), np.inf)
    alpha = 1 - LA.norm(A @ LA.inv(B), np.inf)

    K = (1 + 8. / alpha) * max(kappa_gamma / theta_min, 3*delta*
            max(kappa_sigma*kappa_gamma, kappa_gamma**2*kappa_sigma**3))
    return (K, alpha, kappa_gamma, kappa_sigma, theta_min)

def get_max_degree(graph):
    return int(np.max(np.sum(graph, axis=1)))

def glasso(cov, rho):
    '''
        USING R IMPLEMENTED [FASTER]
    '''
    numpy2ri.activate()
    ret = glasso_lib.glasso(cov, rho, thr=1e-10, maxit=1e5, penalize_diagonal=False)
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

def sign_diff(Q_inv, J):
    '''
        return false negative, false positive with considering signs
    '''
    fn, fp = 0, 0
    for i in range(Q_inv.shape[0]):
        for j in range(i+1, Q_inv.shape[0]):
            if J[i, j] == 0 and Q_inv[i, j] != 0: fn += 1
            elif J[i, j] != 0 and Q_inv[i, j] == 0: fp += 1
            elif J[i, j] != 0 and Q_inv[i, j] != 0 and np.sign(J[i, j]) != np.sign(Q_inv[i, j]):
                if np.sign(J[i, j]) == +1: fp += 1
                else: fn += 1
    return fn, fp


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

def sign_error(cov, Q_inv, _lambda, return_precision_matrix=False):
    J = glasso(cov, _lambda)
    fn, fp = sign_diff(Q_inv, J)
    if return_precision_matrix:
        return fn+fp, fn, fp, J
    return fn+fp, fn, fp

def original_data(samples, Q_inv, _lambda=None, sign=False):
    N = samples.shape[0]
    cov = 1. / N * (samples.T @ samples)
    # cov = np.cov(samples.T)
    ground_graph = sparsity_pattern(Q_inv)
    if sign:
        if _lambda == None:
            raise Exception('_lambda must given in sign error.')
        return sign_error(cov, Q_inv, _lambda)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def sign_method(samples, Q_inv, _lambda=None, sign=False):
    ground_graph = sparsity_pattern(Q_inv)
    sign_samples = np.sign(samples)
    assert(sign_samples[sign_samples==0].shape[0] == 0)
    N = samples.shape[0]
    cov = 1. / N * (sign_samples.T @ sign_samples)
    # cov = np.cov(sign_samples.T)
    cov = np.sin(np.pi * cov / 2.)
    w, v = LA.eig(cov)
    for i in range(w.shape[0]):
        w[i] = max(w[i], 1e-9)
    cov = v @ np.diag(w) @ LA.inv(v)
    if sign:
        if _lambda == None:
            raise Exception('_lambda must given sign error.')
        return sign_error(cov, Q_inv, _lambda)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def per_symbol_quantization_method(samples, Q_inv, r, _lambda=None):
    ground_graph = sparsity_pattern(Q_inv)
    quantized_samples = quantize(samples, r)
    N = samples.shape[0]
    cov = 1. / N * (quantized_samples.T @ quantized_samples)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def joint_method(samples, Q_inv, Hr, Hi, snr, sigma2, _lambda=None, sign=False):
    ground_graph = sparsity_pattern(Q_inv)
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
    # S_y = np.cov(y_samples.T)
    cov = H_inv @ (S_y - sigma2 * np.eye(2*n)) @ H_inv.T
    cov = (cov[:n, :n] + cov[n:, n:]) / 2.
    # np.fill_diagonal(cov, p / 2.)
    w, v = LA.eig(cov)
    for i in range(w.shape[0]):
        w[i] = max(w[i], 1e-9)
    cov = v @ np.diag(w) @ LA.inv(v)
    if sign:
        if _lambda == None:
            raise Exception('_lambda must given in sign error.')
        return sign_error(cov, Q_inv, _lambda)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def kendalltau_method(samples, Q_inv, _lambda=None, sign=False):
    ground_graph = sparsity_pattern(Q_inv)
    N, n = samples.shape
    cov = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            cov[i, j], _ = stats.kendalltau(samples[:, i], samples[:, j], method='asymptotic')
    cov = cov + cov.T
    for i in range(n):
        cov[i, i] = 1
    cov = np.sin(np.pi * cov / 2.)
    if sign:
        if _lambda == None:
            raise Exception('_lambda must given in sign error.')
        return sign_error(cov, Q_inv, _lambda)
    if _lambda == None:
        return best_error(cov, ground_graph)
    return error(cov, ground_graph, _lambda)

def sign_tree_error(samples, Q_inv):
    class union_find:
        def __init__(self, d):
            self.d = d
            self.p = [i for i in range(d)]
        def find(self, i):
            return i if self.p[i] == i else self.find(self.p[i])
        def union(self, i, j):
            x, y = self.find(i), self.find(j)
            self.p[x] = y
    ground_graph = sparsity_pattern(Q_inv)
    sign_samples = np.sign(samples)
    N, d = sign_samples.shape
    theta = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            for n in range(N):
                theta[i, j] += (1. / N) * (1 if sign_samples[n, i]*sign_samples[n, j] == 1 else 0)
    edges = [[np.abs(theta[i, j] - 0.5), i, j] for i in range(d) for j in range(i+1, d)]
    edges.sort(key=lambda x: x[0], reverse=True)
    ds = union_find(d)
    predicted_graph = np.zeros((d, d))
    for edge in edges:
        i, j = edge[1], edge[2]
        if ds.find(i) != ds.find(j):
            ds.union(i, j)
            predicted_graph[i, j] = 1
            predicted_graph[j, i] = 1
    fn, fp = diff(ground_graph, predicted_graph)
    return fn+fp, fn, fp
