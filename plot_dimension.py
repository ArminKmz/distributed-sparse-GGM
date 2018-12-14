import numpy as np
import numpy.linalg as LA
import utils
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

ORIGINAL_METHOD = 0
SIGN_METHOD = 1
JOINT_METHOD = 2
methods = [ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD]

def generate_and_save_plot_data(N, K, mat, run_id):
    '''
        mat should contains sparse precision
        matrix with following format as key:
        'Q_inv_{p}_{id}'
    '''
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, methods

    # reading cov names
    keys = list(mat.keys())
    name_list = []
    tmp = []
    last = -1
    for i in range(len(keys)):
        key = keys[i]
        if len(key) <= 4 or key[:4] != 'Qinv':
            continue
        idx1 = key.find('_')
        idx2 = key.find('_', idx1+1)
        p = int(key[idx1+1:idx2])
        id = int(key[idx2+1:])
        if last not in [p, -1]:
            name_list.append(tmp)
            tmp = []
        tmp.append(key)
        last = p
        if i == len(keys) - 1:
            name_list.append(tmp)
    # name_list[i] -> graphs with p vertices

    p_list      = np.zeros(len(name_list), dtype='int32')
    e_list      = np.zeros(len(name_list))
    d_list      = np.zeros(len(name_list))
    fpr_list    = np.zeros((len(name_list), len(methods)))
    fnr_list    = np.zeros((len(name_list), len(methods)))
    lambda_list = np.zeros((len(name_list), len(methods)))

    for i in range(len(name_list)):
        p_list[i] = mat.get(name_list[i][0]).todense().shape[0]
        e_avg = 0
        d_avg = 0
        fnr_avg = np.zeros(len(methods))
        fpr_avg = np.zeros(len(methods))
        lambda_avg = np.zeros(len(methods))
        for j in range(len(name_list[i])):
            Q_inv = mat.get(name_list[i][j]).todense()
            Q = LA.inv(Q_inv)
            p = Q.shape[0]
            edges = utils.edges(Q_inv)
            e_avg += edges
            non_edges = (p * (p - 1) / 2) - edges
            graph = utils.sparsity_pattern(Q_inv)
            d_avg += utils.get_max_degree(graph)
            for k in range(K):
                samples = np.random.multivariate_normal(np.zeros(p), Q, N)
                fn      = np.zeros(len(methods))
                fp      = np.zeros(len(methods))
                _lambda = np.zeros(len(methods))
                error, fn[ORIGINAL_METHOD], fp[ORIGINAL_METHOD], _lambda[ORIGINAL_METHOD] = utils.original_data(samples, graph)
                error, fn[SIGN_METHOD],     fp[SIGN_METHOD],     _lambda[SIGN_METHOD]     = utils.sign_method(samples, graph)
                error, fn[JOINT_METHOD],    fp[JOINT_METHOD],    _lambda[JOINT_METHOD]    = utils.joint_method(samples, graph, np.eye(p), np.zeros((p, p)), 3,  .1)
                for method in methods:
                    fpr_avg[method]    += fp[method] / (non_edges + .0)
                    fnr_avg[method]    += fn[method] / (edges + .0)
                    lambda_avg[method] += _lambda[method]
        e_list[i] = e_avg / (len(name_list[i]) + .0)
        d_list[i] = d_avg / (len(name_list[i]) + .0)
        denom = (len(name_list[i]) * K + .0)
        for method in methods:
            fpr_list[i, method]    = fpr_avg[method]    / denom
            fnr_list[i, method]    = fnr_avg[method]    / denom
            lambda_list[i, method] = lambda_avg[method] / denom
        print('dimension', p_list[i], 'done.')

    if not os.path.exists('./data/plot_dimension/run_{0}'.format(run_id)):
        print('directory ./data/plot_dimension/run_{0} created.'.format(run_id))
        os.makedirs('./data/plot_dimension/run_{0}'.format(run_id))
    else:
        for f in os.listdir('./data/plot_dimension/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_dimension/run_{0}/{1}'.format(run_id, f))

    np.savetxt('data/plot_dimension/run_{0}/N.txt'.format(run_id), [N], fmt='%d')
    np.savetxt('data/plot_dimension/run_{0}/p_list.txt'.format(run_id), p_list, fmt='%d')
    np.savetxt('data/plot_dimension/run_{0}/d_list.txt'.format(run_id), d_list)
    np.savetxt('data/plot_dimension/run_{0}/e_list.txt'.format(run_id), e_list)
    np.savetxt('data/plot_dimension/run_{0}/fnr_list.txt'.format(run_id), fnr_list)
    np.savetxt('data/plot_dimension/run_{0}/fpr_list.txt'.format(run_id), fpr_list)
    np.savetxt('data/plot_dimension/run_{0}/lambda_list.txt'.format(run_id), lambda_list)
    print('data saved to ./data/plot_dimension/run_{0}.'.format(run_id))

def plot(run_id):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, methods

    N           = np.loadtxt('data/plot_dimension/run_{0}/N.txt'.format(run_id))
    p_list      = np.loadtxt('data/plot_dimension/run_{0}/p_list.txt'.format(run_id))
    d_list      = np.loadtxt('data/plot_dimension/run_{0}/d_list.txt'.format(run_id))
    e_list      = np.loadtxt('data/plot_dimension/run_{0}/e_list.txt'.format(run_id))
    fnr_list    = np.loadtxt('data/plot_dimension/run_{0}/fnr_list.txt'.format(run_id))
    fpr_list    = np.loadtxt('data/plot_dimension/run_{0}/fpr_list.txt'.format(run_id))
    lambda_list = np.loadtxt('data/plot_dimension/run_{0}/lambda_list.txt'.format(run_id))

    red_patch   = mpatches.Patch(color='r', label='Original')
    blue_patch  = mpatches.Patch(color='b', label='Sign')
    joint_patch = mpatches.Patch(color='g', label='Joint')

    plt.plot(p_list, d_list, 'ko-')
    plt.xlabel('p')
    plt.ylabel('d')
    plt.show()
    #-------
    plt.suptitle('N =' + str(N))
    plt.plot(p_list, fnr_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(p_list, fnr_list[:, SIGN_METHOD], 'bo-')
    plt.plot(p_list, fnr_list[:, JOINT_METHOD], 'go-')
    plt.xlabel('p')
    plt.ylabel('False negative rate')
    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.show()
    #-------
    plt.suptitle('N =' + str(N))
    plt.plot(p_list, fpr_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(p_list, fpr_list[:, SIGN_METHOD], 'bo-')
    plt.plot(p_list, fpr_list[:, JOINT_METHOD], 'go-')
    plt.xlabel('p')
    plt.ylabel('False positive rate')
    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.show()
    #-------
    plt.suptitle('N =' + str(N))
    plt.plot(p_list, lambda_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(p_list, lambda_list[:, SIGN_METHOD], 'bo-')
    plt.plot(p_list, lambda_list[:, JOINT_METHOD], 'go-')
    plt.xlabel('p')
    plt.ylabel('lambda')
    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.show()
