import utils
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

ORIGINAL_METHOD = 0
SIGN_METHOD = 1
JOINT_METHOD = 2
KT_METHOD = 3
methods = [ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, KT_METHOD]

def generate_and_save_plot_data(B_list, K, mat, names, run_id):
    '''
        Q_inv -> sparse precision matrix.
    '''
    global SIGN_METHOD, JOINT_METHOD, methods

    fpr_list    = np.zeros((len(B_list), len(methods)))
    fnr_list    = np.zeros((len(B_list), len(methods)))
    lambda_list = np.zeros((len(B_list), len(methods)))

    for i in range(len(B_list)):
        B = B_list[i]
        B_over_logB = int(B / np.log2(B))
        fnr_avg    = np.zeros(len(methods))
        fpr_avg    = np.zeros(len(methods))
        lambda_avg = np.zeros(len(methods))
        for j in range(len(names)):
            Q_inv = mat.get(names[j]).todense()
            Q = LA.inv(Q_inv)
            p = Q_inv.shape[0]
            edges = utils.edges(Q_inv)
            non_edges = (p * (p - 1) / 2) - edges
            for k in range(K):
                samples1 = np.random.multivariate_normal(np.zeros(p), Q, B)
                samples2 = np.random.multivariate_normal(np.zeros(p), Q, B_over_logB)
                fn      = np.zeros(len(methods))
                fp      = np.zeros(len(methods))
                _lambda = np.zeros(len(methods))
                error, fn[SIGN_METHOD],     fp[SIGN_METHOD],     _lambda[SIGN_METHOD]     = utils.sign_method(samples1, Q_inv)
                print('sign')
                error, fn[JOINT_METHOD],    fp[JOINT_METHOD],    _lambda[JOINT_METHOD]    = utils.joint_method(samples1, Q_inv, np.eye(p), np.zeros((p, p)), 3,  .1)
                print('joint')
                error, fn[KT_METHOD],       fp[KT_METHOD],       _lambda[KT_METHOD]       = utils.kendalltau_method(samples2, Q_inv)
                print('kendal')
                for method in methods:
                    fpr_avg[method]    += fp[method] / (non_edges + .0)
                    fnr_avg[method]    += fn[method] / (edges + .0)
                    lambda_avg[method] += _lambda[method]
        denom = (len(names) * K + .0)
        for method in methods:
            fpr_list[i, method]    = fpr_avg[method]    / denom
            fnr_list[i, method]    = fnr_avg[method]    / denom
            lambda_list[i, method] = lambda_avg[method] / denom
        print('#'+str(B), 'done.')

    if not os.path.exists('./data/plot_bit/run_{0}'.format(run_id)):
        print('directory ./data/plot_bit/run_{0} created.'.format(run_id))
        os.makedirs('./data/plot_bit/run_{0}'.format(run_id))
    else:
        for f in os.listdir('./data/plot_bit/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_bit/run_{0}/{1}'.format(run_id, f))

    np.savetxt('data/plot_bit/run_{0}/B_list.txt'.format(run_id), B_list)
    np.savetxt('data/plot_bit/run_{0}/fnr_list.txt'.format(run_id), fnr_list)
    np.savetxt('data/plot_bit/run_{0}/fpr_list.txt'.format(run_id), fpr_list)
    np.savetxt('data/plot_bit/run_{0}/lambda_list.txt'.format(run_id), lambda_list)
    print('data saved to ./data/plot_bit/run_{0}.'.format(run_id))

def plot(run_id):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, KT_METHOD, methods

    B_list      = np.loadtxt('data/plot_bit/run_{0}/B_list.txt'.format(run_id))
    fnr_list    = np.loadtxt('data/plot_bit/run_{0}/fnr_list.txt'.format(run_id))
    fpr_list    = np.loadtxt('data/plot_bit/run_{0}/fpr_list.txt'.format(run_id))
    lambda_list = np.loadtxt('data/plot_bit/run_{0}/lambda_list.txt'.format(run_id))

    red_patch   = mpatches.Patch(color='r', label='Original')
    blue_patch  = mpatches.Patch(color='b', label='Sign')
    joint_patch = mpatches.Patch(color='g', label='Joint')
    kt_patch    = mpatches.Patch(color='y', label='KT')

    plt.plot(B_list, fnr_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(B_list, fnr_list[:, SIGN_METHOD], 'bo-')
    plt.plot(B_list, fnr_list[:, JOINT_METHOD], 'go-')
    plt.plot(B_list, fnr_list[:, KT_METHOD], 'yo-')
    plt.xlabel('Number of bits per dimension')
    plt.ylabel('False negative rate')
    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.show()
    #-------
    plt.plot(B_list, fpr_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(B_list, fpr_list[:, SIGN_METHOD], 'bo-')
    plt.plot(B_list, fpr_list[:, JOINT_METHOD], 'go-')
    plt.plot(B_list, fpr_list[:, KT_METHOD], 'yo-')
    plt.xlabel('Number of bits per dimension')
    plt.ylabel('False positive rate')
    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.show()
    #-------
    plt.plot(B_list, lambda_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(B_list, lambda_list[:, SIGN_METHOD], 'bo-')
    plt.plot(B_list, lambda_list[:, JOINT_METHOD], 'go-')
    plt.plot(B_list, lambda_list[:, KT_METHOD], 'yo-')
    plt.xlabel('Number of samples')
    plt.ylabel('lambda')
    plt.legend(handles=[red_patch, blue_patch, joint_patch, kt_patch])
    plt.show()
