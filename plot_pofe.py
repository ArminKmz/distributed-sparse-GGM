import numpy as np
import numpy.linalg as LA
import utils
from scipy.io import loadmat
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(0)

ORIGINAL_METHOD = 0
SIGN_METHOD = 1
JOINT_METHOD = 2
KT_METHOD = 3
ST_METHOD = 4
methods = [ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, KT_METHOD, ST_METHOD]

def generate_and_save_plot_data(N_list, K, Q_inv, run_id, a1, a2, a3):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, methods

    Q = LA.inv(Q_inv)
    p = Q.shape[0]
    ps_list = np.zeros((len(N_list), len(methods)))
    for i in range(len(N_list)):
        N = N_list[i]
        for k in range(K):
            samples = np.random.multivariate_normal(np.zeros(p), Q, N)
            error = [0 for _ in range(len(methods))]
            l1 = a1 * np.sqrt(np.log(p) / N)
            l2 = a2 * np.sqrt(np.log(p) / N)
            l3 = a3 * np.sqrt(np.log(p) / N)
            error[ORIGINAL_METHOD], _, _ = utils.original_data(samples, Q_inv, l1, True)
            error[SIGN_METHOD],     _, _ = utils.sign_method(samples, Q_inv, l2, True)
            error[JOINT_METHOD],    _, _ = utils.joint_method(samples, Q_inv, np.eye(p), np.zeros((p, p)), 3,  .1, l3, True)

            for method in methods:
                if error[method] == 0:
                    ps_list[i, method] += (1 / K)
        print('#{0} done.'.format(N))

    if not os.path.exists('./data/plot_pofe/run_{0}'.format(run_id)):
        print('directory ./data/plot_pofe/run_{0} created.'.format(run_id))
        os.makedirs('./data/plot_pofe/run_{0}'.format(run_id))
    else:
        for f in os.listdir('./data/plot_pofe/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_pofe/run_{0}/{1}'.format(run_id, f))

    np.savetxt('data/plot_pofe/run_{0}/N_list.txt'.format(run_id), N_list)
    np.savetxt('data/plot_pofe/run_{0}/ps_list.txt'.format(run_id), ps_list)

def plot(run_id):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, methods

    N_list  = np.loadtxt('data/plot_pofe/run_{0}/N_list.txt'.format(run_id))
    ps_list = np.loadtxt('data/plot_pofe/run_{0}/ps_list.txt'.format(run_id))

    red_patch = mpatches.Patch(color='r', label='Original')
    blue_patch = mpatches.Patch(color='b', label='Sign')
    joint_patch = mpatches.Patch(color='g', label='Joint')

    plt.legend(handles=[red_patch, blue_patch, joint_patch])
    plt.plot(N_list, ps_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(N_list, ps_list[:, SIGN_METHOD], 'bo-')
    plt.plot(N_list, ps_list[:, JOINT_METHOD], 'go-')
    plt.xlabel('num of samples')
    plt.ylabel('prob of success')
    plt.show()




def generate_and_save_plot_data_cmp(N_list, K, Q_inv, run_id, a1, a2, a3, a4):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, KT_METHOD, ST_METHOD, methods

    Q = LA.inv(Q_inv)
    p = Q.shape[0]
    ps_list = np.zeros((len(N_list), len(methods)))
    flags = [0 for m in methods]
    for i in range(len(N_list)):
        N = N_list[i]
        for k in range(K):
            samples = np.random.multivariate_normal(np.zeros(p), Q, N)
            error = [1 for _ in range(len(methods))]
            l1 = a1 * np.sqrt(np.log(p) / N)
            l2 = a2 * np.sqrt(np.log(p) / N)
            l3 = a3 * np.sqrt(np.log(p) / N)
            l4 = a4 * np.sqrt(np.log(p) / N)
            if not flags[ORIGINAL_METHOD]: error[ORIGINAL_METHOD], _, _ = utils.original_data(samples, Q_inv, l1, False)
            else: error[ORIGINAL_METHOD] = 0
            if not flags[SIGN_METHOD]: error[SIGN_METHOD],     _, _ = utils.sign_method(samples, Q_inv, l2, False)
            else: error[SIGN_METHOD] = 0
            if not flags[JOINT_METHOD]: error[JOINT_METHOD],    _, _ = utils.joint_method(samples, Q_inv, np.eye(p), np.zeros((p, p)), 3,  .1, l3, False)
            else: error[JOINT_METHOD] = 0
            if not flags[KT_METHOD]: error[KT_METHOD],       _, _ = utils.kendalltau_method(samples, Q_inv, l4, False)
            else: error[KT_METHOD] = 0
            if not flags[ST_METHOD]: error[ST_METHOD],       _, _ = utils.sign_tree_error(samples, Q_inv)
            else: error[ST_METHOD] = 0

            for method in methods:
                if error[method] == 0:
                    ps_list[i, method] += (1 / K)
                if abs(ps_list[i, method] - 1) <= 0.01:
                    flags[method] = 1
        print('#{0} done.'.format(N))

    if not os.path.exists('./data/plot_pofe/run_{0}'.format(run_id)):
        print('directory ./data/plot_pofe/run_{0} created.'.format(run_id))
        os.makedirs('./data/plot_pofe/run_{0}'.format(run_id))
    else:
        for f in os.listdir('./data/plot_pofe/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_pofe/run_{0}/{1}'.format(run_id, f))

    np.savetxt('data/plot_pofe/run_{0}/N_list.txt'.format(run_id), N_list)
    np.savetxt('data/plot_pofe/run_{0}/ps_list.txt'.format(run_id), ps_list)

def plot_cmp(run_id):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, KT_METHOD, ST_METHOD, methods

    N_list  = np.loadtxt('data/plot_pofe/run_{0}/N_list.txt'.format(run_id))
    ps_list = np.loadtxt('data/plot_pofe/run_{0}/ps_list.txt'.format(run_id))

    red_patch = mpatches.Patch(color='r', label='Original')
    blue_patch = mpatches.Patch(color='b', label='Sign')
    joint_patch = mpatches.Patch(color='g', label='Joint')
    st_patch = mpatches.Patch(color='y', label='sign tree')
    kt_patch = mpatches.Patch(color='k', label='kendal tau')

    plt.legend(handles=[red_patch, blue_patch, joint_patch, st_patch, kt_patch])
    plt.plot(N_list, ps_list[:, ORIGINAL_METHOD], 'ro-')
    plt.plot(N_list, ps_list[:, SIGN_METHOD], 'bo-')
    plt.plot(N_list, ps_list[:, JOINT_METHOD], 'go-')
    plt.plot(N_list, ps_list[:, KT_METHOD], 'ko-')
    plt.plot(N_list, ps_list[:, ST_METHOD], 'yo-')
    plt.xlabel('num of samples')
    plt.ylabel('prob of success')
    plt.show()
