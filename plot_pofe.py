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

def generate_and_save_plot_data(N_list, K, Q_inv, run_id):
    global ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD, methods

    Q = LA.inv(Q_inv)
    p = Q.shape[0]
    graph = utils.sparsity_pattern(Q_inv)
    ps_list = np.zeros((len(N_list), len(methods)))
    for i in range(len(N_list)):
        N = N_list[i]
        prob_of_succ = [0 for _ in range(len(methods))]
        for k in range(K):
            samples = np.random.multivariate_normal(np.zeros(p), Q, N)
            error = [0 for _ in range(len(methods))]
            error[ORIGINAL_METHOD], _, _, _ = utils.original_data(samples, graph)
            error[SIGN_METHOD],     _, _, _ = utils.sign_method(samples, graph)
            error[JOINT_METHOD],    _, _, _ = utils.joint_method(samples, graph, np.eye(p), np.zeros((p, p)), 3,  .1)
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
