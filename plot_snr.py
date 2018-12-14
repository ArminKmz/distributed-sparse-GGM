import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import utils
from scipy.io import loadmat
import os
import matplotlib.patches as mpatches

def generate_and_save_plot_data(snr_list, K, N, Q_inv, run_id):
    Q_inv = Q_inv.todense()
    Q = LA.inv(Q_inv)
    p = Q_inv.shape[0]
    edges = utils.edges(Q_inv)
    non_edges = (p * (p - 1) / 2) - edges
    graph = utils.sparsity_pattern(Q_inv)
    samples = np.random.multivariate_normal(np.zeros(p), Q, N)

    Hi_list = np.random.normal(0, 1, (K, p, p))
    Hr_list = np.random.normal(0, 1, (K, p, p))

    _, fnr_org, fpr_org, _ = utils.original_data(samples, graph)
    fnr_org /= (edges + .0)
    fpr_org /= (non_edges + .0)

    fpr_list    = np.zeros(len(snr_list))
    fnr_list    = np.zeros(len(snr_list))
    lambda_list = np.zeros(len(snr_list))

    for i in range(len(snr_list)):
        snr = snr_list[i]
        fnr_avg = 0
        fpr_avg = 0
        lambda_avg = 0
        for k in range(K):
            Hr = Hr_list[k, :, :]
            Hi = Hi_list[k, :, :]
            error, fn, fp, _lambda = utils.joint_method(samples, graph, Hr, Hi, snr, .1)
            fnr_avg    += fn / (edges + .0)
            fpr_avg    += fp / (non_edges + .0)
            lambda_avg += _lambda
        fnr_list[i]    = fnr_avg    / (K + .0)
        fpr_list[i]    = fpr_avg    / (K + .0)
        lambda_list[i] = lambda_avg / (K + .0)
        print('snr {0} done.'.format(snr))

    if not os.path.exists('./data/plot_snr/run_{0}'.format(run_id)):
        os.makedirs('./data/plot_snr/run_{0}'.format(run_id))
        print('directory ./data/plot_snr/run_{0} created.'.format(run_id))
    else:
        for f in os.listdir('./data/plot_snr/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_snr/run_{0}/{1}'.format(run_id, f))

    np.savetxt('./data/plot_snr/run_{0}/snr_list.txt'.format(run_id), snr_list)
    np.savetxt('./data/plot_snr/run_{0}/fnr_list.txt'.format(run_id), fnr_list)
    np.savetxt('./data/plot_snr/run_{0}/fpr_list.txt'.format(run_id), fpr_list)
    np.savetxt('./data/plot_snr/run_{0}/lambda_list.txt'.format(run_id), lambda_list)
    np.savetxt('./data/plot_snr/run_{0}/fnr_org.txt'.format(run_id), [fnr_org])
    np.savetxt('./data/plot_snr/run_{0}/fpr_org.txt'.format(run_id), [fpr_org])

def plot(run_id):
    snr_list    = np.loadtxt('./data/plot_snr/run_{0}/snr_list.txt'.format(run_id))
    fnr_list    = np.loadtxt('./data/plot_snr/run_{0}/fnr_list.txt'.format(run_id))
    fpr_list    = np.loadtxt('./data/plot_snr/run_{0}/fpr_list.txt'.format(run_id))
    lambda_list = np.loadtxt('./data/plot_snr/run_{0}/lambda_list.txt'.format(run_id))
    fnr_org     = np.loadtxt('./data/plot_snr/run_{0}/fnr_org.txt'.format(run_id))
    fpr_org     = np.loadtxt('./data/plot_snr/run_{0}/fpr_org.txt'.format(run_id))

    fnr_org_list = np.ones(snr_list.shape[0]) * fnr_org
    fpr_org_list = np.ones(snr_list.shape[0]) * fpr_org

    red_patch   = mpatches.Patch(color='r', label='Original')
    green_patch = mpatches.Patch(color='g', label='Joint')

    plt.plot(snr_list, fnr_org_list, 'ro-')
    plt.plot(snr_list, fnr_list, 'go-')
    plt.xlabel('SNR')
    plt.ylabel('False negative rate')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    #-----
    plt.plot(snr_list, fpr_org_list, 'ro-')
    plt.plot(snr_list, fpr_list, 'go-')
    plt.xlabel('SNR')
    plt.ylabel('False positive rate')
    plt.legend(handles=[red_patch, green_patch])
    plt.show()
    #-----
    plt.plot(snr_list, lambda_list, 'go-')
    plt.xlabel('SNR')
    plt.ylabel('lambda')
    plt.show()
