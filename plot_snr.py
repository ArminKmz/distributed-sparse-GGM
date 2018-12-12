import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import utils
from scipy.io import loadmat

def generate_and_save_plot_data(snr_list, K, N, Q_inv, run_id):
    Q_inv = Q_inv.todense()
    Q = LA.inv(Q_inv)
    p = Q_inv.shape[0]
    edges = utils.edges(Q_inv)
    non_edges = (p * (p - 1) / 2) - edges
    graph = utils.sparsity_pattern(Q_inv)
    samples = np.random.multivariate_normal(np.zeros(p), Q, N)

    Hi_list = np.random.normal(0, .5, (K, p, p))
    Hr_list = np.random.normal(0, .5, (K, p, p))

    fpr_list    = np.zeros(len(snr_list))
    fnr_list    = np.zeros(len(snr_list))

    for i in range(len(snr_list)):
        snr = snr_list[i]
        fnr_avg = 0
        fpr_avg = 0
        for k in range(K):
            Hr = Hr_list[k, :, :]
            Hi = Hi_list[k, :, :]
            # Hr = np.eye(p)
            # Hi = np.eye(p)
            error, fn, fp, _ = utils.joint_method(samples, graph, Hr, Hi, snr, .1)
            fnr_avg += fn / (edges + .0)
            fpr_avg += fp / (non_edges + .0)
        fnr_avg /= (K + .0)
        fpr_avg /= (K + .0)
        print('snr {0} done.'.format(snr))


    if not os.path.exists('./data/plot_snr/run_{0}'.format(run_id)):
        print('directory ./data/plot_snr/run_{0} created.'.format(run_id))
        os.makedirs('./data/plot_snr/run_{0}'.format(run_id))
    else:
        for f in os.listdir('./data/plot_snr/run_{0}'.format(run_id)):
            print('{0} removed.'.format(f))
            os.remove('./data/plot_snr/run_{0}/{1}'.format(run_id, f))

    np.savetxt('/data/plot_snr/run_{0}/snr_list.txt'.format(run_id), snr_list)
    np.savetxt('/data/plot_snr/run_{0}/fnr_list.txt'.format(run_id), fnr_list)
    np.savetxt('/data/plot_snr/run_{0}/fpr_list.txt'.format(run_id), fpr_list)

def plot(run_id):
    snr_list = np.loadtxt('/data/plot_snr/run_{0}/snr_list.txt'.format(run_id))
    fnr_list = np.loadtxt('/data/plot_snr/run_{0}/fnr_list.txt'.format(run_id))
    fpr_list = np.loadtxt('/data/plot_snr/run_{0}/fpr_list.txt'.format(run_id))

    plt.plot(snr_list, fnr_list, 'go-')
    plt.xlabel('SNR')
    plt.ylabel('False negative rate')
    plt.show()
    #-----
    plt.plot(snr_list, fpr_list, 'go-')
    plt.xlabel('SNR')
    plt.ylabel('False negative rate')
    plt.show()

mat = loadmat('cov_generator/random_covs.mat')
Q_inv = mat.get('Qinv_50_4')
snr_list = [.5*i for i in range(1, 11)]
generate_and_save_plot_data(snr_list, 20, 10*1000, Q_inv, '1')
plot('1')
