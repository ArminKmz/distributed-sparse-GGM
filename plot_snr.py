import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA
import utils

def generate_and_save_plot_data(snr_list, K, N, Q_inv):
    Q_inv = Q_inv.todense()
    Q = LA.inv(Q_inv)
    p = Q_inv.shape[0]
    edges = utils.edges(Q_inv)
    non_edges = (p * (p - 1) / 2) - edges
    graph = utils.sparsity_pattern(Q_inv)
    samples = np.random.multivariate_normal(np.zeros(p), Q, N)

    Hi_list = np.random.normal(0, .5, (p, p))
    Hr_list = np.random.normal(0, .5, (p, p))

    fpr_list    = np.zeros((len(name_list), len(methods)))
    fnr_list    = np.zeros((len(name_list), len(methods)))

    for i in range(len(snr_list)):
        snr = snr_list[i]



def plot():
    pass
