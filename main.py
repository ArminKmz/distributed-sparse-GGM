from scipy.io import loadmat
import plot_dimension, plot_sample, plot_pofe, plot_snr

def run_1_p():
    mat = loadmat('cov_generator/random_covs.mat')
    plot_dimension.generate_and_save_plot_data(1000, 10, mat, '1')
    plot_dimension.plot('1')

def run_2_p():
    mat = loadmat('cov_generator/random_covs.mat')
    plot_dimension.generate_and_save_plot_data(10*1000, 10, mat, '2')
    plot_dimension.plot('2')


def run_3_p():
    mat = loadmat('cov_generator/random_covs.mat')
    plot_dimension.generate_and_save_plot_data(100*1000, 10, mat, '3')
    plot_dimension.plot('3')

def run_1_n():
    mat = loadmat('cov_generator/random_covs.mat')
    names = ['Qinv_20_{0}'.format(j) for j in range(1, 21)]
    l = [5000 * (i) for i in range(1, 21)]
    plot_sample.generate_and_save_plot_data(l, 10, mat, names, '1')
    plot_sample.plot('1')

def run_2_n():
    mat = loadmat('cov_generator/random_covs.mat')
    names = ['Qinv_50_{0}'.format(j) for j in range(1, 21)]
    l = [5000*i for i in range(1, 21)]
    plot_sample.generate_and_save_plot_data(l, 10, mat, names, '2')
    plot_sample.plot('2')

def run_3_n():
    mat = loadmat('cov_generator/random_covs.mat')
    names = ['Qinv_100_{0}'.format(j) for j in range(1, 21)]
    l = [5000*i for i in range(1, 21)]
    plot_sample.generate_and_save_plot_data(l, 10, mat, names, '3')
    plot_sample.plot('3')

def run_1_pofe():
    mat = loadmat('cov_generator/star.mat')
    Q_inv = mat.get('Q_inv')
    N_list = [100*i for i in range(1,31)]
    plot_pofe.generate_and_save_plot_data(N_list, 1, Q_inv, '1')
    plot_pofe.plot('1')

def run_1_snr():
    mat = loadmat('cov_generator/random_covs.mat')
    Q_inv = mat.get('Qinv_40_1')
    snr_list = [.5*i for i in range(1, 51)]
    plot_snr.generate_and_save_plot_data(snr_list, 100, 10*1000, Q_inv, '1')
    plot_snr.plot('1')


# run_1_p()
# run_2_p()
# run_3_p()
# run_1_n()
# run_2_n()
# run_3_n()
# run_1_pofe()
# run_1_snr()
