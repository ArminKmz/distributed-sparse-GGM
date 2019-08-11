from scipy.io import loadmat
import plot_dimension, plot_sample, plot_pofe, plot_snr
import utils

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

def run_pofe_dimension(id, p, a1, a2, a3):
    Q_inv = utils.get_star(p, 0.25, 50)
    N_list = [100*i for i in range(1,60)]
    plot_pofe.generate_and_save_plot_data(N_list, 100, Q_inv, 'dimension-'+str(id), a1, a2, a3)
    plot_pofe.plot('dimension-'+str(id))

def run_pofe_dimension_chain(id, p, a1, a2, a3):
    Q_inv = utils.get_chain(p, 0.25)
    N_list = [100*i for i in range(1,60)]
    plot_pofe.generate_and_save_plot_data(N_list, 100, Q_inv, 'dimension-'+str(id), a1, a2, a3)
    plot_pofe.plot('dimension-'+str(id))

def run_pofe_dimension_grid(id, p, a1, a2, a3):
    Q_inv = utils.get_grid(p, 0.1)
    N_list = [100*i for i in range(30, 200)]
    plot_pofe.generate_and_save_plot_data(N_list, 100, Q_inv, 'dimension-'+str(id), a1, a2, a3)
    plot_pofe.plot('dimension-'+str(id))

def run_pofe_degree(id, d, a1, a2, a3):
    Q_inv = utils.get_star(128, 10, d, True)
    N_list = [100*i for i in range(1,150)]
    plot_pofe.generate_and_save_plot_data(N_list, 100, Q_inv, 'degree-'+str(id), a1, a2, a3)
    plot_pofe.plot('degree-'+str(id))

def run_pofe_H(id, Hr, Hi, a):
    Q_inv = utils.get_star(128, 0.2)
    N_list = [100*i for i in range(1,100)]
    plot_pofe.generate_and_save_plot_data_H(N_list, 100, Q_inv, 'H-'+str(id), a, Hr, Hi)
    # plot_pofe.plot_H('H-'+str(id))


def run_1_snr():
    mat = loadmat('cov_generator/random_covs.mat')
    Q_inv = mat.get('Qinv_40_1')
    snr_list = [.5*i for i in range(1, 100, 5)]
    plot_snr.generate_and_save_plot_data(snr_list, 100, 10*1000, Q_inv, '1')
    plot_snr.plot('1')

#---------------------------------
run_1_p()
run_2_p()
# run_3_p()
#---------------------------------
# run_1_n()
run_2_n()
run_3_n()
#---------------------------------
# a1, a2, a3 = 3.1, 4.5, 0.75
# run_pofe_dimension(1, 64, a1, a2, a3)
# run_pofe_dimension(2, 128, a1, a2, a3)
# run_pofe_dimension(3, 256, a1, a2, a3)
#---------------------------------
# run_pofe_degree(1, 40, a1, a2, a3)
# run_pofe_degree(2, 50, a1, a2, a3)
# run_pofe_degree(3, 60, a1, a2, a3)
#---------------------------------
# import numpy as np
# from numpy import linalg as LA
# p  = 128
# Hr = np.random.normal(0, 1, (p, p))
# Hi = np.random.normal(0, 1, (p, p))
# H = np.zeros((2*p, 2*p))
# H[:p, :p], H[p:, p:] = Hr, Hr
# H[:p, p:], H[p:, :p] = -Hi, Hi
# min_eig = np.sqrt(np.min(np.absolute(LA.eigvals(H @ H.T))))
# max_eig = np.sqrt(np.max(np.absolute(LA.eigvals(H @ H.T))))
# print('min_sv:', min_eig * np.sqrt(.3/2))
# print('max_sv:', max_eig * np.sqrt(.3/2))
# a3 = 0.8
# # run_pofe_H(1, Hr, Hi, a3)
#
# cof = 5
#
# from numpy.linalg import eig, inv, svd
# u, s, vh = svd(H, full_matrices=True)
# for i in range(s.shape[0]):
#     if s[i] < cof * min_eig:
#         s[i] *= cof
# H = np.real(u * s @ vh)
# min_eig = np.sqrt(np.min(np.absolute(LA.eigvals(H @ H.T))))
# max_eig = np.sqrt(np.max(np.absolute(LA.eigvals(H @ H.T))))
# print('min_sv:', min_eig * np.sqrt(.3/2))
# print('max_sv:', max_eig * np.sqrt(.3/2))
# # H[:p, :p] is same as H[:p, :p]
# # H[:p, p:] is same as -H[p:, :p]
# Hr = H[:p, :p]
# Hi = H[p:, :p]
# a3 = 0.75
# # run_pofe_H(2, Hr, Hi, a3)
#
# u, s, vh = svd(H, full_matrices=True)
# for i in range(s.shape[0]):
#     if s[i] < cof * min_eig:
#         s[i] *= cof
# H = np.real(u * s @ vh)
# # H[:p, :p] is same as H[:p, :p]
# # H[:p, p:] is same as -H[p:, :p]
# Hr = H[:p, :p]
# Hi = H[p:, :p]
# min_eig = np.sqrt(np.min(np.absolute(LA.eigvals(H @ H.T))))
# max_eig = np.sqrt(np.max(np.absolute(LA.eigvals(H @ H.T))))
# print('min_sv:', min_eig * np.sqrt(.3/2))
# print('max_sv:', max_eig * np.sqrt(.3/2))
# a3 = 0.7
# run_pofe_H(3, Hr, Hi, a3)
#---------------------------------
# run_1_snr()
#---------------------------------
# a1, a2, a3 = 2.3, 3.6, .6
# run_pofe_dimension_chain(4, 64, a1, a2, a3)
# run_pofe_dimension_chain(5, 128, a1, a2, a3)
# run_pofe_dimension_chain(6, 256, a1, a2, a3)

# a1, a2, a3 = 3, 3.8, 0.6
# run_pofe_dimension_grid(4, 5, a1, a2, a3)
# run_pofe_dimension_grid(5, 6, a1, a2, a3)
# run_pofe_dimension_grid(6, 7, a1, a2, a3)
