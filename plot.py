import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ORIGINAL_METHOD = 0
SIGN_METHOD = 1
JOINT_METHOD = 2
methods = [ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD]

# N_list  = np.loadtxt('data/plot_pofe/run_dimension-1/N_list.txt')
# ps_64_list = np.loadtxt('data/plot_pofe/run_dimension-1/ps_list.txt')
# ps_128_list = np.loadtxt('data/plot_pofe/run_dimension-2/ps_list.txt')
# ps_256_list = np.loadtxt('data/plot_pofe/run_dimension-3/ps_list.txt')

# N_list = np.loadtxt('data/plot_pofe/run_degree-1/N_list.txt')
# ps_40_list = np.loadtxt('data/plot_pofe/run_degree-1/ps_list.txt')
# ps_50_list = np.loadtxt('data/plot_pofe/run_degree-2/ps_list.txt')
# ps_60_list = np.loadtxt('data/plot_pofe/run_degree-3/ps_list.txt')

N_list = np.loadtxt('data/plot_pofe/run_H-1/N_list.txt')
ps_H1_list = np.loadtxt('data/plot_pofe/run_H-1/ps_list.txt')
ps_H2_list = np.loadtxt('data/plot_pofe/run_H-2/ps_list.txt')
ps_H3_list = np.loadtxt('data/plot_pofe/run_H-3/ps_list.txt')


# red_patch = mpatches.Patch(color='r', label='Original')
# blue_patch = mpatches.Patch(color='b', label='Sign')
joint_patch = mpatches.Patch(color='g', label='Joint')

# o = mlines.Line2D([], [], marker='o', linestyle='None', label='p=64')
# x = mlines.Line2D([], [], marker='x', linestyle='None', label='p=128')
# t = mlines.Line2D([], [], marker='^', linestyle='None', label='p=256')

# o = mlines.Line2D([], [], marker='o', linestyle='None', label='d=40')
# x = mlines.Line2D([], [], marker='x', linestyle='None', label='d=50')
# t = mlines.Line2D([], [], marker='^', linestyle='None', label='d=60')

o = mlines.Line2D([], [], marker='o', linestyle='None', label='H1')
x = mlines.Line2D([], [], marker='x', linestyle='None', label='H2')
t = mlines.Line2D([], [], marker='^', linestyle='None', label='H3')

# plt.legend(handles=[red_patch, blue_patch, joint_patch, o, x, t])
plt.legend(handles=[joint_patch, o, x, t])

# plt.plot(N_list/40, ps_40_list[:, ORIGINAL_METHOD], 'ro-')
# plt.plot(N_list/40, ps_40_list[:, SIGN_METHOD], 'bo-')
# plt.plot(N_list/40, ps_40_list[:, JOINT_METHOD], 'go-')
#
# plt.plot(N_list/50, ps_50_list[:, ORIGINAL_METHOD], 'rx-')
# plt.plot(N_list/50, ps_50_list[:, SIGN_METHOD], 'bx-')
# plt.plot(N_list/50, ps_50_list[:, JOINT_METHOD], 'gx-')
#
# plt.plot(N_list/60, ps_60_list[:, ORIGINAL_METHOD], 'r^-')
# plt.plot(N_list/60, ps_60_list[:, SIGN_METHOD], 'b^-')
# plt.plot(N_list/60, ps_60_list[:, JOINT_METHOD], 'g^-')

# plt.plot(N_list, ps_H1_list, 'go-')
# plt.plot(N_list, ps_H2_list, 'gx-')
# plt.plot(N_list, ps_H3_list, 'g^-')

def f(x):
    return 1
    return (1 + x)

plt.plot(N_list / f(65.98373400525072), ps_H1_list, 'go-')
plt.plot(N_list / f(65.96995446935013), ps_H2_list, 'gx-')
plt.plot(N_list / f(64.71554128192355), ps_H3_list, 'g^-')

plt.xlabel('num of samples')
plt.ylabel('prob of success')
plt.show()
