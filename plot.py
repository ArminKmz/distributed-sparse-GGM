import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ORIGINAL_METHOD = 0
SIGN_METHOD = 1
JOINT_METHOD = 2
methods = [ORIGINAL_METHOD, SIGN_METHOD, JOINT_METHOD]

N_list  = np.loadtxt('data/plot_pofe/run_dimension-1/N_list.txt')
ps_64_list = np.loadtxt('data/plot_pofe/run_dimension-1/ps_list.txt')
ps_128_list = np.loadtxt('data/plot_pofe/run_dimension-2/ps_list.txt')
ps_256_list = np.loadtxt('data/plot_pofe/run_dimension-3/ps_list.txt')

red_patch = mpatches.Patch(color='r', label='Original')
blue_patch = mpatches.Patch(color='b', label='Sign')
joint_patch = mpatches.Patch(color='g', label='Joint')

o = mlines.Line2D([], [], marker='o', linestyle='None', label='p=64')
x = mlines.Line2D([], [], marker='x', linestyle='None', label='p=128')
t = mlines.Line2D([], [], marker='^', linestyle='None', label='p=256')

plt.legend(handles=[red_patch, blue_patch, joint_patch, o, x, t])


plt.plot(N_list / 6, ps_64_list[:, ORIGINAL_METHOD], 'ro-')
plt.plot(N_list / 6, ps_64_list[:, SIGN_METHOD], 'bo-')
plt.plot(N_list / 6, ps_64_list[:, JOINT_METHOD], 'go-')

plt.plot(N_list / 7, ps_128_list[:, ORIGINAL_METHOD], 'rx-')
plt.plot(N_list / 7, ps_128_list[:, SIGN_METHOD], 'bx-')
plt.plot(N_list / 7, ps_128_list[:, JOINT_METHOD], 'gx-')

plt.plot(N_list / 8, ps_256_list[:, ORIGINAL_METHOD], 'r^-')
plt.plot(N_list / 8, ps_256_list[:, SIGN_METHOD], 'b^-')
plt.plot(N_list / 8, ps_256_list[:, JOINT_METHOD], 'g^-')

plt.xlabel('num of samples / log p')
plt.ylabel('prob of success')
plt.show()
