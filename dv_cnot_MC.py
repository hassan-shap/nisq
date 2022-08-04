import numpy as np
from math import pi, sqrt
import itertools
import time

repeat = 24
Nrep = 1000 # number of iterations
pc_list = np.linspace(0,0.5,50)
pm = 0 # measuerment error rate

from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     

def cz_err_gen(err,i_c,i_t,v_x,v_z):
    if err == 1:
        v_x[i_c] += 1
    elif err == 2:
        v_x[i_c] += 1
        v_z[i_c] += 1
    elif err == 3:
        v_z[i_c] += 1
    elif err == 4:
        v_z[i_t] += 1
    elif err == 5:
        v_x[i_c] += 1
        v_z[i_t] += 1
    elif err == 6:
        v_x[i_c] += 1
        v_z[[i_c,i_t]] += 1
    elif err == 7:
        v_z[[i_c,i_t]] += 1
    return 0
def cz37_err_gen(err,v_x,v_z):
    if err == 1:
        v_z[[2,7]] += 1
    elif err == 2:
        v_z[[2,3,7]] += 1
    elif err == 3:
        v_z[3] += 1
    elif err == 4:
        v_z[7] += 1
    elif err == 5:
        v_z[2] += 1
    elif err == 6:
        v_z[[2,3]] += 1
    elif err == 7:
        v_z[[3,7]] += 1
    return 0

"""
X0----Y1----Y2----Y3----Y4-----Y5-----6
                  |
                  Y7
                  |
X8----X9---X10---Y11----X12----X13-----14
"""

Xinds_z_c = [2,3,4,7]
Zinds_z_c = [0,2,3,4,7,8,10]
inds_x_c = [1,2,4,5]
inds_z_t = [8,10,12]
Xinds_x_t = [1,2,7,11]
Zinds_x_t = [1,2,7,9,11,13]

def runner(i_rep):
    err_mc = np.zeros((16,len(pc_list)))
    tic = time.time()
    
    for i_p, p_c in enumerate(pc_list):
        p_m = p_c
        weights_cz = np.concatenate(([1-14/15*p_c],np.ones(7)*2/15*p_c))
        weights_single_qubit = np.array([1-p_m,p_m/3,p_m/3,p_m/3]) # I, X, Z, Y

        for i_iter in range(Nrep):
            vec_x = np.zeros(15,dtype=int)
            vec_z = np.zeros(15,dtype=int)

            for i_l in range(14):
                err_cz = np.random.choice(8, 1, p=weights_cz)
                if i_l <= 5 or i_l >= 8: # lower/upper rung
                    cz_err_gen(err_cz,i_l,i_l+1,vec_x,vec_z)
                elif i_l == 6: # 3-7
                    cz37_err_gen(err_cz,vec_x,vec_z)
                elif i_l == 7: # 7-11
                    cz_err_gen(err_cz,7,11,vec_x,vec_z)

                err_q = np.random.choice(4, 1, p=weights_single_qubit)
                vec_x[i_l] += (err_q%2)
                vec_z[i_l] += int(err_q/2)

            sz_c = np.sum(vec_x[Xinds_z_c]) + np.sum(vec_z[Zinds_z_c])
            sx_c = np.sum(vec_x[inds_x_c]) + np.sum(vec_z[inds_x_c])
            sz_t = np.sum(vec_z[inds_z_t])
            sx_t = np.sum(vec_x[Xinds_x_t]) + np.sum(vec_z[Zinds_x_t])

            i_e = np.array([sz_c%2,sx_c%2,sz_t%2,sx_t%2])@ (2**np.arange(4))
            err_mc[i_e,i_p] += 1

    err_mc /= Nrep

    toc = time.time()
    print("Finished r=%d in %.1f secs" % (i_rep,toc-tic))
    # fname = "data_dv_cnot/" + "pm_%.2f_i_%d.npz" % (pm,i_rep)
    fname = "data_dv_cnot/" + "pm_eq_pc_i_%d_mc.npz" % (i_rep)
    np.savez(fname, pc_list=pc_list, err_mc=err_mc, Nrep=Nrep)

    return 0

results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))
