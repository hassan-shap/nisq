import numpy as np
from math import pi, sqrt
import itertools
import time

repeat = 100
Nrep = 1000 # number of iterations
pc_list = np.linspace(0,0.5,50)

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

"""
X0----Y1----Y2----Y3----4
"""
def runner(i_rep):
    err_mc = np.zeros((16,len(pc_list)))
    tic = time.time()
    
    for i_p, p_c in enumerate(pc_list):
        p_m = p_c
        # p_c = 0
        weights_cz = np.concatenate(([1-14/15*p_c],np.ones(7)*2/15*p_c))
        weights_single_qubit = np.array([1-p_m,p_m/3,p_m/3,p_m/3]) # I, X, Z, Y

        for i_iter in range(Nrep):
            vec_x = np.zeros(5,dtype=int)
            vec_z = np.zeros(5,dtype=int)

            for i_l in range(4):
                i_1 = np.random.choice(8, 1, p=weights_cz)
                cz_err_gen(i_1,i_l,i_l+1,vec_x,vec_z)

                err_q = np.random.choice(4, 1, p=weights_single_qubit)
                vec_x[i_l] += (err_q%2)
                vec_z[i_l] += int(err_q/2)

            xcomp = (vec_z[0]+ vec_x[2] + vec_z[2] + vec_x[3] + vec_z[3] + vec_x[4] )%2
            zcomp = (vec_x[1] + vec_z[1] + vec_x[2] + vec_z[2] + vec_z[4] )%2

            err_mc[2*zcomp+xcomp,i_p] += 1

    err_mc /= Nrep

    toc = time.time()
    print("Finished r=%d in %.1f secs" % (i_rep,toc-tic))
    fname = "data_dv_H/" + "pm_eq_pc_i_%d.npz" % (i_rep)
    np.savez(fname, pc_list=pc_list, err_mc=err_mc, Nrep=Nrep)

    return 0

results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))
