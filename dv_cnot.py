import numpy as np
from math import pi, sqrt
import itertools
import time

repeat = 24
Nrep = 10 # number of iterations
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

## for errors in measurement apparatus
def Hilbertspace_Zr(N,r):
    states=np.zeros((r**N,N),dtype=int)
    if N>0:
        for i_1 in range(r**N):
            num_str=np.base_repr(i_1,base=r)[::-1]
            for i_2 in range(len(num_str)):
                states[i_1,i_2]=int(num_str[i_2])
    else:
        states=[[0]]
        
    return states
Nq = 14
vec = Hilbertspace_Zr(Nq,2)
###

def runner(i_rep):
    err_mc = np.zeros((16,len(pc_list)))
    tic = time.time()
    
    for i_p, p_c in enumerate(pc_list):
        weights = np.concatenate(([1-14/15*p_c],np.ones(7)*2/15*p_c))
        p_flip = 2/3*p_c
        p_vec = p_flip**np.arange(15)*(1-p_flip)**np.arange(14,-1,-1)

        for i_iter in range(Nrep):
            vec_x = np.zeros(15,dtype=int)
            vec_z = np.zeros(15,dtype=int)

            for i_l in range(14):
                i_1 = np.random.choice(8, 1, p=weights)
                if i_l <= 5 or i_l >= 8: # lower/upper rung
                    cz_err_gen(i_1,i_l,i_l+1,vec_x,vec_z)
                elif i_l == 6: # 3-7
                    cz37_err_gen(i_1,vec_x,vec_z)
                elif i_l == 7: # 7-11
                    cz_err_gen(i_1,7,11,vec_x,vec_z)

            sz_c = np.sum(vec_x[Xinds_z_c]) + np.sum(vec_z[Zinds_z_c])
            sx_c = np.sum(vec_x[inds_x_c]) + np.sum(vec_z[inds_x_c])
            sz_t = np.sum(vec_z[inds_z_t])
            sx_t = np.sum(vec_x[Xinds_x_t]) + np.sum(vec_z[Zinds_x_t])
                    
            for i in range(2**Nq):
                s_flip=vec[i,:]    
                sz_c += np.sum(vec[i,Zinds_z_c])
                sx_c += np.sum(vec[i,inds_x_c])
                sz_t += np.sum(vec[i,inds_z_t])
                sx_t += np.sum(vec[i,Zinds_x_t])
                p_exp = np.sum(vec[i,:])

                i_e = np.array([sz_c%2,sx_c%2,sz_t%2,sx_t%2])@ (2**np.arange(4))
                err_mc[i_e,i_p] += p_vec[p_exp]

    err_mc /= Nrep

    toc = time.time()
    print("Finished r=%d in %.1f secs" % (i_rep,toc-tic))
    # fname = "data_dv_cnot/" + "pm_%.2f_i_%d.npz" % (pm,i_rep)
    fname = "data_dv_cnot/" + "pm_eq_pc_i_%d.npz" % (i_rep)
    np.savez(fname, pc_list=pc_list, err_mc=err_mc, Nrep=Nrep)

    return 0

results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))
