import numpy as np
from math import pi, sqrt
import itertools
import time

repeat = 24
Nrep = 100000 # number of iterations

η_list = np.array([0.75,0.85,0.95,1.0])
σ2_list = np.logspace(-3,-0.5,20)/2

from joblib import Parallel, delayed
import multiprocessing
# what are your inputs, and what operation do you want to
# perform on each input. For example...
num_cores = 12#multiprocessing.cpu_count()                                     

def noisy_cz_gate(σc,Σ1,i_c,i2_c):
    # i2_c = i_c+1
    N = int(np.size(Σ1,1)/2)
    S_c = np.eye(2*N)
    S_c[i_c+N,i2_c] = 1
    S_c[i2_c+N,i_c] = 1
    Σc = np.zeros((2*N,2*N))
    Σc[i_c,i_c] = σc
    Σc[N+i_c,N+i_c] = 4*σc/3
    Σc[i2_c,i2_c] = σc
    Σc[N+i2_c,N+i2_c] = 4*σc/3
    Σc[i_c,i2_c+N] = σc/2
    Σc[i2_c+N,i_c] = σc/2
    Σc[i2_c,i_c+N] = σc/2
    Σc[i_c+N,i2_c] = σc/2
    return S_c.dot(Σ1.dot(S_c.T)) + Σc

def covariance_matrix_cnot(σgkp,σc,σm):
    N = 13*2 + 1
    Σ0 = σgkp* np.eye(2*N)
    Σ1 = np.copy(Σ0)

    # first line
    for i_c in range(6):
        Σ1 = noisy_cz_gate(σc,Σ1,i_c,i_c+1)
    ##vertical qubit
    Σ1 = noisy_cz_gate(σc,Σ1,6,13)
    for i_c in np.arange(6,12):
        Σ1 = noisy_cz_gate(σc,Σ1,i_c,i_c+1)
    for i_c in np.arange(14,20):
        Σ1 = noisy_cz_gate(σc,Σ1,i_c,i_c+1)
    Σ1 = noisy_cz_gate(σc,Σ1,13,20)
    # second line
    for i_c in np.arange(20,N-1):
        Σ1 = noisy_cz_gate(σc,Σ1,i_c,i_c+1)

    X_list = [3,17,18,19,21,22]
    Y_list = [4,5,6,7,8,13,20]
    inds = np.sort(np.array(Y_list+X_list))+N
    Um = np.array([[1,1],[1,-1]])/2**0.5
    Umat = np.eye(2*N)
    for qy in Y_list:
        Umat[np.ix_([qy,qy+N],[qy,qy+N])] = Um
    Σ1_new = Umat.dot(Σ1.dot(Umat.T))
    # print(Σ1_new[np.ix_([q1+N,q2+N,q3,q4+N],[q1+N,q2+N,q3,q4+N])]) 
    return Σ1_new[np.ix_(inds,inds)] + σm*np.eye(13)
    
for i_n, η in enumerate(η_list):
    σm2 = (1-η)/(2*η)
    
    print("Loss= %.2f" % (η))
    def runner(i_rep):
        XX = np.zeros(len(σ2_list))
        YX = np.zeros(len(σ2_list))
        ZX = np.zeros(len(σ2_list))
        err_mc = np.zeros((16,len(σ2_list)))
        
        tic = time.time()
        for i_s, σgkp2 in enumerate(σ2_list):
            σc2 = σgkp2
            Σ = covariance_matrix_cnot(σgkp2,σc2,σm2)
            rand_vec = np.random.multivariate_normal(np.zeros(len(Σ)),Σ,Nrep)
            rand_comp = np.abs(rand_vec) > (pi**0.5/2)
            vx_c = np.sum(rand_comp[:,[1,2,4,5]],axis=1)%2
            vx_t = np.sum(rand_comp[:,[1,2,6,8,10,12]],axis=1)%2
            vz_c = np.sum(rand_comp[:,[0,2,3,4,6,7,9]],axis=1)%2
            vz_t = np.sum(rand_comp[:,[7,9,11]],axis=1)%2
            # XX[i_s] = np.sum(vx_c*(1-vz_c)*vx_t*(1-vz_t))/Nrep
            # ZX[i_s] = np.sum(vz_c*(1-vx_c)*vx_t*(1-vz_t))/Nrep
            # YX[i_s] = np.sum(vx_c*vz_c*vx_t*(1-vz_t))/Nrep
            for sz_c in range(2):
                for sx_c in range(2):
                    for sz_t in range(2):
                        for sx_t in range(2):
                            i_e = np.array([sz_c,sx_c,sz_t,sx_t])@ (2**np.arange(4))
                            err_mc[i_e,i_s] = np.sum(np.abs((1-sz_c-vz_c)*(1-sx_c-vx_c)*(1-sz_t-vz_t)*(1-sx_t-vx_t)))/Nrep           
        toc = time.time()
        print("Finished Loss= %.2f, r=%d in %.1f secs" % (η,i_rep,toc-tic))
        fname = "data_cnot/" + "sc_eq_sgkp_p_%.2f_i_%d.npz" % (η,i_rep)
        # fname = "data_cnot/" + "sc_0_p_%.2f_i_%d.npz" % (η,i_rep)
        # np.savez(fname, σ2_list=σ2_list, XX=XX, YX=YX, ZX=ZX, Nrep=Nrep)
        np.savez(fname, σ2_list=σ2_list, err_mc=err_mc, Nrep=Nrep)
        
        return 0
    
    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))
