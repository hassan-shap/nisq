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

def covariance_matrix(σgkp,σc,σm,hadmard=True,N=20):
    Σ0 = σgkp* np.eye(2*N)
    Σ1 = np.copy(Σ0)
    for i_c in range(N-1):
        i2_c = i_c+1
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
        Σ1 = S_c.dot(Σ1.dot(S_c.T)) + Σc
    if hadmard:
        # Hadamard gate X1 Y2 Y3 Y4
        q1 = int(N/2)
        q2 = q1+1
        q3 = q1+2
        q4 = q1+3
        Um = np.array([[1,1],[1,-1]])/2**0.5
        Umat = np.eye(2*N)
        Umat[np.ix_([q2,q2+N],[q2,q2+N])] = Um
        Umat[np.ix_([q3,q3+N],[q3,q3+N])] = Um
        Umat[np.ix_([q4,q4+N],[q4,q4+N])] = Um
        Σ1_new = Umat.dot(Σ1.dot(Umat.T))
        return Σ1_new[np.ix_([q1+N,q2+N,q3+N,q4+N],[q1+N,q2+N,q3+N,q4+N])] + σm*np.eye(4)
    else:
        # Phase gate X1 X2 Y3 X4
        q1 = int(N/2)
        q2 = q1+1
        q3 = q1+2
        q4 = q1+3
        Um = np.array([[1,1],[1,-1]])/2**0.5
        Umat = np.eye(2*N)
        Umat[np.ix_([q3,q3+N],[q3,q3+N])] = Um
        Σ1_new = Umat.dot(Σ1.dot(Umat.T))
        return Σ1_new[np.ix_([q1+N,q2+N,q3+N,q4+N],[q1+N,q2+N,q3+N,q4+N])] + σm*np.eye(4)


for i_n, η in enumerate(η_list):
    σm2 = (1-η)/(2*η)
    
    print("Loss= %.2f" % (η))
    def runner(i_rep):
        probX_mc = np.zeros(len(σ2_list))
        probY_mc = np.zeros(len(σ2_list))
        probZ_mc = np.zeros(len(σ2_list))

        
        tic = time.time()
        for i_s, σgkp2 in enumerate(σ2_list):
            σc2 = 0*σgkp2
            Σ = covariance_matrix(σgkp2,σc2,σm2)
            rand_vec = np.random.multivariate_normal(np.zeros(4),Σ,Nrep)
            rand_comp = np.abs(rand_vec) > (pi**0.5/2)
            vx = np.sum(rand_comp[:,[0,2,3]],axis=1)%2
            vz = np.sum(rand_comp[:,[1,2]],axis=1)%2
            probX_mc[i_s] = np.sum(vx*(1-vz))/Nrep
            probZ_mc[i_s] = np.sum(vz*(1-vx))/Nrep
            probY_mc[i_s] = np.sum(vx*vz)/Nrep
            # print(i_s)
            
        toc = time.time()
        print("Finished Loss= %.2f, r=%d in %.1f secs" % (η,i_rep,toc-tic))
        # fname = "data_hadamard/" + "sc_eq_sgkp_p_%.2f_i_%d.npz" % (η,i_rep)
        fname = "data_hadamard/" + "sc_0_p_%.2f_i_%d.npz" % (η,i_rep)
        np.savez(fname, σ2_list=σ2_list, probX_mc=probX_mc, probY_mc=probY_mc, probZ_mc=probZ_mc, Nrep=Nrep)

        return 0
    
    results = Parallel(n_jobs=num_cores)(delayed(runner)(i_rep) for i_rep in range(repeat))



