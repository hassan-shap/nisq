# -*- coding: utf-8 -*-
"""
Author: Yuxuan Zhang 
Date: Jun. 15 2022
Just a naive simulator to study MBQC noise model with Qiskit
"""

# Qiskit simulator for MBQC gates
import qiskit
import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sqrt
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import Aer
from scipy.stats import unitary_group
from scipy.special import erf 
# u2(0,np.pi/2) = sdg() * h()
# u2(0,np.pi) = h()
def count_to_fid(counts):
    ct_valid = 0
    ct_total = 0
    for key in counts.keys():
        if key[0] == '0':
            ct_valid += counts[key]
        ct_total += counts[key]
    return ct_valid/ct_total

#erorr model - measurement error only
def get_noise(p):
    error_meas = pauli_error([('X',p/3),('Z',p/3),('Y',p/3), ('I', 1 - p)]) #isotropic error model

    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas,'unitary') # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_meas,'u2') 
    return noise_model

def arbitrary_measurement(a):
    uni = np.array([[1,np.exp(1j*a)],[1,-1*np.exp(1j*a)]])/np.sqrt(2)
    return uni

def linear_cluster(size, random_ini = True):
    q= QuantumRegister(size)
    c= ClassicalRegister(size)
    circ= QuantumCircuit(q, c)
    
    #Initializing; apply the generated random unitary if desired
    if random_ini ==True:
        rand_u = unitary_group.rvs(2) #random unitary that rotates the first qubit in the graph
        rand_u_inv = np.conj(rand_u.T) #inverse of the above line
        ru = Operator(rand_u)
        rui = Operator(rand_u_inv)
        circ.unitary(rand_u, q[0], label='rand_u')
        
    for i in range (1,size):
        circ.h(q[i])
    for i in range (size-1):
        circ.cz(q[i],q[i+1])
    if random_ini ==True:
        return circ, rand_u_inv
    else: 
        return circ
def linear_cluster_with_ancilla(size, random_ini = True): #with an ancilla to record some extra operations
    q= QuantumRegister(size+1)
    c= ClassicalRegister(size+1)
    circ= QuantumCircuit(q, c)
    
    #Initializing; apply the generated random unitary if desired
    if random_ini ==True:
        rand_u = unitary_group.rvs(2) #random unitary that rotates the first qubit in the graph
        rand_u_inv = np.conj(rand_u.T) #inverse of the above line
        ru = Operator(rand_u)
        rui = Operator(rand_u_inv)
        circ.unitary(rand_u, q[0], label='rand_u')
        
    for i in range (1,size):
        circ.h(q[i])
    for i in range (size-1):
        circ.cz(q[i],q[i+1])
    if random_ini ==True:
        return circ, rand_u_inv
    else: 
        return circ    
    
def benchmarking_RZ(theta,shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    noise_model = get_noise(p)
    circ, rand_u_inv= linear_cluster(5,random_ini = True)
    q, c = circ.qubits, circ.clbits

    #Teleportation Measurements
    circ.u2(0,np.pi,q[0])#rotating into x-basis
    circ.measure(q[0],c[0])
    circ.u2(0,np.pi,q[1])#rotating into x-basis
    circ.measure(q[1],c[1])
    #adaptively choose measurement base for the third qubit
    circ.unitary(Operator(arbitrary_measurement(-1*theta)),q[2],label='unitary').c_if(c[1], 1)
    circ.unitary(Operator(arbitrary_measurement(theta)),q[2],label='unitary').c_if(c[1], 0)
    circ.measure(q[2],c[2])
    circ.u2(0,np.pi,q[3])#rotating into x-basis
    circ.measure(q[3],c[3])
    for i in [1,3]:
        circ.x(q[-1]).c_if(c[i], 1)
    for i in [0,2]:
        circ.z(q[-1]).c_if(c[i], 1)
    
    circ.rz(-1*theta,q[-1])
    circ.unitary(rand_u_inv, q[-1], label='rand_u_inv')
    circ.measure(q[-1],c[-1])
    # Run the quantum circuit on a simulator backend
    simulator = Aer.get_backend('aer_simulator')
    # Create a Quantum Program for execution
    job = simulator.run(circ)

    result = simulator.run(circ,noise_model=noise_model,shot= shots).result()
    counts = result.get_counts(circ)
    return count_to_fid(counts)


def benchmarking_H(shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    noise_model = get_noise(p)
    circ, rand_u_inv= linear_cluster(5,random_ini = True)
    q, c = circ.qubits, circ.clbits
    
    #Teleportation Measurements
    circ.u2(0,np.pi,q[0])#rotating into x-basis
    for qubit in range (1,4):
        circ.u2(0,np.pi/2,q[qubit])#rotating into y-basis
    circ.measure(q[:-1],c[:-1])
    for i in [1,2]:
        circ.z(q[-1]).c_if(c[i], 1)
    for i in [0,2,3]:
        circ.x(q[-1]).c_if(c[i], 1)
    
    circ.h(q[-1])
    circ.unitary(rand_u_inv, q[-1], label='rand_u_inv')
    circ.measure(q[-1],c[-1])
    # Run the quantum circuit on a simulator backend
    simulator = Aer.get_backend('aer_simulator')
    # Create a Quantum Program for execution
    job = simulator.run(circ)

    result = simulator.run(circ,noise_model=noise_model,shot= shots).result()
    counts = result.get_counts(circ)
    return count_to_fid(counts)

def benchmarking_S(shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    noise_model = get_noise(p)
    circ, rand_u_inv = linear_cluster(5,random_ini = True)
    q, c = circ.qubits, circ.clbits
    
    #Teleportation Measurements
    for qubit in [0,1,3]:
        circ.u2(0,np.pi,q[qubit])#rotating into x-basis
    circ.u2(0,np.pi/2,q[2])
    circ.measure(q[:-1],c[:-1])
    
    #correction rules
    for i in [0,1,2]:
        circ.z(q[-1]).c_if(c[i], 1)
    for i in [1,3]:
        circ.x(q[-1]).c_if(c[i], 1)
    circ.z(q[-1])
    
    circ.sdg(q[-1])
    circ.unitary(rand_u_inv, q[-1], label='rand_u_inv')
    circ.measure(q[-1],c[-1])
    # Run the quantum circuit on a simulator backend
    simulator = Aer.get_backend('aer_simulator')
    # Create a Quantum Program for execution
    job = simulator.run(circ)

    result = simulator.run(circ,noise_model=noise_model,shot= shots).result()
    counts = result.get_counts(circ)
    return count_to_fid(counts)

def benchmarking_I(shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    noise_model = get_noise(p)
    circ, rand_u_inv = linear_cluster(3,random_ini = True)
    q, c = circ.qubits, circ.clbits
      
    for i in range (1,3):
        circ.h(q[i])
    for i in range (3-1):
        circ.cz(q[i],q[i+1])        
        
    circ.u2(0,np.pi,q[0])
    circ.u2(0,np.pi,q[1])
    circ.measure(q[:-1],c[:-1])
    
    circ.z(q[-1]).c_if(c[0], 1)
    circ.x(q[-1]).c_if(c[1], 1)
    
    circ.unitary(rand_u_inv, q[-1], label='rand_u_inv')
    circ.measure(q[-1],c[-1])

    # Run the quantum circuit on a simulator backend
    simulator = Aer.get_backend('aer_simulator')
    # Create a Quantum Program for execution
    job = simulator.run(circ)

    result = simulator.run(circ,noise_model=noise_model,shot= shots).result()
    counts = result.get_counts(circ)

    return count_to_fid(counts)

def benchmarking_Rotation(angles,shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    [t1,t2,t3] = angles
    noise_model = get_noise(p)
    circ, rand_u_inv= linear_cluster_with_ancilla(5,random_ini = True)
    q, c = circ.qubits, circ.clbits
    #a clever trick that combines classical measurement results
    
    #Teleportation Measurements
    circ.u2(0,np.pi,q[0])#rotating into x-basis
    circ.measure(q[0],c[0])
    
    #adaptively choose measurement base for the second qubit
    circ.unitary(Operator(arbitrary_measurement(-1*t1)),q[1],label='unitary').c_if(c[0], 1)
    circ.unitary(Operator(arbitrary_measurement(t1)),q[1],label='unitary').c_if(c[0], 0)
    #circ.u2(0,np.pi,q[1])
    circ.measure(q[1],c[1])
    
    #adaptively choose measurement base for the third qubit
    circ.unitary(Operator(arbitrary_measurement(-1*t2)),q[2],label='unitary').c_if(c[1], 1)
    circ.unitary(Operator(arbitrary_measurement(t2)),q[2],label='unitary').c_if(c[1], 0)
    #circ.u2(0,np.pi,q[2])
    circ.measure(q[2],c[2])
    
    
    circ.x(q[-1]).c_if(c[0], 1)
    circ.x(q[-1]).c_if(c[2], 1)
    circ.measure(q[-1],c[-1])
    #adaptively choose measurement base for the fourth qubit
    circ.unitary(Operator(arbitrary_measurement(-1*t3)),q[3],label='unitary').c_if(c[-1], 1)
    circ.unitary(Operator(arbitrary_measurement(t3)),q[3],label='unitary').c_if(c[-1], 0)
    #circ.u2(0,np.pi,q[3])
    circ.measure(q[3],c[3])
    
    for i in [1,3]:
        circ.x(q[4]).c_if(c[i], 1)
    for i in [0,2]:
        circ.z(q[4]).c_if(c[i], 1)
    
    #circ.u3(-1*t1,-1*t3,-1*t2,q[-1])
    circ.rx(-1*t3,q[4])
    circ.rz(-1*t2,q[4])
    circ.rx(-1*t1,q[4])
    circ.unitary(rand_u_inv, q[4], label='rand_u_inv')
    circ.measure(q[4],c[4])
    
    # Run the quantum circuit on a simulator backend
    simulator = Aer.get_backend('aer_simulator')
    # Create a Quantum Program for execution
    job = simulator.run(circ)
    
    result = simulator.run(circ,noise_model=noise_model,shot= shots).result()
    counts = result.get_counts(circ)
    old_keys = list(counts.keys())
    #get rid of the ancilla registers...
    for i in range (len(list(counts.keys()))):    
        counts[old_keys[i][1:]] = counts.pop(old_keys[i])
    count_to_fid(counts)    
    return count_to_fid(counts)
def error_rules(p, error_type, theta = 0):
    #analytically calculated fidelity rules
    p_flip = 2*p/3
    if error_type == 'H' or error_type == 'S':
        p00 = (1-p_flip)**4 + 2*p_flip**3 * (1-p_flip) + p_flip**2 * (1-p_flip)**2 
        p01 = 2*p_flip * (1-p_flip)**3 + p_flip**2 * (1-p_flip)**2 + p_flip**4
        p10 = p_flip * (1-p_flip)**3 + 2* p_flip**2 * (1-p_flip)**2 + p_flip**3 * (1-p_flip)
        p11 = p_flip * (1-p_flip)**3 + 2* p_flip**2 * (1-p_flip)**2 + p_flip**3 * (1-p_flip)
        F_avg = 0.5 + (3*p00-p01-p10-p11)/6 
        
    elif error_type == 'I':
        p00 = (1-p_flip)**2
        p01 = (1-p_flip)*p_flip
        p10 = (1-p_flip)*p_flip
        p11 = p_flip**2
        F_avg = 0.5 + (3*p00-p01-p10-p11)/6 
        
    elif error_type == 'RZ':
        p00 = (1-p_flip)**4  + p_flip**2 * (1-p_flip)**2 
        q00 = p_flip**4 + p_flip**2 * (1-p_flip)**2 
        p01 = 2*p_flip * (1-p_flip)**3 
        q01 = 2*p_flip**3 * (1-p_flip)
        p10 = 2*p_flip * (1-p_flip)**3 + 2*p_flip**3 * (1-p_flip)
        p11 = 4* p_flip**2 * (1-p_flip)**2 
        F_avg = 0.5 + (3*p00 + (1+2*np.cos(2*theta))*q00-p01-p10-p11 + (1-2*np.cos(2*theta))*q01)/6 
    
    return F_avg
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

def CNOT_error(p):
    p_flip = p*2/3
    #output a error model for the CNOT gate (in the nearest neighbor setting)
    Nq = 14
    vec = Hilbertspace_Zr(Nq,2)
    err_vec = np.zeros((16,Nq+1),dtype=int)
    for i in range(2**Nq):
        s_flip=vec[i,:]
        sz_c = np.sum(vec[i,[0,2,3,4,7,8,10]])
        sx_c = np.sum(vec[i,[1,2,4,5]])
        sz_t = np.sum(vec[i,[8,10,12]])
        sx_t = np.sum(vec[i,[1,2,7,9,11,13]])
        p_exp = np.sum(vec[i,:])
        
        i_e = np.array([sz_c%2,sx_c%2,sz_t%2,sx_t%2])@ (2**np.arange(4))
        err_vec[i_e,p_exp] += 1
        
    p_vec = p_flip**np.arange(15)*(1-p_flip)**np.arange(14,-1,-1)
    err_prob = err_vec @ p_vec
    pauli_vec = Hilbertspace_Zr(4,2)
    pauli_combined = []
    for i in range(16):
        p_st = ''
        p1 = pauli_vec[i,:2]@ np.array([1,2])
        if  p1 == 0:
            p_st += 'I'
        elif  p1 == 1:
            p_st += 'Z'
        elif  p1 == 2:
            p_st += 'X'
        else:
            p_st += 'Y'
        p2 = pauli_vec[i,2:]@ np.array([1,2])
        if  p2 == 0:
            p_st += 'I'
        elif  p2 == 1:
            p_st += 'Z'
        elif  p2 == 2:
            p_st += 'X'
        else:
            p_st += 'Y'
    
        pauli_combined.append((p_st,err_prob[i]))
    return pauli_combined

def success_prob_calc(Δ):
    return (erf( (pi**0.5)/(2*Δ) )- erf(-(pi**0.5)/(2*Δ) )  )/2

def u3_error_gkp(σc2, σgkp2, σm2,k):

    σin = (2**0.5) * (k*(2*σgkp2 + 17/6*σc2))**0.5
    σout = (2**0.5) * (σgkp2 + σm2)**0.5
    
    σn = (2**0.5) * (3*σgkp2+ 11/3*σc2 + σm2)**0.5
    Xerr = 1-success_prob_calc(σn)
    σn = (2**0.5) * ( (3*σgkp2+ 11/3*σc2+ σgkp2+ 2*σc2)/2 + σm2)**0.5
    Yerr = 1-success_prob_calc(σn)
    #check the factor
    Nerr = 2*(1-success_prob_calc(σin))/3 + (1-success_prob_calc(σout))
    
    prob_X = 2*Nerr*(1-Nerr)**2*(1-Xerr)+2*Nerr**2*(1-Nerr)*(Xerr)
    prob_Z = Nerr*(1-Nerr)**2*(1-Xerr)+ Xerr*(1-Nerr)**2*(1-Nerr)+(1-Xerr)*Nerr**3+Xerr*(1-Nerr)*Nerr**2
    prob_Y = 2*Nerr*Xerr*(1-Nerr)**2 + 2*Nerr*Nerr*(1-Xerr)* (1-Nerr)
    total_error = prob_X + prob_Z + prob_Y
    
    return [('I',1 - total_error), ('X',prob_X),('Y',prob_Y),('Z',prob_Z)]


def cnot_error_gkp(σc2, σgkp2, σm2):
    #Calculate the raw error rates
    Nq = 13
    vec = Hilbertspace_Zr(Nq,2)

    # qubits measured in X or Y basis
    X_list = [0,7,8,9,11,12]
    Y_list = [1,2,3,4,5,6,10]
    Nx = len(X_list)+1
    Ny = len(Y_list)+1

    err_vec = np.zeros((16,Ny*Nx),dtype=int)
    for i in range(2**Nq):
        s_flip=vec[i,:]

        sz_c = np.sum(vec[i,[0,2,3,4,6,7,9]])
        sx_c = np.sum(vec[i,[1,2,4,5]])
        sz_t = np.sum(vec[i,[7,9,11]])
        sx_t = np.sum(vec[i,[1,2,6,8,10,12]])
        px_exp = np.sum(vec[i,X_list])
        py_exp = np.sum(vec[i,Y_list])

        i_e = np.array([sz_c%2,sx_c%2,sz_t%2,sx_t%2])@ (2**np.arange(4))
        err_vec[i_e,py_exp*Nx + px_exp] += 1

    err_prob = np.zeros(16)

    σn = (2**0.5) * ( 3*σgkp2+ 11/3*σc2 + σm2)**0.5
    Xerr = 1-success_prob_calc(σn)
    px_vec = Xerr**np.arange(Nx)*(1-Xerr)**np.arange(Nx-1,-1,-1)
    σn = (2**0.5) * ( (3*σgkp2+ 11/3*σc2+ σgkp2+ 2*σc2)/2 + σm2)**0.5
    Yerr = 1-success_prob_calc(σn)
    py_vec = Yerr**np.arange(Ny)*(1-Yerr)**np.arange(Ny-1,-1,-1)
    p_vec = np.kron(py_vec,px_vec)
    err_prob = err_vec @ p_vec    
    
    err_list={0: 'II', 1: 'ZI', 2: 'XI', 3: 'YI', 4: 'IZ', 5: 'ZZ', 6: 'XZ',\
          7: 'YZ', 8: 'IX', 9: 'ZX', 10: 'XX', 11: 'YX', 12: 'IY',\
          13: 'ZY', 14: 'XY', 15: 'YY'}
    err_model = []
    for i in range (16):
        err_model.append((err_list[i],err_prob[i]))
    
    return err_model


def error_model_gkp(σc2, σgkp2, σm2, k=1, gate_type = 'U3'): 
    #a wrapper function that, given a Gaussian noise distribution, returns a noise channel
    '''
    σgkp2: GKP error
    σc2: CZ error 
    σm2: measurement error
    '''
    if gate_type == 'U3':
        error_model = u3_error_gkp(σc2, σgkp2, σm2, k)
        
    elif gate_type == 'CNOT':
        error_model = cnot_error_gkp(σc2, σgkp2, σm2)
    else:
        raise NotImplementedError('Shit, the requested gate type is not implemented')
    return error_model
    