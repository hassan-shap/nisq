#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 14:47:41 2022

@author: yuxuzhan
"""

# Qiskit simulator for MBQC gates
import qiskit
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit,QuantumRegister,ClassicalRegister
from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import Aer
from scipy.stats import unitary_group

# u2(0,np.pi/2) = sdg() * h()
# u2(0,np.pi) = h()
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
    #noise_model = get_noise(p)
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

    result = simulator.run(circ,noise_model=get_noise(p),shot= shots).result()
    counts = result.get_counts(circ)
    return count_to_fid(counts)

def benchmarking_I(shots,p = 0): # the benchmarking circuit for a single random input state vector instance
    noise_model = get_noise(p)
    circ, rand_u_inv = linear_cluster(3,random_ini = True)
    q, c = circ.qubits, circ.clbits
      
        
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

# post processing the data: from output data to infidelity 
# the last qubit should be 0 if a teleportation gate is successful
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
    p2 = p/15 
    #p1 = 1e-10
    #error_meas = pauli_error([('X',p1/3),('Z',p1/3),('Y',p1/3), ('I', 1 - p1)]) #isotropic error model
    error_cz = pauli_error([('IX',p2),('IZ',p2),('IY',p2),('XI',p2),('ZI',p2),('YI',p2), ('XX',p2),('XY',p2),('XZ',p2),('YX',p2),('YY',p2),('YZ',p2),('ZX',p2),('ZY',p2),('ZZ',p2),('II', 1 - p)]) #isotropic error model

    noise_model = NoiseModel()
    #noise_model.add_all_qubit_quantum_error(error_meas,'unitary') # measurement error is applied to measurements
    #noise_model.add_all_qubit_quantum_error(error_meas,'u2')
    noise_model.add_all_qubit_quantum_error(error_cz,'cz') 

    return noise_model

def error_rules(p_cz, p_m, error_type, theta = 0):
    #analytically calculated fidelity rules
    p_flip = 2*(p_m)/3+10*p_cz/15
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
#noise_model = get_noise(p)
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

result = simulator.run(circ,noise_model=get_noise(p),shot= shots).result()
counts = result.get_counts(circ)
count_to_fid(counts)
        