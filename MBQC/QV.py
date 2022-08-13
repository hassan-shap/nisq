import matplotlib.pyplot as plt
from IPython.display import clear_output
from scipy.special import erf 
#Import Qiskit classes
import qiskit
from qiskit import assemble, transpile
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error, thermal_relaxation_error
from MBQC_basic_functions import*
#from mpi4py import MPI
#Import the qv function
import qiskit.ignis.verification.quantum_volume as qv


import warnings
warnings.filterwarnings('ignore')
def QV_experiment(σc2, σgkp2, σm2, k,ntrials):
    qubit_lists = [[0,1,3],[0,1,3,5],[0,1,3,5,7],[0,1,3,5,7,10],[0,1,3,5,7,10,12],[0,1,3,5,7,10,12,15],[0,1,3,5,7,10,12,15,17],[0,1,3,5,7,10,12,15,17,19],]
     ##[list(np.arange(i)*2) for i in range(3,10)]#[[0,1,3],[0,1,3,5],[0,1,3,5,7],[0,1,3,5,7,10],[0,1,3,5,7,10]]
    # ntrials: Number of random circuits to create for each subset

    qv_circs, qv_circs_nomeas = qv.qv_circuits(qubit_lists, ntrials)
    # pass the first trial of the nomeas through the transpiler to illustrate the circuit
    qv_circs_nomeas[0] = qiskit.compiler.transpile(qv_circs_nomeas[0], basis_gates=['u1','u2','u3','cx'])
    
    sv_sim = qiskit.Aer.get_backend('aer_simulator')
    ideal_results = []
    for trial in range(ntrials):
        clear_output(wait=True)
        for qc in qv_circs_nomeas[trial]:
            qc.save_statevector()
        result = qiskit.execute(qv_circs_nomeas[trial], backend=sv_sim).result()
        ideal_results.append(result)
    qv_fitter = qv.QVFitter(qubit_lists=qubit_lists)
    qv_fitter.add_statevectors(ideal_results)
    
    aer_sim = qiskit.Aer.get_backend('aer_simulator')
    basis_gates = ['u3','cx'] # use U,CX for now
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(pauli_error(error_model_gkp(σc2, σgkp2, σm2, k=k, gate_type = 'U3')), 'u3')
    noise_model.add_all_qubit_quantum_error(pauli_error(error_model_gkp(σc2, σgkp2, σm2, k=1, gate_type = 'CNOT')), 'cx')
    
    exp_results = []
    for trial in range(ntrials):
        clear_output(wait=True)
        t_qcs = transpile(qv_circs[trial], basis_gates=basis_gates, optimization_level=3)
        qobj = assemble(t_qcs)
        result = aer_sim.run(qobj, noise_model=noise_model, max_parallel_experiments=0).result()
        exp_results.append(result)
        print(f'Completed trial {trial+1}/{ntrials}')
    qv_fitter.add_data(exp_results)
    return qv_fitter
# setting the stage
dB = 15
σ2 = 10**(-1/10*dB)/2
k = 5
η = .95
σm2 = (1-η)/(2*η)
σc2 = 0
σgkp2 = σ2
ntrials = 20

# Plot the essence by calling plot_rb_data
#qv_fitter.plot_qv_data(ax=ax, show_plt=False)
qv_fitter = QV_experiment(σc2, σgkp2, σm2, k,ntrials)
print(qv_fitter.ydata[0],)
'''
qv_fitter = QV_experiment(σc2, σgkp2, σm2, k,ntrials)

hvy_avg_all = comm.gather(qv_fitter.ydata[0], root=0)
hvy_var_all = comm.gather(qv_fitter.ydata[1], root=0)

if rank == 0:
    mean = sum(hvy_avg_all)/len(hvy_avg_all)
    var = sum(hvy_var_all)/len(hvy_var_all)/np.sqrt(len(hvy_var_all))
    data = [mean, var]
    np.save(f'data/QV_raw_data_dB{dB}_σc_σgkp_η{η}_k{k}',data)'''
#ax.set_title('Quantum Volume for up to %d Qubits \n and %d Trials'%(len(qubit_lists[-1]), ntrials), fontsize=18)
#plt.savefig('QV_example.pdf')