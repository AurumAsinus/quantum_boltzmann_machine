import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# params
num_sys = 3
num_anc = 2
num_total = num_sys + num_anc
num_layers = 5 

# W state
w_state = np.zeros(8)
w_state[1] = 1/np.sqrt(3) 
w_state[2] = 1/np.sqrt(3)
w_state[4] = 1/np.sqrt(3)
superpos_state = (w_state + np.array([0,0,0,0,0,0,0,1])) / np.sqrt(2)

qc_data = QuantumCircuit(num_sys)
qc_data.initialize(superpos_state, qc_data.qubits)
rho_target = DensityMatrix.from_instruction(qc_data)

# purification ansatz
def purification_ansatz(params):
    qc = QuantumCircuit(num_total)
    param_shape = (num_layers, num_total, 3)
    params = np.reshape(params, param_shape)
    for layer in range(num_layers):
        for q in range(num_total):
            qc.rx(params[layer, q, 0], q)
            qc.ry(params[layer, q, 1], q)
            qc.rz(params[layer, q, 2], q)
        for i in range(num_total - 1):
            qc.cx(i, i + 1)
    return qc

# extract reduced density matrix from purification
def get_reduced_dm(qc):
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    state = result.get_statevector()
    dm = DensityMatrix(state)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
    return reduced_dm

# loss calculated through fidelity
def loss(params):
    qc = purification_ansatz(params)
    rho_model = get_reduced_dm(qc)
    fid = state_fidelity(rho_target, rho_model)
    print(f"Fidelity: {fid:.6f}")
    return 1 - fid

# param initialization
param_count = num_layers * num_total * 3
init_params = np.random.uniform(0, 2 * np.pi, param_count)
# optimization
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 700, 'disp': True})
opt_params = res.x

#learnt state
final_qc = purification_ansatz(opt_params)
rho_learned = get_reduced_dm(final_qc)

# target vs learnt plot
def plot_dm_bars(dm, title, ax):
    probs = np.real(np.diag(dm.data))
    labels = [format(i, f'0{num_sys}b') for i in range(2**num_sys)]
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plot_dm_bars(rho_target, "Target Superposition Distribution", axs[0])
plot_dm_bars(rho_learned, "Learned QBM Distribution", axs[1])
plt.tight_layout()
plt.show()

print("\nTarget Density Matrix:\n", rho_target.data)
print("\nLearned Density Matrix:\n", rho_learned.data)
print(f"\nFinal Fidelity: {state_fidelity(rho_target, rho_learned):.6f}")
