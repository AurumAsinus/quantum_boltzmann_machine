import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# hyperparameters
num_sys = 3
num_anc = 3
num_total = num_sys + num_anc
beta = 1.0

# example target distro over 3 qubits
data_probs = np.array([0.05, 0.10, 0.15, 0.20, 0.10, 0.05, 0.25, 0.10])
data_probs /= data_probs.sum()

# quantum Hamiltonian construction
def quantum_hamiltonian(params):
    # params: w0,w1,w2, g0,g1,g2, J01,J12,J02 (9 in total)
    w = params[:num_sys]          # Z terms
    g = params[num_sys:2*num_sys] # X terms
    J = params[2*num_sys:]        # ZZ couplings for (0,1), (1,2), (0,2)

    paulis = []
    coeffs = []

    # single qubit Z
    for i in range(num_sys):
        p = ['I']*num_sys
        p[i] = 'Z'
        paulis.append("".join(p))
        coeffs.append(w[i])

    # single qubit X
    for i in range(num_sys):
        p = ['I']*num_sys
        p[i] = 'X'
        paulis.append("".join(p))
        coeffs.append(g[i])

    # ZZ couplings
    pairs = [(0,1), (1,2), (0,2)]
    for idx, (i,j) in enumerate(pairs):
        p = ['I']*num_sys
        p[i] = 'Z'
        p[j] = 'Z'
        paulis.append("".join(p))
        coeffs.append(J[idx])

    h_sys = PauliSumOp.from_list(list(zip(paulis, coeffs)))
    id_anc = PauliSumOp.from_list([("I"*num_anc, 1.0)])
    h_full = h_sys.tensor(id_anc)
    return h_full

# purification ansatz
def purification_ansatz(params):
    qc = QuantumCircuit(num_total)
    num_layers = len(params) // (3 * num_total)
    params = np.reshape(params, (num_layers, num_total, 3))
    for layer in range(num_layers):
        for q in range(num_total):
            qc.rx(params[layer, q, 0], q)
            qc.ry(params[layer, q, 1], q)
            qc.rz(params[layer, q, 2], q)
        for i in range(num_total - 1):
            qc.cx(i, i + 1)
    return qc

# partial tracing to obtain reduced matrix
def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))  # Trace out ancillas
    return reduced_dm

# estimator for energy
estimator = Estimator()

def expected_energy(state_params, ham_params):
    qc = purification_ansatz(state_params)
    hamiltonian = quantum_hamiltonian(ham_params)
    energy = estimator.run(circuits=[qc], observables=[hamiltonian]).result().values[0]
    return energy

def free_energy(state_params, ham_params):
    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    energy = expected_energy(state_params, ham_params)
    ent = entropy(reduced_dm)
    return energy - ent / beta


loss_history = []
# loss function (cross_entropy + λ*free_energy)
def loss(params):
    ham_params = params[:9]
    state_params = params[9:]
    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    p_qbm = np.real(np.diag(reduced_dm.data))
    p_qbm = p_qbm.copy()
    p_qbm /= p_qbm.sum()
    p_qbm = np.clip(p_qbm, 1e-8, 1)
    cross_entropy = -np.sum(data_probs * np.log(p_qbm))
    f_energy = free_energy(state_params, ham_params)
    total_loss = cross_entropy + 0.1 * f_energy
    loss_history.append(total_loss)
    print(f"Loss: {total_loss:.4f}")
    return total_loss

# parameter initialization
num_layers = 2
purif_param_count = num_layers * num_total * 3 
init_ham_params = np.random.uniform(-1, 1, 9)
init_state_params = np.random.uniform(0, 2 * np.pi, purif_param_count)
init_params = np.concatenate([init_ham_params, init_state_params])

# optimization
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 50})

opt_ham_params = res.x[:9]
opt_state_params = res.x[9:]

print("Optimized Hamiltonian params:", opt_ham_params)

# QBM probs
qc = purification_ansatz(opt_state_params)
reduced_dm = get_reduced_density_matrix(qc)
qbm_probs = np.real(np.diag(reduced_dm.data))
qbm_probs = qbm_probs.copy()
qbm_probs /= qbm_probs.sum()

print("\nTarget distribution:", data_probs)
print("Learned QBM distribution:", qbm_probs)

# plotting      
labels = [format(i, '03b') for i in range(2**num_sys)]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, data_probs, width, label='Data')
plt.bar(x + width/2, qbm_probs, width, label='QBM')
plt.xticks(x, labels)
plt.ylabel('Probability')
plt.title('Target vs QBM learned distribution (3 qubits)')
plt.legend()
plt.show()
