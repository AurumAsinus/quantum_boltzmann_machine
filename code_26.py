import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# hyperparams
num_sys = 2        # Number of system qubits
num_anc = 2        # Number of ancilla qubits for purification
num_total = num_sys + num_anc
beta = 1.0         # inverse temperature for free energy term

# target distribution
data_probs = np.array([0.1, 0.2, 0.65, 0.05])

# estimator for energy
estimator = Estimator()

# quantum Hamiltonian construction
def quantum_hamiltonian(params):
    # For num_sys=2, params = [w0, w1, J01, G0, G1]
    w0, w1, J01, G0, G1 = params
    op_z0 = PauliSumOp.from_list([("ZI", w0)])
    op_z1 = PauliSumOp.from_list([("IZ", w1)])
    op_zz = PauliSumOp.from_list([("ZZ", J01)])
    op_x0 = PauliSumOp.from_list([("XI", G0)])
    op_x1 = PauliSumOp.from_list([("IX", G1)])

    h_sys = op_z0 + op_z1 + op_zz + op_x0 + op_x1

    id_anc = PauliSumOp.from_list([("II", 1.0)])  # identity on ancillas

    return h_sys.tensor(id_anc)

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

# partial tracing to obtain reduced density matrix
def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    return partial_trace(dm, list(range(num_sys, num_total)))

loss_history = []
free_energy_history = []
kl_div_history = []

# loss function
def loss(full_params):
    # full_params = [ham_params(5), state_params(variable length)]
    ham_params = full_params[:5]
    state_params = full_params[5:]

    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)

    # QBM probabilities (diagonal of reduced density matrix)
    p_qbm = np.real(np.diag(reduced_dm.data))
    p_qbm = p_qbm.copy()
    p_qbm /= np.sum(p_qbm)

    # cross-entropy loss (target vs learnt)
    p_qbm = np.clip(p_qbm, 1e-8, 1)
    cross_entropy = -np.sum(data_probs * np.log(p_qbm))

    # free energy regularization term
    hamiltonian = quantum_hamiltonian(ham_params)
    energy = estimator.run(circuits=[qc], observables=[hamiltonian]).result().values[0]
    ent = entropy(reduced_dm)
    free_energy = energy - ent / beta

    # KL divergence
    kl = np.sum(data_probs * np.log(data_probs / p_qbm))

    # history monitoring
    loss_history.append(cross_entropy)
    free_energy_history.append(free_energy)
    kl_div_history.append(kl)

    print(f"Loss: {cross_entropy:.6f}, Free energy: {free_energy:.6f}, KL: {kl:.6f}")

    # total loss
    return cross_entropy + 0.1 * free_energy        # λ = 0.1

# initialize params
num_layers = 2
purif_param_count = num_layers * num_total * 3
init_ham_params = np.random.uniform(-1, 1, 5)
init_state_params = np.random.uniform(0, 2 * np.pi, purif_param_count)
init_params = np.concatenate([init_ham_params, init_state_params])

# optimization
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 100})

optimized_ham_params = res.x[:5]
optimized_state_params = res.x[5:]

print("\nOptimized Hamiltonian parameters:", optimized_ham_params)
print("Optimized purification parameters shape:", optimized_state_params.shape)

# final QBM probs
qc_final = purification_ansatz(optimized_state_params)
rho_final = get_reduced_density_matrix(qc_final)
qbm_probs = np.real(np.diag(rho_final.data))
qbm_probs = qbm_probs.copy()
qbm_probs /= np.sum(qbm_probs)

print("\nTarget data distribution:", data_probs)
print("QBM learned distribution:", qbm_probs)
print("Final KL divergence:", np.sum(data_probs * np.log(data_probs / np.clip(qbm_probs, 1e-8, 1))))

# plotting
labels = [format(i, f'0{num_sys}b') for i in range(2 ** num_sys)]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, data_probs, width, label='Target Data')
plt.bar(x + width/2, qbm_probs, width, label='QBM Learned')
plt.xticks(x, labels)
plt.ylabel('Probability')
plt.title('Target vs QBM Learned Distribution')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_history, label='Cross-Entropy Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(free_energy_history, label='Free Energy', color='orange')
plt.xlabel('Iteration')
plt.ylabel('Free Energy')
plt.title('Free Energy During Training')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(kl_div_history, label='KL Divergence', color='green')
plt.xlabel('Iteration')
plt.ylabel('KL Divergence')
plt.title('KL Divergence (Data || QBM)')
plt.grid(True)

plt.tight_layout()
plt.show()
