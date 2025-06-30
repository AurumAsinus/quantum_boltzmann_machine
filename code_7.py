import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy, state_fidelity
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# hyperarameters
num_sys = 2
num_anc = 2
num_total = num_sys + num_anc
beta = 1.0

# target density matrix
qc_data = QuantumCircuit(2)
qc_data.h(0)
qc_data.cx(0, 1)
rho_target = DensityMatrix.from_instruction(qc_data)

# use quantum Hamiltonian
def quantum_hamiltonian(params):
    w0, w1, J01, G0, G1 = params
    op_z0 = PauliSumOp.from_list([("ZI", w0)])
    op_z1 = PauliSumOp.from_list([("IZ", w1)])
    op_zz = PauliSumOp.from_list([("ZZ", J01)])
    op_x0 = PauliSumOp.from_list([("XI", G0)])
    op_x1 = PauliSumOp.from_list([("IX", G1)])
    h_sys = op_z0 + op_z1 + op_zz + op_x0 + op_x1
    id_anc = PauliSumOp.from_list([("II", 1.0)])
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

# reduce to system-only density matrix
def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
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

# loss = fidelity loss + thermal regularization
def loss(params):
    ham_params = params[:5]
    state_params = params[5:]
    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    fid = state_fidelity(reduced_dm, rho_target)
    fidelity_loss = 1 - fid
    f_energy = free_energy(state_params, ham_params)
    total_loss = fidelity_loss + 0.1 * f_energy
    print(f"Fidelity loss: {fidelity_loss:.4f}, Free energy: {f_energy:.4f}, Total loss: {total_loss:.4f}")
    return total_loss

# optimization
num_layers = 2
purif_param_count = num_layers * num_total * 3
init_ham_params = np.random.uniform(-1, 1, 5)
init_state_params = np.random.uniform(0, 2 * np.pi, purif_param_count)
init_params = np.concatenate([init_ham_params, init_state_params])

res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 100})

optimized_ham_params = res.x[:5]
optimized_state_params = res.x[5:]

# learnt density matrix
qc = purification_ansatz(optimized_state_params)
rho_learned = get_reduced_density_matrix(qc)

# compare diagonal probs
labels = ["00", "01", "10", "11"]
x = np.arange(len(labels))
width = 0.35

probs_target = np.real(np.diag(rho_target.data))
probs_learned = np.real(np.diag(rho_learned.data))

plt.figure(figsize=(6, 4))
plt.bar(x - width/2, probs_target, width, label='Target')
plt.bar(x + width/2, probs_learned, width, label='Learned')
plt.xticks(x, labels)
plt.ylim(0, 1)
plt.ylabel('Probability')
plt.title('Diagonal Probabilities: Target vs Learned')
plt.legend()
plt.tight_layout()
plt.show()

# full density matrix heatmaps plotting
def plot_dm_heatmap(dm, title, ax):
    dm_real = np.real(dm.data)
    sns.heatmap(dm_real, annot=True, fmt=".2f", cmap='viridis', cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Basis")
    ax.set_ylabel("Basis")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_dm_heatmap(rho_target, "Target Density Matrix", axes[0])
plot_dm_heatmap(rho_learned, "Learned Density Matrix", axes[1])
plt.tight_layout()
plt.show()

# energy, entropy, free energy final
E = expected_energy(optimized_state_params, optimized_ham_params)
S = entropy(rho_learned)
F = E - S / beta

print("\nFinal QBM Energy:", E)
print("Final Entropy:", S)
print("Free Energy:", F)

w0, w1, J01, G0, G1 = optimized_ham_params
print(f"\nLearned Hamiltonian:\n w0*Z0 + w1*Z1 + J01*Z0Z1 + G0*X0 + G1*X1")
print(f"w0 = {w0:.4f}, w1 = {w1:.4f}, J01 = {J01:.4f}, G0 = {G0:.4f}, G1 = {G1:.4f}")
