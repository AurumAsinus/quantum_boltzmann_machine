import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy, Pauli
from qiskit.opflow import PauliSumOp, StateFn, ExpectationFactory, CircuitStateFn
from qiskit.primitives import Estimator
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit.visualization import circuit_drawer
import matplotlib.pyplot as plt

num_sys = 2
num_anc = 2
num_total = num_sys + num_anc
beta = 1.0

# data distro
data_probs = np.array([0.1, 0.2, 0.65, 0.05])

from qiskit.opflow import I

# Hamiltonian
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

def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
    return reduced_dm

estimator = Estimator()

# <H>
def expected_energy(state_params, ham_params):
    qc = purification_ansatz(state_params)
    hamiltonian = quantum_hamiltonian(ham_params)
    energy = estimator.run(circuits=[qc], observables=[hamiltonian]).result().values[0]
    return energy
# F = <H> - S/β
def free_energy(state_params, ham_params):
    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    energy = expected_energy(state_params, ham_params)
    ent = entropy(reduced_dm)
    return energy - ent / beta

loss_history = []
free_energy_history = []
kl_div_history = []

def loss(params):
    # params = [ham_params, state_params]
    ham_params = params[:5]
    state_params = params[5:]
    
    # get reduced density matrix
    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    # QBM distro
    p_qbm = np.real(np.diag(reduced_dm.data)).copy()
    p_qbm /= np.sum(p_qbm)

    # cross-entropy loss vs data distribution
    p_qbm = np.clip(p_qbm, 1e-8, 1)
    cross_entropy = -np.sum(data_probs * np.log(p_qbm))

    # regularize with free energy to encourage thermal approx
    f_energy = free_energy(state_params, ham_params)

    # KL divergence
    kl = np.sum(data_probs * np.log(data_probs / np.clip(p_qbm, 1e-8, 1)))

    # history monitoring
    loss_history.append(cross_entropy)
    free_energy_history.append(f_energy)
    kl_div_history.append(kl)

    print(f"Loss: {cross_entropy:.4f}, Free energy: {f_energy:.4f}")

    return cross_entropy + 0.1 * f_energy  # weighted sum

# param initialization
num_layers = 2
purif_param_count = num_layers * num_total * 3
init_ham_params = np.random.uniform(-1, 1, 5)  # w0, w1, J01, G0, G1
init_state_params = np.random.uniform(0, 2 * np.pi, purif_param_count)
init_params = np.concatenate([init_ham_params, init_state_params])

res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 100})

optimized_ham_params = res.x[:5]
optimized_state_params = res.x[5:]

print("Optimized Hamiltonian params:", optimized_ham_params)
print("Optimized purification params:", optimized_state_params)

# energy and entropy calculation
def energy_and_entropy(rho, ham_params):
    w0, w1, J01, _, _ = ham_params          # Only Z and ZZ for reduced system
    Z0 = np.array([1, 1, -1, -1])           # Z on qubit 0
    Z1 = np.array([1, -1, 1, -1])           # Z on qubit 1
    ZZ = Z0 * Z1
    H_diag = w0 * Z0 + w1 * Z1 + J01 * ZZ
    probs = np.real(np.diag(rho.data))
    energy = np.sum(probs * H_diag)
    ent = entropy(rho)
    return energy, ent

# build final state
qc = purification_ansatz(optimized_state_params)
rho = get_reduced_density_matrix(qc)

# QBM distro
qbm_probs = np.real(np.diag(rho.data)).copy()
qbm_probs /= np.sum(qbm_probs)

print("\nTarget data distribution:", data_probs)
print("QBM learned distribution:", qbm_probs)
print("KL Divergence (data || model):", np.sum(data_probs * np.log(data_probs / np.clip(qbm_probs, 1e-8, 1))))


# plotting

labels = ["00", "01", "10", "11"]
x = np.arange(len(labels))
width = 0.35

plt.bar(x - width/2, data_probs, width, label='Data')
plt.bar(x + width/2, qbm_probs, width, label='QBM')
plt.xticks(x, labels)
plt.ylabel('Probability')
plt.title('Target vs QBM learned distribution')
plt.legend()
plt.show()


# compute expected QBM energy, entropy, free energy
sv = Statevector.from_instruction(qc)
dm = DensityMatrix(sv)
reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))

E, S = energy_and_entropy(reduced_dm, optimized_ham_params)
F = E - S / beta

print("\nFinal QBM Energy:", E)
print("Final Entropy:", S)
print("Free Energy:", F)


w0, w1, J01, G0, G1 = optimized_ham_params
print(f"\nLearned Hamiltonian:\n w0*Z0 + w1*Z1 + J01*Z0Z1 + G0*X0 + G1*X1")
print(f"w0 = {w0:.4f}, w1 = {w1:.4f}, J01 = {J01:.4f}, G0 = {G0:.4f}, G1 = {G1:.4f}")

# create optimized energy
final_ansatz = purification_ansatz(optimized_state_params)

# save circuit
circuit_drawer(final_ansatz, output='mpl', filename='purification_ansatz.png')

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
plt.savefig("training_progress.png", dpi=300)
plt.show()
