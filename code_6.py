import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy
from qiskit.opflow import PauliSumOp
from qiskit.primitives import Estimator
from scipy.optimize import minimize

# hyperparams
num_sys = 2
num_anc = 2
num_total = num_sys + num_anc
beta = 1.0
num_layers = 2

# quantum Hamtilonian construction
def quantum_hamiltonian(params):
    w0, w1, J01, G0, G1 = params
    op_z0 = PauliSumOp.from_list([("ZI", w0)])
    op_z1 = PauliSumOp.from_list([("IZ", w1)])
    op_zz = PauliSumOp.from_list([("ZZ", J01)])
    op_x0 = PauliSumOp.from_list([("XI", G0)])
    op_x1 = PauliSumOp.from_list([("IX", G1)])
    h_sys = op_z0 + op_z1 + op_zz + op_x0 + op_x1
    id_anc = PauliSumOp.from_list([("II", 1.0)])
    return h_sys.tensor(id_anc)

# purification circuit
def purification_ansatz(params):
    qc = QuantumCircuit(num_total)
    params = np.reshape(params, (num_layers, num_total, 3))
    for layer in range(num_layers):
        for q in range(num_total):
            qc.rx(params[layer, q, 0], q)
            qc.ry(params[layer, q, 1], q)
            qc.rz(params[layer, q, 2], q)
        for i in range(num_total - 1):
            qc.cx(i, i + 1)
    return qc

# reduced dm
def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    return partial_trace(dm, list(range(num_sys, num_total)))

# calculate expectation of energy
estimator = Estimator()

def expected_energy(state_params, ham_params):
    qc = purification_ansatz(state_params)
    hamiltonian = quantum_hamiltonian(ham_params)
    return estimator.run(circuits=[qc], observables=[hamiltonian]).result().values[0]

# Free energy loss function
def loss(params):
    ham_params = params[:5]
    state_params = params[5:]

    qc = purification_ansatz(state_params)
    reduced_dm = get_reduced_density_matrix(qc)
    entropy_val = entropy(reduced_dm)
    energy = expected_energy(state_params, ham_params)

    free_E = energy - entropy_val / beta
    print(f"Free Energy: {free_E:.6f}")
    return free_E

# param initialization
purif_param_count = num_layers * num_total * 3
init_ham_params = np.random.uniform(-1, 1, 5)
init_state_params = np.random.uniform(0, 2*np.pi, purif_param_count)
init_params = np.concatenate([init_ham_params, init_state_params])

# optimization
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 100})
opt_ham_params = res.x[:5]
opt_state_params = res.x[5:]

# final state probs
qc = purification_ansatz(opt_state_params)
rho = get_reduced_density_matrix(qc)
qbm_probs = np.real(np.diag(rho.data)).copy()
qbm_probs /= np.sum(qbm_probs)

# show distribution
labels = ["00", "01", "10", "11"]
x = np.arange(len(labels))

plt.bar(x, qbm_probs, width=0.4, label="Learned Thermal Distribution")
plt.xticks(x, labels)
plt.ylabel("Probability")
plt.title("Thermal State Learned from Free Energy Minimization")
plt.legend()
plt.grid(True)
plt.show()

# final energy, entropy, free energy
def manual_energy_entropy(rho, ham_params):
    w0, w1, J01, _, _ = ham_params
    Z0 = np.array([1, 1, -1, -1])
    Z1 = np.array([1, -1, 1, -1])
    ZZ = Z0 * Z1
    H_diag = w0 * Z0 + w1 * Z1 + J01 * ZZ
    probs = np.real(np.diag(rho.data))
    energy = np.sum(probs * H_diag)
    ent = entropy(rho)
    return energy, ent

E, S = manual_energy_entropy(rho, opt_ham_params)
F = E - S / beta

print("\nFinal Energy:", E)
print("Final Entropy:", S)
print("Final Free Energy:", F)
print("\nLearned Hamiltonian Parameters:")
print(f"w0 = {opt_ham_params[0]:.4f}, w1 = {opt_ham_params[1]:.4f}, J01 = {opt_ham_params[2]:.4f}")
print(f"G0 = {opt_ham_params[3]:.4f}, G1 = {opt_ham_params[4]:.4f}")

# optimized prufication saved
qc_opt = purification_ansatz(opt_state_params)
fig = qc_opt.draw(output='mpl')
fig.savefig("purification_ansatz_try_6_new.png")
print("Purification ansatz circuit saved as 'purification_ansatz_try_6_new.png'")
