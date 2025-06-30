import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import partial_trace, DensityMatrix, entropy
from scipy.optimize import minimize

num_sys = 2
num_anc = 2
num_total = num_sys + num_anc
beta = 1.0  # inverse temperature

data_states = ["00", "01", "10", "11"]
data_probs = np.array([0.3, 0.4, 0.2, 0.1])
data_probs /= np.sum(data_probs)

def purification_ansatz(params):
    qc = QuantumCircuit(num_total)
    for i in range(num_total):
        qc.rx(params[i], i)
    for i in range(num_sys):
        qc.cx(i, i + num_sys)
    return qc

def get_reduced_density_matrix(qc):
    backend = Aer.get_backend('statevector_simulator')
    sv = backend.run(transpile(qc, backend)).result().get_statevector()
    dm = DensityMatrix(sv)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
    return reduced_dm

def energy_and_entropy(rho, ham_params):
    Z0 = np.array([1, 1, -1, -1])
    Z1 = np.array([1, -1, 1, -1])
    ZZ = Z0 * Z1
    H_diag = ham_params[0]*Z0 + ham_params[1]*Z1 + ham_params[2]*ZZ
    probs = np.real(np.diag(rho.data))
    energy = np.sum(probs * H_diag)
    ent = entropy(rho)
    return energy, ent

# history for monitoring
energy_history = []
entropy_history = []
free_energy_history = []
cross_entropy_history = []
ham_params_history = []
purif_params_history = []

def combined_loss(all_params):
    # split parameters
    ham_params = all_params[:3]
    purif_params = all_params[3:]

    qc = purification_ansatz(purif_params)
    rho = get_reduced_density_matrix(qc)

    p_qbm = np.real(np.diag(rho.data)).copy()
    p_qbm /= np.sum(p_qbm)
    p_qbm = np.clip(p_qbm, 1e-8, 1)

    cross_entropy = -np.sum(data_probs * np.log(p_qbm))
    energy, ent = energy_and_entropy(rho, ham_params)
    free_energy = energy - ent / beta

    # history monitoring
    cross_entropy_history.append(cross_entropy)
    energy_history.append(energy)
    entropy_history.append(ent)
    free_energy_history.append(free_energy)
    ham_params_history.append(ham_params.copy())
    purif_params_history.append(purif_params.copy())

    print(f"Loss: {cross_entropy:.4f}, Energy: {energy:.4f}, Entropy: {ent:.4f}, Free energy: {free_energy:.4f}")
    print(f"Model probs: {p_qbm}")
    print(f"Data probs:  {data_probs}\n")

    return cross_entropy

# initialize params
initial_ham_params = np.random.uniform(-1, 1, 3)
initial_purif_params = np.random.uniform(0, 2*np.pi, num_total)
all_initial_params = np.concatenate([initial_ham_params, initial_purif_params])

# optimize both sets together
res = minimize(combined_loss, all_initial_params, method='COBYLA', options={'maxiter': 100})

print("Optimized Hamiltonian params:", res.x[:3])
print("Optimized Purification params:", res.x[3:])

# final model distro
qc_final = purification_ansatz(res.x[3:])
rho_final = get_reduced_density_matrix(qc_final)
p_qbm_final = np.real(np.diag(rho_final.data)).copy()
p_qbm_final /= np.sum(p_qbm_final)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(cross_entropy_history, label='Cross Entropy Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Cross Entropy Loss during training')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(energy_history, label='Energy')
plt.plot(entropy_history, label='Entropy')
plt.plot(free_energy_history, label='Free Energy')
plt.xlabel('Iteration')
plt.title('Energy / Entropy / Free Energy')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
ham_params_history_arr = np.array(ham_params_history)
for i in range(3):
    plt.plot(ham_params_history_arr[:, i], label=f'H param {i}')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Hamiltonian Parameters over Iterations')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.bar(range(len(data_probs)), data_probs, alpha=0.6, label='Data Distribution')
plt.bar(range(len(p_qbm_final)), p_qbm_final, alpha=0.6, label='Model Distribution')
plt.xticks(range(len(data_states)), data_states)
plt.title('Final Probability Distributions')
plt.xlabel('Basis State')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
