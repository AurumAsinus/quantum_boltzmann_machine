import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from qiskit.visualization import plot_histogram
from collections import Counter

num_sys = 2
num_anc = 2
num_total = num_sys + num_anc
num_layers = 2

# target density matrix
qc_data = QuantumCircuit(num_sys)
qc_data.h(0)
qc_data.cx(0, 1)
rho_target = DensityMatrix.from_instruction(qc_data)

# purification
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

# partial tracing
def get_reduced_dm(qc):
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    state = result.get_statevector()
    dm = DensityMatrix(state)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
    return reduced_dm

# loss function (fidelity)
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
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 200, 'disp': True})
opt_params = res.x

# learnt density matrix
final_qc = purification_ansatz(opt_params)
rho_learned = get_reduced_dm(final_qc)

# compare target vs learnt
def plot_dm_bars(dm, title, ax):
    probs = np.real(np.diag(dm.data))
    labels = ["00", "01", "10", "11"]
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
plot_dm_bars(rho_target, "Target (Bell) Distribution", axs[0])
plot_dm_bars(rho_learned, "Learned QBM Distribution", axs[1])
plt.tight_layout()
plt.show()

print("\nTarget Density Matrix:\n", rho_target.data)
print("\nLearned Density Matrix:\n", rho_learned.data)
print(f"\nFinal Fidelity: {state_fidelity(rho_target, rho_learned):.6f}")

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
plt.title('Target vs Learned Distribution')
plt.legend()
plt.tight_layout()
plt.show()

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
plt.title('Target vs Learned Diagonal Probabilities')
plt.legend()
plt.tight_layout()
plt.show()

# density matrix heatmaps
def plot_density_matrix_heatmap(dm, title, ax):
    dm_real = np.real(dm.data)
    sns.heatmap(dm_real, annot=True, fmt=".2f", cmap='viridis', cbar=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Basis state")
    ax.set_ylabel("Basis state")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_density_matrix_heatmap(rho_target, "Target Density Matrix", axes[0])
plot_density_matrix_heatmap(rho_learned, "Learned Density Matrix", axes[1])
plt.tight_layout()
plt.show()


# sampling from learnt model
def sample_learned_model(opt_params, shots=1000):
    qc = purification_ansatz(opt_params)
    qc_meas = qc.copy()
    qc_meas.measure_all()

    # sample only from system qubits
    qc_sample = QuantumCircuit(num_total, num_sys)
    qc_sample.compose(qc, inplace=True)
    qc_sample.measure(range(num_sys), range(num_sys))

    backend = Aer.get_backend('qasm_simulator')
    compiled = transpile(qc_sample, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    return probs

sampled_probs = sample_learned_model(opt_params, shots=1000)
print("\nSampled distribution from learned model:", sampled_probs)

target_probs = {"00": 0.5, "11": 0.5, "01": 0.0, "10": 0.0}
for k in ["00", "01", "10", "11"]:
    if k not in sampled_probs:
        sampled_probs[k] = 0.0

fig, ax = plt.subplots(figsize=(6, 4))
labels = ["00", "01", "10", "11"]
x = np.arange(len(labels))
width = 0.35
sampled = [sampled_probs[k] for k in labels]
target = [target_probs[k] for k in labels]

ax.bar(x - width/2, target, width, label="Target")
ax.bar(x + width/2, sampled, width, label="Sampled QBM")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Probability")
ax.set_title("Sampled vs Target Distribution")
ax.legend()
plt.tight_layout()
plt.show()
