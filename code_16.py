import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import DensityMatrix, partial_trace, state_fidelity
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

# params
num_sys = 3     # system qubits
num_anc = 2     # ancillas
num_total = num_sys + num_anc
num_layers = 2

# target GHZ state
qc_data = QuantumCircuit(num_sys)
qc_data.h(0)
qc_data.cx(0, 1)
qc_data.cx(0, 2)
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

# extract reduced state
def get_reduced_dm(qc):
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(transpile(qc, backend)).result()
    state = result.get_statevector()
    dm = DensityMatrix(state)
    reduced_dm = partial_trace(dm, list(range(num_sys, num_total)))
    return reduced_dm

# loss function : 1-fidelity
def loss(params):
    qc = purification_ansatz(params)
    rho_model = get_reduced_dm(qc)
    fid = state_fidelity(rho_target, rho_model)
    print(f"Fidelity: {fid:.6f}")
    return 1 - fid

 # initialize params and optimization
param_count = num_layers * num_total * 3
init_params = np.random.uniform(0, 2 * np.pi, param_count)
res = minimize(loss, init_params, method='COBYLA', options={'maxiter': 250, 'disp': True})
opt_params = res.x

# final learnt density matrix
final_qc = purification_ansatz(opt_params)
rho_learned = get_reduced_dm(final_qc)

# target vs learnt (diagonal probs)
def plot_dm_bars(dm, title, ax):
    probs = np.real(np.diag(dm.data))
    labels = [format(i, f'0{num_sys}b') for i in range(2**num_sys)] # extract from diagonal
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_ylabel("Probability")

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
plot_dm_bars(rho_target, "Target GHZ Distribution", axs[0])
plot_dm_bars(rho_learned, "Learned QBM Distribution", axs[1])
plt.tight_layout()
plt.show()

print("\nTarget Density Matrix:\n", rho_target.data)
print("\nLearned Density Matrix:\n", rho_learned.data)
print(f"\nFinal Fidelity: {state_fidelity(rho_target, rho_learned):.6f}")


# measure learnt model to aquire sampled distro
def sample_learned_model(opt_params, shots=1000):
    qc = purification_ansatz(opt_params)
    qc_sample = QuantumCircuit(num_total, num_sys)
    qc_sample.compose(qc, inplace=True)
    qc_sample.measure(range(num_sys), range(num_sys))

    backend = Aer.get_backend('qasm_simulator')
    compiled = transpile(qc_sample, backend)
    result = backend.run(compiled, shots=shots).result()
    counts = result.get_counts()

    # normalize
    total = sum(counts.values())
    probs = {k: v / total for k, v in counts.items()}

    # plot for all probs
    for k in [format(i, f'0{num_sys}b') for i in range(2**num_sys)]:
        if k not in probs:
            probs[k] = 0.0
    return probs

# sampled distro
sampled_probs = sample_learned_model(opt_params, shots=1000)
print("\nSampled distribution from learned model:", sampled_probs)

# sample distro vs initial target
target_probs = {"000": 0.5, "111": 0.5}
for k in [format(i, f'0{num_sys}b') for i in range(2**num_sys)]:
    if k not in target_probs:
        target_probs[k] = 0.0

labels = sorted(target_probs.keys())
x = np.arange(len(labels))
width = 0.35
sampled = [sampled_probs[k] for k in labels]
target = [target_probs[k] for k in labels]

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(x - width/2, target, width, label="Target")
ax.bar(x + width/2, sampled, width, label="Learnt QBM")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Probability")
ax.set_title("Learnt vs Target Distribution (3-qubit GHZ)")
ax.legend()
plt.tight_layout()
plt.show()
