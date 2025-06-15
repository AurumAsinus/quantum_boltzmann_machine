# parametric circuit to simulate 4-bit distributions

import numpy as np
from qiskit import Aer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms.optimizers import COBYLA
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from itertools import product
from qiskit.visualization import circuit_drawer

num_qubits = 4
num_layers = 3  # ansatz layers
total_params = num_qubits * num_layers

# example target distro
target_distribution = {
    '0100': 0.05,
    '0101': 0.1,
    '0110': 0.15,
    '0111': 0.2,
    '1000': 0.2,
    '1001': 0.15,
    '1010': 0.1,
    '1011': 0.05
}


# parametric ansatz
def build_qbm_circuit(weights):
    qc = QuantumCircuit(num_qubits)
    param_idx = 0
    for _ in range(num_layers):
        # rotations
        for qubit in range(num_qubits):
            qc.ry(weights[param_idx], qubit)
            param_idx += 1
        # entanglement (CNOT chain)
        for qubit in range(num_qubits - 1):
            qc.cx(qubit, qubit + 1)
    qc.measure_all()
    return qc

# KL divergence
def kl_divergence(p, q, epsilon=1e-8):
    return sum(p[k] * np.log((p[k] + epsilon) / (q.get(k, epsilon))) for k in p)

# backend simulator
backend = Aer.get_backend("aer_simulator")
qi = QuantumInstance(backend, shots=2048)
optimizer = COBYLA(maxiter=500)

# Kl divergence calculation
def objective(weights):
    qc = build_qbm_circuit(weights)
    compiled = transpile(qc, backend)
    job = backend.run(compiled, shots=2048)
    counts = job.result().get_counts()
    sampled_probs = {k: v / 2048 for k, v in counts.items()}
    return kl_divergence(target_distribution, sampled_probs)

# parameter initialization in (-2π, 2π)
init_weights = np.random.uniform(-np.pi, np.pi, size=total_params)

# optimization using COBYLA classical optimizer
result = minimize(objective, init_weights, method='COBYLA', options={'maxiter': 500})

print("Optimized weights:", result.x)
print("Final KL divergence:", result.fun)

# sample from optimized circuit
qc_opt = build_qbm_circuit(result.x)
compiled = transpile(qc_opt, backend)
job = backend.run(compiled, shots=2048)
counts = job.result().get_counts()
sampled_probs = {k: v / 2048 for k, v in counts.items()}

# print and plot learnt distro
print("Sampled distribution after training:")
for bitstring, prob in sampled_probs.items():
    print(f"{bitstring}: {prob:.3f}")

plt.bar(sampled_probs.keys(), sampled_probs.values())
plt.title("Sampled Distribution After Training")
plt.show()


# show probs for all states
all_states = [''.join(bits) for bits in product('01', repeat=4)]
full_target = {state: target_distribution.get(state, 0.0) for state in all_states}
full_learned = {state: sampled_probs.get(state, 0.0) for state in all_states}

# target vs learnt distro
print("\nState    Target    Learned")
print("----------------------------")
for state in all_states:
    print(f"{state:6}  {full_target[state]:7.3f}  {full_learned[state]:7.3f}")

states = sorted(all_states)
target_vals = [full_target[state] for state in states]
learned_vals = [full_learned[state] for state in states]
x = np.arange(len(states))

plt.figure(figsize=(10, 5))
bar_width = 0.35
plt.bar(x - bar_width/2, target_vals, width=bar_width, label='Target', color='skyblue')
plt.bar(x + bar_width/2, learned_vals, width=bar_width, label='Learned', color='salmon')
plt.xticks(x, states, rotation=90)
plt.ylabel('Probability')
plt.title('Target vs Learned Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# save ansatz with optimized params
# qc_optimized = build_qbm_circuit(result.x)
# circuit_drawer(qc_optimized, output='mpl', filename='optimized_ansatz_nn6.png')
