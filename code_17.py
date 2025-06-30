# more advanced parametric circuit for 2bit
# distros modeling (visible + hidden)

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit

# target distro
target_distribution = {
    '00': 0.2,
    '01': 0.3,
    '10': 0.3,
    '11': 0.2
}


# target GHZ
qc = QuantumCircuit(3)
qc.h(0)
qc.cx(0, 1)
qc.cx(0, 2)
state = Statevector.from_instruction(qc)
# marginal from GHZ 3-qubit entanglement
marginal = state.probabilities_dict(qargs=[0, 1]) # only visible
target_distribution = marginal



# ansatz : 2 qubits (v) + 1 qubit (h)
def create_ansatz(params):
    qc = QuantumCircuit(3)

    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.ry(params[2], 2)
    # visible interacting with hidden
    qc.cx(0, 2)
    qc.cx(1, 2)
    qc.ry(params[3], 0)
    qc.ry(params[4], 1)
    qc.ry(params[5], 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.measure_all()
    return qc


sampler = Sampler()

# measure visible
def get_visible_counts(params, shots=1000):
    qc = create_ansatz(params)      # 6 params : 2 layers * 3 qubits
    result = sampler.run(qc, shots=shots).result()
    counts = result.quasi_dists[0].binary_probabilities()
    
    # extract only visible qubits
    visible_counts = {}
    for full_key, prob in counts.items():
        visible_key = full_key[-2:]
        visible_counts[visible_key] = visible_counts.get(visible_key, 0) + prob
    return visible_counts

# KL Divergence
def kl_divergence(p_model, p_target):
    epsilon = 1e-10
    keys = set(p_model) | set(p_target)
    kl = 0.0
    for key in keys:
        p = p_target.get(key, epsilon)
        q = p_model.get(key, epsilon)
        kl += p * np.log(p / q)
    return kl

def objective(params):
    probs = get_visible_counts(params)
    return kl_divergence(probs, target_distribution)

# optimization
np.random.seed(42)
# initialization of params in (0, 2π)
initial_params = np.random.uniform(0, 2 * np.pi, size=6)
# classical optimization using COBYLA
result = minimize(objective, initial_params, method='COBYLA')
opt_params = result.x
final_probs = get_visible_counts(opt_params)
kl = kl_divergence(final_probs, target_distribution)

# show all probs
all_keys = sorted(set(target_distribution.keys()).union(final_probs.keys()))
target_vals = [target_distribution.get(k, 0) for k in all_keys]
learned_vals = [final_probs.get(k, 0) for k in all_keys]
x = range(len(all_keys))


# target vs learnt prob distro
plt.figure(figsize=(8, 5))
bar_width = 0.35

plt.bar(x, target_vals, width=bar_width, label='Target', alpha=0.7)
plt.bar([i + bar_width for i in x], learned_vals, width=bar_width, label='Learned', alpha=0.7)

plt.xticks([i + bar_width / 2 for i in x], all_keys)
plt.ylabel('Probability')
plt.title('Target vs Learned (Visible) Distribution')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


final_probs = {i: float(j) for i, j in final_probs.items()}

print("\nOptimization complete.")
print("Final parameters:", opt_params)
print("Final KL divergence:", kl)
print("Target distribution:", target_distribution)
print("Learned (visible) distribution:", final_probs)
