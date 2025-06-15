# Simple parametric circuit to learn target distribution

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer

# example target distro
target_distribution = {
    '00': 0.2,
    '01': 0.3,
    '10': 0.3,
    '11': 0.2
}

# target_distribution = {
#     '00': 0.5,
#     '11': 0.5,
# }

kl_progress = []

#  parametric ansatz
def create_ansatz(params):
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.cx(0, 1)
    qc.ry(params[2], 0)
    qc.ry(params[3], 1)
    qc.measure_all()
    return qc

# sample from ansatz to estimate prob distro
sampler = Sampler()

def get_counts(params, shots=100):
    qc = create_ansatz(params)
    result = sampler.run(qc, shots=shots).result()
    probs = result.quasi_dists[0].binary_probabilities()
    return probs

# kl divergence as metric
def kl_divergence(p_model, p_target):
    epsilon = 1e-10
    keys = set(p_model.keys()) | set(p_target.keys())
    kl = 0.0
    for key in keys:
        p = p_target.get(key, epsilon)
        q = p_model.get(key, epsilon)
        kl += p * np.log(p / q)
    return kl

# kl calculation
def objective(params):
    probs = get_counts(params)
    kl_value = kl_divergence(probs, target_distribution)
    kl_progress.append(kl_value)
    return kl_value

# initialize 4 params in (0, 2π)
np.random.seed(42)
initial_params = np.random.uniform(0, 2 * np.pi, size=4)

# optimization using classical optimizer COBYLA
result = minimize(objective, initial_params, method='COBYLA')
opt_params = result.x


final_probs = get_counts(opt_params)
kl = kl_divergence(final_probs, target_distribution)

print("\nOptimization complete.")
print("Final parameters:", opt_params)
print("Final KL divergence:", kl)
print("Target distribution:", target_distribution)
print("Learned distribution:", final_probs)


# show probs for all states (learnt vs target)
final_probs = {k: float(v) for k, v in final_probs.items()}
all_keys = sorted(set(target_distribution.keys()).union(final_probs.keys()))
target_vals = [target_distribution.get(k, 0) for k in all_keys]
learned_vals = [final_probs.get(k, 0) for k in all_keys]

x = range(len(all_keys))
bar_width = 0.35


# target vs learnt distro visualization
plt.figure(figsize=(8, 5))
plt.bar(x, target_vals, width=bar_width, label='Target', alpha=0.7)
plt.bar([i + bar_width for i in x], learned_vals, width=bar_width, label='Learned', alpha=0.7)
plt.xticks([i + bar_width / 2 for i in x], all_keys)
plt.ylabel("Probability")
plt.title("Target vs Learned Distribution")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# kl progress visualization
plt.figure(figsize=(8, 4))
plt.plot(kl_progress, marker='o', color='darkblue')
plt.xlabel("Optimization Step")
plt.ylabel("KL Divergence")
plt.title("KL Divergence During Optimization")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
#plt.savefig("kl_divergence_progress.png")
plt.show()




# save optimized circuit
optimized_circuit = create_ansatz(opt_params)
circuit_drawer(optimized_circuit, output='mpl', filename='optimized_pqc.png')

