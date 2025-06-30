import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# xor distro
target_distribution = {
    '000': 0.25,
    '011': 0.25,
    '101': 0.25,
    '110': 0.25
}

kl_progress = []
sampler = Sampler()

# parametric circuit
def create_ansatz(params):
    qc = QuantumCircuit(3)
    for i in range(3):
        qc.ry(params[i], i)
    qc.cx(0, 1)
    qc.cx(1, 2)
    for i in range(3):
        qc.ry(params[3 + i], i)
    qc.measure_all()
    return qc

# KL divergence
def kl_divergence(p_model, p_target):
    epsilon = 1e-10
    keys = set(p_model.keys()) | set(p_target.keys())
    kl = 0.0
    for key in keys:
        p = p_target.get(key, epsilon)
        q = p_model.get(key, epsilon)
        kl += p * np.log(p / q)
    return kl

# optimization
def objective(params):
    qc = create_ansatz(params)
    result = sampler.run(qc, shots=1000).result()
    probs = result.quasi_dists[0].binary_probabilities()

    # KL divergence between sampled and target
    kl = kl_divergence(probs, target_distribution)
    kl_progress.append(kl)
    return kl

np.random.seed(42)
initial_params = np.random.uniform(0, 2 * np.pi, size=6)
result = minimize(objective, initial_params, method='COBYLA')
opt_params = result.x

# sampled learnt distro
final_circuit = create_ansatz(opt_params)
final_probs = sampler.run(final_circuit, shots=2000).result().quasi_dists[0].binary_probabilities()
print("Final KL divergence:", kl_progress[-1])
print("Target distribution:")
for k, v in target_distribution.items():
    print(f"  {k}: {v}")
# generate learnt distribution
print("Generated distribution:")
for k, v in sorted(final_probs.items()):
    print(f"  {k}: {v:.3f}")

# KL divergence plot
plt.figure(figsize=(8, 4))
plt.plot(kl_progress, marker='o')
plt.xlabel("Step")
plt.ylabel("KL Divergence")
plt.title("KL Divergence During Training")
plt.grid(True)
plt.tight_layout()
plt.show()

all_keys = sorted(set(target_distribution.keys()) | set(final_probs.keys()))
target_vals = [target_distribution.get(k, 0) for k in all_keys]
learned_vals = [final_probs.get(k, 0) for k in all_keys]

x = np.arange(len(all_keys))
bar_width = 0.35

plt.figure(figsize=(9, 4))
plt.bar(x, target_vals, width=bar_width, label='Target', alpha=0.7)
plt.bar(x + bar_width, learned_vals, width=bar_width, label='Generated', alpha=0.7)
plt.xticks(x + bar_width / 2, all_keys)
plt.ylabel("Probability")
plt.title("Target vs Generated XOR Distribution")
plt.legend()
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
