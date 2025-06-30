from qiskit import Aer, execute, QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize
import numpy as np

# Set backend and shots
backend = Aer.get_backend('aer_simulator')
shots = 1024

# Define target distribution (two-modal)
target_distribution = {'00': 0.5, '11': 0.5, '01': 0.0, '10': 0.0}

# Define ansatz (parameterized quantum circuit)
num_qubits = 2
params = ParameterVector('θ', 4)

def construct_ansatz(theta):
    qc = QuantumCircuit(num_qubits)
    qc.ry(theta[0], 0)
    qc.ry(theta[1], 1)
    qc.cx(0, 1)
    qc.ry(theta[2], 0)
    qc.ry(theta[3], 1)
    qc.measure_all()
    return qc

# Loss function: L2 distance between sampled and target distributions
def compute_loss(output_counts, target_probs, shots):
    measured_probs = {k: v / shots for k, v in output_counts.items()}
    for key in target_probs:
        if key not in measured_probs:
            measured_probs[key] = 0.0
    return sum((measured_probs[k] - target_probs[k])**2 for k in target_probs)

# Evaluation function used by the optimizer
def evaluate(theta_vals):
    qc = construct_ansatz(theta_vals)
    job = execute(qc, backend, shots=shots)
    counts = job.result().get_counts()
    return compute_loss(counts, target_distribution, shots)

# Initialize parameters and optimize
initial_theta = np.random.uniform(0, 2 * np.pi, size=4)
result = minimize(evaluate, initial_theta, method='COBYLA', options={'maxiter': 100})

# Get optimized parameters
optimal_theta = result.x
final_qc = construct_ansatz(optimal_theta)

# Evaluate the final circuit
job = execute(final_qc, backend, shots=shots)
final_counts = job.result().get_counts()
final_probs = {k: round(v / shots, 2) for k, v in final_counts.items()}

# Output
print("Learned distribution:", final_probs)
print("Target distribution :", target_distribution)
print("Optimized parameters:", np.round(optimal_theta, 3))

import matplotlib.pyplot as plt

# Ensure all keys exist in both dictionaries
all_keys = sorted(set(target_distribution.keys()) | set(final_probs.keys()))
target_probs = [target_distribution.get(k, 0.0) for k in all_keys]
learned_probs = [final_probs.get(k, 0.0) for k in all_keys]

# Plot
x = np.arange(len(all_keys))
width = 0.35

fig, ax = plt.subplots()
ax.bar(x - width/2, target_probs, width, label='Target')
ax.bar(x + width/2, learned_probs, width, label='Learned')

ax.set_xticks(x)
ax.set_xticklabels(all_keys)
ax.set_ylabel('Probability')
ax.set_title('Target vs Learned Distribution')
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
