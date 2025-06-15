# Simple parametric circuit to approach certain target state

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# target statevector
raw_psi4 = np.array([0.6, 0.3, 0.2 + 0.2j, 0.1 + 0.3j])
#raw_psi4 = np.array([0.5, 0.5j, 0.5, -0.5j])
norm_psi4 = raw_psi4 / np.linalg.norm(raw_psi4)

# pqc
def parametric_circuit(params, return_circuit=False):
    qc = QuantumCircuit(2)
    
    # u3 rotations
    qc.u(params[0], params[1], params[2], 0)
    qc.u(params[3], params[4], params[5], 1)
    
    # entanglement
    qc.cx(0, 1)
    
    # u3 rotations
    qc.u(params[6], params[7], params[8], 0)
    qc.u(params[9], params[10], params[11], 1)

    # return circuit to print out
    if return_circuit:
        return qc
    
    return Statevector.from_instruction(qc)

# Euclidean distance as cost function
def cost_function(params):
    # learnt statevector from pqc
    statevector_learnt = parametric_circuit(params)
    
    # compute Euclidean distance
    distance = np.linalg.norm(statevector_learnt - norm_psi4)    
    return distance

# initialization of 12 parameters into (0, 2π)
initial_params = np.random.rand(12) * 2 * np.pi  # 12 parameters for more flexibility

# optimization using Powell classical optimizer
result = minimize(cost_function, initial_params, method='Powell', options={'maxiter': 1000, 'disp': True})

# optimized params
optimized_params = result.x
print("Optimized parameters:", optimized_params)
final_statevector = parametric_circuit(optimized_params)
print("Final statevector ψ4:")
print(final_statevector)

target_vec = norm_psi4
learned_vec = final_statevector.data

# fidelity calculation
fidelity = np.abs(np.vdot(target_vec, learned_vec))**2

# Euclidean distance calculation
euclidean_dist = np.linalg.norm(target_vec - learned_vec)

print(f"Fidelity: {fidelity:.6f}")
print(f"Euclidean distance: {euclidean_dist:.6f}")

# plot absolute values of real and imaginary parts side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# real parts
axs[0].bar(range(len(target_vec)), np.abs(target_vec.real), width=0.4, label='Target', alpha=0.7, align='edge')
axs[0].bar(np.arange(len(learned_vec)) - 0.4, np.abs(learned_vec.real), width=0.4, label='Learned', alpha=0.7, align='edge')
axs[0].set_title('Absolute value of Real parts')
axs[0].set_xlabel('Basis State')
axs[0].set_ylabel('Magnitude')
axs[0].legend()
axs[0].grid(True)

# imaginary parts
axs[1].bar(range(len(target_vec)), np.abs(target_vec.imag), width=0.4, label='Target', alpha=0.7, align='edge')
axs[1].bar(np.arange(len(learned_vec)) - 0.4, np.abs(learned_vec.imag), width=0.4, label='Learned', alpha=0.7, align='edge')
axs[1].set_title('Absolute value of Imaginary parts')
axs[1].set_xlabel('Basis State')
axs[1].set_ylabel('Magnitude')
axs[1].legend()
axs[1].grid(True)

# set x-axis labels
basis_labels = ['|00⟩', '|01⟩', '|10⟩', '|11⟩']
for ax in axs:
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(basis_labels)

plt.tight_layout()
plt.show()
