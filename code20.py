import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.primitives import Estimator
from qiskit.opflow import PauliSumOp
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms import NumPyEigensolver
import matplotlib.pyplot as plt
import seaborn as sns

num_qubits = 2

# example Hamiltonian
pauli_ops = [
    PauliSumOp.from_list([("ZI", 1.0)]),  # Z on qubit 0
    PauliSumOp.from_list([("IZ", 1.0)]),  # Z on qubit 1
    PauliSumOp.from_list([("ZZ", 1.0)])   # ZZ interaction
]

# construct hamiltonian
def param_hamiltonian(params):
    op0 = PauliSumOp(SparsePauliOp(pauli_ops[0].primitive.paulis, params[0] * pauli_ops[0].primitive.coeffs))
    op1 = PauliSumOp(SparsePauliOp(pauli_ops[1].primitive.paulis, params[1] * pauli_ops[1].primitive.coeffs))
    op2 = PauliSumOp(SparsePauliOp(pauli_ops[2].primitive.paulis, params[2] * pauli_ops[2].primitive.coeffs))
    return op0 + op1 + op2

# parametric ansatz
def qbm_ansatz(weights):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(weights[i], i)
    qc.cx(0, 1)
    return qc

# Estimator backend
estimator = Estimator()

# expected energy with hamiltonian params h_params
def expected_energy(state_params, h_params):
    qc = qbm_ansatz(state_params)
    h_op = param_hamiltonian(h_params)
    energy = estimator.run(circuits=[qc], observables=[h_op]).result().values[0]
    return energy

# optimize ansatz for fixed hamiltonian params
def optimize_state(h_params):
    init = np.random.uniform(0, 2 * np.pi, size=num_qubits)
    res = minimize(expected_energy, init, args=(h_params,), method='COBYLA')
    return res.x

# energy minimization
energy_log = []
def loss(h_params):
    opt_state = optimize_state(h_params)
    # minimal energy of optimal state
    energy = expected_energy(opt_state, h_params)
    energy_log.append(energy)
    print(f"H params: {h_params}, Min energy: {energy:.4f}")
    return energy

# param initialization
init_h_params = np.random.uniform(-1, 1, size=3)

# classical optimization
result = minimize(loss, init_h_params, method='COBYLA', options={'maxiter': 30})

print("\nOptimization finished.")
print("Optimized Hamiltonian params:", result.x)

final_state_params = optimize_state(result.x)
print("Final optimized state params:", final_state_params)
final_energy = expected_energy(final_state_params, result.x)
print(f"Final expected energy: {final_energy:.4f}")


# Hamiltonian with optimized parameters
final_h_op = param_hamiltonian(result.x)

# Exact ground state calculation
eigensolver = NumPyEigensolver(k=1)
exact_result = eigensolver.compute_eigenvalues(operator=final_h_op)
exact_ground_energy = exact_result.eigenvalues[0].real

print(f"Exact ground state energy from NumPyEigensolver: {exact_ground_energy:.4f}")
print(f"Difference (Variational - Exact): {final_energy - exact_ground_energy:.4f}")


# plotting

# ground state searching
plt.figure(figsize=(6, 4))
plt.plot(energy_log, marker='o')
plt.xlabel("Outer Optimization Iteration")
plt.ylabel("Minimal Energy Found")
plt.title("Energy Convergence During Hamiltonian Learning")
plt.grid(True)
plt.tight_layout()
plt.show()

# exact vs found ground energy
plt.figure(figsize=(4, 4))
plt.bar(["VQE Energy", "Exact Energy"], [final_energy, exact_ground_energy], color=["blue", "green"])
plt.ylabel("Energy")
plt.title("Final Learned vs. Exact Ground State Energy")
plt.tight_layout()
plt.show()

qc_final = qbm_ansatz(final_state_params)
qc_final.draw('mpl')

