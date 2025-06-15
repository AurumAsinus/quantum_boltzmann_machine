# VQE to find ground state of selected Hamiltonian

from qiskit import Aer
from qiskit.algorithms import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.algorithms.optimizers import COBYLA
from qiskit.utils import QuantumInstance
from qiskit.opflow import Z, I
from qiskit.quantum_info import Pauli
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
import numpy as np
import matplotlib.pyplot as plt

# example Hamiltonian
hamiltonian = (
    PauliSumOp.from_list([("ZZ", 1.0)])  # Z_0 (on qubit 0)
    + PauliSumOp.from_list([("IX", 1.0)])  # X_1 (on qubit 1)
    + PauliSumOp.from_list([("ZZ", 0.5)])  # Z_0 Z_1
    + PauliSumOp.from_list([("XX", 0.5)])  # X_0 X_1
)

# TFIM
J = 1.0
h = 0.5
hamiltonian = (
    -J * PauliSumOp.from_list([("ZZ", 1.0)])    # -J * Z_0 Z_1
    - h * PauliSumOp.from_list([("XI", 1.0)])   # -h * X_0
    - h * PauliSumOp.from_list([("IX", 1.0)])   # -h * X_1
)



# parametric ansatz
ansatz = RealAmplitudes(num_qubits=2, reps=1)

# simulator
backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend=backend, shots=1024)

energy_history = []
def store_energy(eval_count, params, energy, stddev):
    energy_history.append(energy)

# classical optimization
optimizer = COBYLA(maxiter=500)

# VQE
vqe = VQE(ansatz=ansatz, optimizer=optimizer, quantum_instance=quantum_instance, callback=store_energy)
result = vqe.compute_minimum_eigenvalue(operator=hamiltonian)

print("Ground state energy:", result.eigenvalue.real)
print("Optimal parameters:", result.optimal_parameters)

# compute exact ground state
matrix = hamiltonian.to_matrix()
eigenvalues = np.linalg.eigvalsh(matrix)
print("Exact ground state energy:", eigenvalues[0])

# plot convergence
plt.plot(range(1, len(energy_history) + 1), energy_history, marker='o')
plt.xlabel("Iteration")
plt.ylabel("Estimated Energy")
plt.title("VQE Optimization Convergence")
plt.grid(True)
plt.tight_layout()
plt.show()


