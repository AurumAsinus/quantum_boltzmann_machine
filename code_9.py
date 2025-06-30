import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.primitives import Estimator
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp, Statevector
from scipy.optimize import minimize
from qiskit.quantum_info import Pauli
from qiskit.quantum_info import Statevector
import seaborn as sns
from qiskit.visualization import circuit_drawer
from qiskit.visualization import circuit_drawer

num_qubits = 2
num_layers = 2  
param_shape = (num_layers, num_qubits, 3)
num_params = np.prod(param_shape)

# pauli_ops = [
#     PauliSumOp.from_list([("ZI", 1.0)]),
#     PauliSumOp.from_list([("IZ", 1.0)]),
#     PauliSumOp.from_list([("ZZ", 1.0)])
# ]
# example Hamiltonian ops
pauli_ops = [
    PauliSumOp.from_list([("ZI", 1.0)]),
    PauliSumOp.from_list([("IZ", 1.0)]),
    PauliSumOp.from_list([("ZZ", 1.0)]),
    PauliSumOp.from_list([("XX", 1.0)]),
    PauliSumOp.from_list([("YY", 1.0)])
]

# 2-qubit Pauli to matrix
def pauli_to_matrix(pauli_str):
    pauli_matrices = {
        'I': np.array([[1, 0], [0, 1]]),
        'X': np.array([[0, 1], [1, 0]]),
        'Y': np.array([[0, -1j], [1j, 0]]),
        'Z': np.array([[1, 0], [0, -1]])
    }
    return np.kron(pauli_matrices[pauli_str[0]], pauli_matrices[pauli_str[1]])


# construct hamiltonian
def param_hamiltonian(params):
    ops = []
    for i in range(len(params)):
        scaled_op = PauliSumOp(
            SparsePauliOp(pauli_ops[i].primitive.paulis, params[i] * pauli_ops[i].primitive.coeffs)
        )
        ops.append(scaled_op)
    hamiltonian = ops[0]
    for op in ops[1:]:
        hamiltonian += op
    return hamiltonian

# parametric ansatz
def expressive_ansatz(flat_weights):
    weights = np.reshape(flat_weights, param_shape)
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        for q in range(num_qubits):
            rx, ry, rz = weights[layer, q]
            qc.rx(rx, q)
            qc.ry(ry, q)
            qc.rz(rz, q)
        for i in range(num_qubits - 1):
            qc.cx(i, i + 1)
    return qc

# estimator to compute expected energy
estimator = Estimator()

def expected_energy(state_params, h_params):
    qc = expressive_ansatz(state_params)
    h_op = param_hamiltonian(h_params)
    energy = estimator.run(circuits=[qc], observables=[h_op]).result().values[0]
    return energy

# minimize energy with respect to Hamiltonian params
def optimize_state(h_params):
    init = np.random.uniform(0, 2 * np.pi, size=num_params)
    res = minimize(expected_energy, init, args=(h_params,), method='COBYLA')
    return res.x

# history monitoring
energy_history = []
hparam_history = []

def loss(h_params):
    opt_state = optimize_state(h_params)
    energy = expected_energy(opt_state, h_params)
    print(f"H params: {h_params}, Min energy: {energy:.4f}")
    energy_history.append(energy)
    hparam_history.append(h_params.copy())
    return energy

init_h_params = np.random.uniform(-1, 1, size=5)
# for each h_params optimize state params
result = minimize(loss, init_h_params, method='COBYLA', options={'maxiter': 30})

final_hparams = result.x
final_state_params = optimize_state(final_hparams)
final_energy = expected_energy(final_state_params, final_hparams)

print("\n--- Optimization Finished ---")
print("Best Hamiltonian parameters:", final_hparams)
print("Best state parameters (flat):", final_state_params)
print(f"Final expected energy: {final_energy:.4f}")


# plotting
sns.set(style="whitegrid")

# energy minimization
plt.figure(figsize=(7, 4))
plt.plot(energy_history, marker='o')
plt.title("Expected Energy vs. Optimization Step")
plt.xlabel("Iteration")
plt.ylabel("Minimum Energy")
plt.tight_layout()
plt.show()

# hamiltonian parameters evolution
hparam_history = np.array(hparam_history)
plt.figure(figsize=(7, 4))
for i in range(5):
    plt.plot(hparam_history[:, i], label=f"h[{i}]")
plt.title("Hamiltonian Parameters During Optimization")
plt.xlabel("Iteration")
plt.ylabel("Parameter Value")
plt.legend()
plt.tight_layout()
plt.show()

# final ansatz
print("\nQuantum Circuit for Optimized State:")
print(expressive_ansatz(final_state_params).draw("text"))


# calculate energies for each state
basis_states = ["00", "01", "10", "11"]
basis_sv = [Statevector.from_label(b) for b in basis_states]
hamiltonian = param_hamiltonian(final_hparams)
energies = [sv.expectation_value(hamiltonian).real for sv in basis_sv]
# show ground state
ground_energy = min(energies)
ground_state = basis_states[np.argmin(energies)]
print(f"\nKnown ground state: |{ground_state}⟩, Energy: {ground_energy:.4f}")

# Compare learned state
learned_sv = Statevector.from_instruction(expressive_ansatz(final_state_params))
fidelity = np.abs(learned_sv.inner(Statevector.from_label(ground_state)))**2
print(f"Fidelity with ground state |{ground_state}⟩: {fidelity:.4f}")

# learnt vs exact ground state
plt.figure(figsize=(8, 5))
plt.bar(basis_states, energies, color='lightblue', label='Basis state energy')
plt.axhline(final_energy, color='red', linestyle='--', label='Learned state energy')
plt.axhline(ground_energy, color='green', linestyle=':', label='Ground state energy')
plt.title("Energy Comparison: Learned vs Ground State")
plt.xlabel("Computational Basis States")
plt.ylabel("Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# learnt probs of states in ground state energy
probs = learned_sv.probabilities_dict()
plt.bar(probs.keys(), probs.values(), color='skyblue')
plt.title("Learned State: Basis Probabilities")
plt.ylabel("Probability")
plt.xlabel("Basis State")
plt.grid(True)
plt.show()



# optimized Hamiltonian
pauli_labels = ["ZI", "IZ", "ZZ", "XX", "YY"]
H_matrix = np.zeros((4,4), dtype=complex)
for coeff, label in zip(final_hparams, pauli_labels):
    H_matrix += coeff * pauli_to_matrix(label)

# exact ground state calculation
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
ground_energy = np.min(eigenvalues)
ground_state_vec = eigenvectors[:, np.argmin(eigenvalues)]
print(f"\nExact ground state energy: {ground_energy:.4f}")

# ground state vector
print("Ground state vector (amplitudes in computational basis |00>,|01>,|10>,|11>):")
for i, amp in enumerate(ground_state_vec):
    print(f"|{format(i, '02b')}>: {amp:.4f}")

# fidelity between learnt and exact ground state
from qiskit.quantum_info import Statevector
learned_sv = Statevector.from_instruction(expressive_ansatz(final_state_params))
fidelity = np.abs(np.dot(np.conj(ground_state_vec), learned_sv.data))**2
print(f"Fidelity of learned state with exact ground state: {fidelity:.4f}")

# exact ground state of example Hamiltonian
pauli_labels = ["ZI", "IZ", "ZZ", "XX", "YY"]       # adjust according to Hamiltonian
H_matrix = np.zeros((4,4), dtype=complex)
for coeff, label in zip(final_hparams, pauli_labels):
    H_matrix += coeff * pauli_to_matrix(label)
eigenvalues, eigenvectors = np.linalg.eigh(H_matrix)
ground_energy = np.min(eigenvalues)
ground_state_vec = eigenvectors[:, np.argmin(eigenvalues)]

print(f"\nExact ground state energy: {ground_energy:.6f}")

print("Ground state vector amplitudes in computational basis (|00>, |01>, |10>, |11>):")
for i, amp in enumerate(ground_state_vec):
    print(f"  |{format(i, '02b')}>: {amp:.6f}")

# fidelity with learned state
learned_sv = Statevector.from_instruction(expressive_ansatz(final_state_params))
fidelity = np.abs(np.dot(np.conj(ground_state_vec), learned_sv.data))**2
print(f"Fidelity of learned state with exact ground state: {fidelity:.6f}")


# final ansatz circuit
final_circuit = expressive_ansatz(final_state_params)
circuit_drawer(final_circuit, output='mpl', filename='final_parametric_circuit.png')
