import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# target distro
target_distribution = {
    '00': 0.1,
    '01': 0.4,
    '10': 0.3,
    '11': 0.2
}

# KL divergence
def kl_divergence(p_model, p_target):
    epsilon = 1e-10
    keys = set(p_model) | set(p_target)
    kl = 0.0
    for key in keys:
        p = p_target.get(key, epsilon)
        q = p_model.get(key, epsilon)
        kl += p * np.log(p / q)
    return kl

# symmetric ansatz
def simple_ansatz(params):
    qc = QuantumCircuit(2)
    theta0, theta1 = params
    qc.ry(theta0, 0)
    qc.ry(theta1, 1)
    qc.cx(0, 1)
    return qc

# asymmetric ansatz
def asymmetric_ansatz(params):
    qc = QuantumCircuit(2)
    params = np.reshape(params, (2, 2, 3))  # (layers, qubits, (rx, ry, rz))
    for layer in range(2):
        for q in range(2):
            rx, ry, rz = params[layer][q]
            qc.rx(rx, q)
            qc.ry(ry, q)
            qc.rz(rz, q)
        qc.cz(0, 1)
    return qc

# ansatz optimizer
def optimize_ansatz(ansatz_func, num_params):
    def objective(params):
        qc = ansatz_func(params)
        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()
        trimmed = {k: probs.get(k, 1e-10) for k in target_distribution}
        return kl_divergence(trimmed, target_distribution)

    np.random.seed(42)
    init = np.random.uniform(-np.pi, np.pi, num_params)
    result = minimize(objective, init, method='COBYLA')
    return result.x, result.fun

# optimization for both ansatzes
params_simple, kl_simple = optimize_ansatz(simple_ansatz, 2)
params_asym, kl_asym = optimize_ansatz(asymmetric_ansatz, 12)

# compare distros
def get_distribution(ansatz_func, params):
    qc = ansatz_func(params)
    sv = Statevector.from_instruction(qc)
    return {k: round(v, 4) for k, v in sv.probabilities_dict().items()}

dist_simple = get_distribution(simple_ansatz, params_simple)
dist_asym = get_distribution(asymmetric_ansatz, params_asym)

print("\n=== Results Comparison ===")
print("Target Distribution:", target_distribution)
print("\nSimple Ansatz:")
print("KL Divergence:", round(kl_simple, 6))
print("Learned Distribution:", dist_simple)

print("\nAsymmetric Ansatz:")
print("KL Divergence:", round(kl_asym, 6))
print("Learned Distribution:", dist_asym)
