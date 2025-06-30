import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from scipy.optimize import minimize

# xor dataset for classification
xor_data = {
    (0, 0): 0,
    (0, 1): 1,
    (1, 0): 1,
    (1, 1): 0
}

# Initialize sampler
sampler = Sampler()

# parametric circuit
def create_pqc(input_bits, params):
    # input encoding
    a, b = input_bits
    qc = QuantumCircuit(3)
    if a == 1:
        qc.x(0)
    if b == 1:
        qc.x(1)

    # Ry rotations on all qubits
    qc.ry(params[0], 0)
    qc.ry(params[1], 1)
    qc.ry(params[2], 2)

    # entanglement
    qc.cx(0, 2)
    qc.cx(1, 2)


    qc.ry(params[3], 2)
    qc.measure_all()
    return qc

# cross-entropy loss function
def compute_loss(params):
    loss = 0
    for input_bits, target in xor_data.items():
        qc = create_pqc(input_bits, params)
        result = sampler.run(qc, shots=100).result()
        probs = result.quasi_dists[0].binary_probabilities()
        prob_1 = sum(p for bitstr, p in probs.items() if bitstr[0] == '1')  # qubit 2 first in readout
        prob_0 = 1 - prob_1

        if target == 1:
            loss -= np.log(prob_1 + 1e-10)
        else:
            loss -= np.log(prob_0 + 1e-10)
    return loss / len(xor_data)

# training
np.random.seed(42)
init_params = np.random.uniform(0, 2*np.pi, size=4)

opt_result = minimize(compute_loss, init_params, method='COBYLA', options={'maxiter': 200})
opt_params = opt_result.x

# evaluate model
print("Final parameters:", opt_params)
for input_bits, target in xor_data.items():
    qc = create_pqc(input_bits, opt_params)
    probs = sampler.run(qc, shots=1000).result().quasi_dists[0].binary_probabilities()
    prob_1 = sum(p for bit, p in probs.items() if bit[0] == '1')
    predicted = 1 if prob_1 > 0.5 else 0
    print(f"Input: {input_bits}, Target: {target}, Predicted: {predicted}, P: {prob_1:.3f}")
