# classical simulation of QBM

import numpy as np
from functools import reduce
from scipy.linalg import expm
from qiskit.opflow import I as QI, X as QX, Y as QY, Z as QZ, PauliOp
import matplotlib.pyplot as plt
import seaborn as sns

# parameters
n_visible = 4
n_hidden = 2
n_qubits = n_visible + n_hidden
eta = 0.5
Gamma = 1.0


# Pauli op across all qubits
def multi_pauli_op(pauli_map, n):
    base_ops = {'I': QI, 'X': QX, 'Y': QY, 'Z': QZ}
    result = None
    for i in range(n):
        op_char = pauli_map.get(i, 'I')
        op = base_ops[op_char]
        result = op if result is None else result ^ op
    return result

# Pauli op
def pauli_op(op_char, idx, n):
    return multi_pauli_op({idx: op_char}, n)

# construct Hamiltonian (TFIM)
def build_hamiltonian(b, w, gamma=Gamma):
    H_terms = []
    for i in range(n_qubits):
        if gamma != 0:  # include transverse field terms
            H_terms.append((-gamma) * pauli_op('X', i, n_qubits))  # Only add if non-zero
        H_terms.append((-b[i]) * pauli_op('Z', i, n_qubits))
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if not (i >= n_visible and j >= n_visible):
                zz_op = pauli_op('Z', i, n_qubits) @ pauli_op('Z', j, n_qubits)
                H_terms.append((-w[i, j]) * zz_op)
    return reduce(lambda x, y: x + y, H_terms).to_matrix()

# extract clamped hamiltonian conditioned on visible state v
def build_clamped_hamiltonian(b, w, v, gamma=Gamma):
    eff_b = b.copy()
    # update bias of hidden units based on fixed visible state
    for j in range(n_visible, n_qubits):
        eff_b[j] += sum(w[j, i] * v[i] for i in range(n_visible))
    return build_hamiltonian(eff_b, w, gamma=gamma)

# computes thermal expectation (observation)
def boltzmann_expectation(H, ops):
    eH = expm(-H)
    Z = np.trace(eH)
    rho = eH / Z
    return [np.trace(rho @ op.to_matrix()).real for op in ops]

# example 1 : data distro biased
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1
P_data = np.array([np.exp(0.5 * np.sum(v)) for v in data])
P_data /= np.sum(P_data)

# example 2 : Gaussian
# P_data = np.array([np.exp(-((i - 7.5)**2) / 5) for i in range(2**n_visible)])
# P_data /= np.sum(P_data)


# example 3 : XOR-like
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1  # Convert to ±1

# example 4 : XOR-like (even parity) distribution
parities = np.sum((data + 1) // 2, axis=1) % 2
P_data = np.where(parities == 0, 1.0, 1e-6)
P_data /= np.sum(P_data)

# example 5 : correlation and anti-correlation favoring
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1
P_data = np.zeros(len(data))
for i, v in enumerate(data):
    corr_03 = 1 if v[0] == v[3] else -1
    anticorr_12 = 1 if v[1] != v[2] else -1
    score = corr_03 + anticorr_12
    P_data[i] = np.exp(score)

P_data /= np.sum(P_data)


# param initialization
np.random.seed(42)
b = np.random.randn(n_qubits)
w = np.random.randn(n_qubits, n_qubits)
w = (w + w.T) / 2
np.fill_diagonal(w, 0)

# operators
all_ops_z = [pauli_op('Z', i, n_qubits) for i in range(n_qubits)]
all_ops_zz = [[pauli_op('Z', i, n_qubits) @ pauli_op('Z', j, n_qubits)
               for j in range(n_qubits)] for i in range(n_qubits)]

kl_values = []

# training loop
for epoch in range(50):
    d_b = np.zeros_like(b)
    d_w = np.zeros_like(w)
    # compute model expects
    H = build_hamiltonian(b, w)
    avg_z = boltzmann_expectation(H, all_ops_z)
    avg_zz = [[boltzmann_expectation(H, [all_ops_zz[i][j]])[0]
               for j in range(n_qubits)] for i in range(n_qubits)]

    for v_idx, v in enumerate(data):
        # for each data point v compute expectations (clamped)
        Hv = build_clamped_hamiltonian(b, w, v)
        z_expect = boltzmann_expectation(Hv, all_ops_z)
        zz_expect = [[boltzmann_expectation(Hv, [all_ops_zz[i][j]])[0]
                      for j in range(n_qubits)] for i in range(n_qubits)]
        # update params
        d_b += eta * P_data[v_idx] * (np.array(z_expect) - avg_z)
        d_w += eta * P_data[v_idx] * (np.array(zz_expect) - avg_zz)

    b += d_b
    w += d_w
    np.fill_diagonal(w, 0)

    # Kl divergence
    Z = np.trace(expm(-H))      # partition function
    log_probs = []
    for v in data:
        Hv = build_clamped_hamiltonian(b, w, v)
        eHv = expm(-Hv)
        Pv = np.trace(eHv) / Z
        log_probs.append(np.log(Pv.real + 1e-12))

    KL = -np.sum(P_data * np.array(log_probs)) - np.sum(P_data[P_data > 0] * np.log(P_data[P_data > 0]))
    kl_values.append(KL.real)
    print(f"Epoch {epoch}: ∆b norm = {np.linalg.norm(d_b):.4f}, ∆w norm = {np.linalg.norm(d_w):.4f}, KL = {KL:.4f}")


# KL divergence plot
plt.figure(figsize=(8,5))
plt.plot(range(len(kl_values)), kl_values, marker='o', label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.title('KL Divergence During Training')
plt.grid(True)
plt.legend()
plt.show()

# final Hamiltonian
H_final = build_hamiltonian(b, w)
print("Sample Hamiltonian matrix (first 4 rows):\n", H_final[:4, :4])
# final partition function
Z_final = np.trace(expm(-H_final))

qbm_probs = []
unnormalized = []
for v in data:
    Hv = build_clamped_hamiltonian(b, w, v)
    eHv = expm(-Hv)
    Pv = np.trace(eHv) / Z_final
    unnormalized.append(Pv.real)

Z_total = sum(unnormalized)
qbm_probs = [p / Z_total for p in unnormalized]

# Plot target vs QBM-learnt distro
indices = np.arange(len(data))
width = 0.4

plt.figure(figsize=(10,6))
plt.bar(indices - width/2, P_data, width=width, alpha=0.7, label='Data')
plt.bar(indices + width/2, qbm_probs, width=width, alpha=0.7, label='QBM Model')
plt.xlabel('Visible State Index')
plt.ylabel('Probability')
plt.title('Data vs QBM Model Probability Distribution')
plt.legend()
plt.show()

# coupling matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(w, annot=True, cmap='coolwarm', center=0)
plt.title('Coupling Matrix w')
plt.show()

# bias plot
plt.figure(figsize=(6,4))
plt.bar(range(n_qubits), b)
plt.xlabel('Qubit Index')
plt.ylabel('Bias (b)')
plt.title('Bias Vector b')
plt.show()

# print most probable visible state
best_idx = np.argmax(qbm_probs)
best_state = data[best_idx]
print(f"Most probable visible state: {best_state} with probability {qbm_probs[best_idx]:.4f}")

# print results
print(f"{'Visible State':<20} | {'Data Probability':<16} | {'QBM Probability'}")
print("-" * 60)
for v, pdata, pqbm in zip(data, P_data, qbm_probs):
    print(f"{str(v):<20} | {pdata:<16.4f} | {pqbm:.4f}")


