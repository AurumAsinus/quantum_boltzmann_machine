import numpy as np
from functools import reduce
from scipy.linalg import expm
from qiskit.opflow import I as QI, X as QX, Y as QY, Z as QZ
import matplotlib.pyplot as plt
import seaborn as sns
import time

# hyperparameters
n_visible = 4
n_hidden = 2
n_qubits = n_visible + n_hidden
eta = 0.1  # learning rate

# construct n-qubit operator
def multi_pauli_op(pauli_map, n):
    base_ops = {'I': QI, 'X': QX, 'Y': QY, 'Z': QZ}
    result = None
    for i in range(n):
        op_char = pauli_map.get(i, 'I')
        op = base_ops[op_char]
        result = op if result is None else result ^ op
    return result

def pauli_op(op_char, idx, n):
    return multi_pauli_op({idx: op_char}, n)

# construct full hamiltonian
def build_hamiltonian(b, w):
    H_terms = []
    for i in range(n_qubits):
        H_terms.append((-b[i]) * pauli_op('Z', i, n_qubits))        # bias terms
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            if not (i >= n_visible and j >= n_visible):             # No h-h connections
                zz_op = pauli_op('Z', i, n_qubits) @ pauli_op('Z', j, n_qubits)
                H_terms.append((-w[i, j]) * zz_op)                  # interaction terms
    return reduce(lambda x, y: x + y, H_terms).to_matrix()

# hamiltonian with visible state fixed
def build_clamped_hamiltonian(b, w, v):
    eff_b = b.copy()
    for j in range(n_visible, n_qubits):
        eff_b[j] += sum(w[j, i] * v[i] for i in range(n_visible))
    return build_hamiltonian(eff_b, w)

# compute expectations of observables under Boltzmann distro
def boltzmann_expectation(H, ops):
    eH = expm(-H)
    Z = np.trace(eH)
    rho = eH / Z
    return [np.trace(rho @ op.to_matrix()).real for op in ops]

# data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
# data = 2 * data - 1
# P_data = np.array([np.exp(0.5 * np.sum(v)) for v in data])
# P_data /= np.sum(P_data)


# even parity high prob
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1  # Convert to ±1 encoding (spin 1/2)
parities = np.sum((data + 1) // 2, axis=1) % 2
P_data = np.where(parities == 0, 1.0, 1e-6)
P_data /= np.sum(P_data)


# data distro : (0,3) correlated , (1, 2) anti-correlated
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1

P_data = np.zeros(len(data))
for i, v in enumerate(data):
    corr_03 = 1 if v[0] == v[3] else -1
    anticorr_12 = 1 if v[1] != v[2] else -1
    score = corr_03 + anticorr_12
    P_data[i] = np.exp(score)
P_data /= np.sum(P_data)


# # Favor states where most visible spins align (ferromagnetic dataset)
# def ferromagnetic_probabilities(data):
#     spin_sums = np.sum(data, axis=1)
#     # assign higher probability to states with spins aligned
#     scores = np.abs(spin_sums)
#     P = np.exp(scores)  
#     P /= np.sum(P)      
#     return P

# P_data = ferromagnetic_probabilities(data)


# Simple biased distribution
alpha = 1.5
data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2**n_visible)])
data = 2 * data - 1  # convert to ±1

P_data = np.exp(alpha * np.sum(data, axis=1))
P_data /= np.sum(P_data)


data = np.array([list(map(int, format(i, f'0{n_visible}b'))) for i in range(2 ** n_visible)])
data = 2 * data - 1
P_data = np.zeros(len(data))
for i, v in enumerate(data):
    score = 0
    # (0,1) correlation
    if v[0] == v[1]:
        score += 1
    # (2,3) anti-correlation
    if v[2] != v[3]:
        score += 1
    P_data[i] = np.exp(score)

# normalize
P_data /= np.sum(P_data)


# parameter initialization
np.random.seed(42)
b = np.random.randn(n_qubits)
w = np.random.randn(n_qubits, n_qubits)
w = (w + w.T) / 2           # symmetric weight matrix
np.fill_diagonal(w, 0)

# Z, ZZ operators
all_ops_z = [pauli_op('Z', i, n_qubits) for i in range(n_qubits)]
all_ops_zz = [[pauli_op('Z', i, n_qubits) @ pauli_op('Z', j, n_qubits)
               for j in range(n_qubits)] for i in range(n_qubits)]

kl_values = []


# training loop
start_time = time.time()
for epoch in range(50):
    d_b = np.zeros_like(b)
    d_w = np.zeros_like(w)
    
    # compute model expectations (free)
    H = build_hamiltonian(b, w)
    avg_z = boltzmann_expectation(H, all_ops_z)
    avg_zz = [[boltzmann_expectation(H, [all_ops_zz[i][j]])[0]
               for j in range(n_qubits)] for i in range(n_qubits)]


    # compute data-dependent (clamped) expectations
    for v_idx, v in enumerate(data):
        Hv = build_clamped_hamiltonian(b, w, v)
        z_expect = boltzmann_expectation(Hv, all_ops_z)
        zz_expect = [[boltzmann_expectation(Hv, [all_ops_zz[i][j]])[0]
                      for j in range(n_qubits)] for i in range(n_qubits)]
        # gradient copmutations
        d_b += eta * P_data[v_idx] * (np.array(z_expect) - avg_z)
        d_w += eta * P_data[v_idx] * (np.array(zz_expect) - avg_zz)
    # update params
    b += d_b
    w += d_w
    np.fill_diagonal(w, 0)

    # KL divergence computation between P_data and model
    Z = np.trace(expm(-H))
    log_probs = []
    for v in data:
        Hv = build_clamped_hamiltonian(b, w, v)
        eHv = expm(-Hv)
        Pv = np.trace(eHv) / Z
        log_probs.append(np.log(Pv.real + 1e-12))  # avoid log(0)

    KL = -np.sum(P_data * np.array(log_probs)) - np.sum(P_data[P_data > 0] * np.log(P_data[P_data > 0]))
    kl_values.append(KL.real)
    print(f"Epoch {epoch}: ∆b norm = {np.linalg.norm(d_b):.4f}, ∆w norm = {np.linalg.norm(d_w):.4f}, KL = {KL:.4f}")

end_time = time.time()
elapsed = end_time - start_time

# KL divergence over epochs
plt.figure(figsize=(8, 5))
plt.plot(range(len(kl_values)), kl_values, marker='o', label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.title('KL Divergence During Training (Classical QBM)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Final probabilities
H_final = build_hamiltonian(b, w)
Z_final = np.trace(expm(-H_final))

unnormalized = []
for v in data:
    Hv = build_clamped_hamiltonian(b, w, v)
    eHv = expm(-Hv)
    Pv = np.trace(eHv) / Z_final
    unnormalized.append(Pv.real)

Z_total = sum(unnormalized)
qbm_probs = [p / Z_total for p in unnormalized]

# Data vs QBM probabilities
visible_labels = [''.join(str(int((s + 1) / 2)) for s in v) for v in data]

x = np.arange(len(data))
width = 0.4
plt.figure(figsize=(12, 6))
plt.bar(x - width/2, P_data, width=width, alpha=0.7, label='Target Distribution')
plt.bar(x + width/2, qbm_probs, width=width, alpha=0.7, label='QBM Output')
plt.xticks(x, visible_labels, rotation=90)
plt.xlabel('Visible State')
plt.ylabel('Probability')
plt.title('Data vs QBM Probability Distribution')
plt.legend()
plt.tight_layout()
plt.show()

# heatmap of couplings
plt.figure(figsize=(6, 5))
sns.heatmap(w, annot=True, fmt=".2f", cmap='coolwarm', center=0,
            xticklabels=[f'q{i}' for i in range(n_qubits)],
            yticklabels=[f'q{i}' for i in range(n_qubits)])
plt.title('Coupling Matrix w')
plt.tight_layout()
plt.show()

# bias vector bar plot
plt.figure(figsize=(6, 4))
plt.bar(range(n_qubits), b)
plt.xlabel('Qubit Index')
plt.ylabel('Bias (b)')
plt.title('Bias Vector b')
plt.xticks(range(n_qubits), [f'q{i}' for i in range(n_qubits)])
plt.tight_layout()
plt.show()

# final state evaluation
best_idx = np.argmax(qbm_probs)
best_state = data[best_idx]
print(f"\nMost probable visible state: {best_state} with probability {qbm_probs[best_idx]:.4f}")

# compare all probabilities
print(f"\n{'Visible State':<20} | {'Data Probability':<16} | {'QBM Probability'}")
print("-" * 60)
for v, pdata, pqbm in zip(data, P_data, qbm_probs):
    bin_str = ''.join(str(int((s + 1) / 2)) for s in v)
    print(f"{bin_str:<20} | {pdata:<16.4f} | {pqbm:.4f}")

# energy landscape of visible states
visible_energies = []
for v in data:
    Hv = build_clamped_hamiltonian(b, w, v)
    energy = np.real(np.trace(Hv @ expm(-Hv)) / np.trace(expm(-Hv)))
    visible_energies.append(energy)

plt.figure(figsize=(10, 5))
plt.plot(range(len(data)), visible_energies, marker='o', color='darkgreen')
plt.xticks(range(len(data)), visible_labels, rotation=90)
plt.title('Energy Landscape of Visible States')
plt.xlabel('Visible State')
plt.ylabel('Average Energy ⟨H⟩')
plt.grid(True)
plt.tight_layout()
plt.show()
