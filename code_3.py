import quimb as qu
import numpy as np
import matplotlib.pyplot as plt
import math
from training import QBM, train_qbm
import numpy as np
import matplotlib.pyplot as plt


# 2-qubit Hamiltonian with ZZ and XX interactions
ham_ops = [
    qu.ikron(qu.pauli('X'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('X'), [2, 2], inds=[1]),  # XX
    qu.ikron(qu.pauli('Y'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('Y'), [2, 2], inds=[1]),  # YY
    qu.ikron(qu.pauli('Z'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('Z'), [2, 2], inds=[1]),  # ZZ
]

# entangled state: |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩
theta = np.pi / 6  
psi_target = qu.qu([np.cos(theta), 0, 0, np.sin(theta)]).reshape((4, 1))

# density matrix
eta = qu.outer(psi_target, psi_target)
# exepctation values
eta_evals = qu.eigvalsh(eta)
target_expects = [qu.expec(eta, op) for op in ham_ops]
print("Target Expectation Values:", target_expects)

# QBM initialization
coeffs = np.random.uniform(-1, 1, len(ham_ops))
qbm = QBM(ham_ops, coeffs)

# train QBM
trained_qbm, grad_history, qre_history = train_qbm(
    qbm,
    target_expects,
    learning_rate=0.1,
    epochs=100,
    eps=1e-6,
    sigma=0.01,
    compute_qre=True,
    target_eta=eta,
    target_eta_ev=eta_evals
)

# visualize training results
plt.plot(grad_history, label="Max Gradient")
plt.xlabel("Epoch")
plt.ylabel("Gradient")
plt.title("Gradient Descent in QBM Training (Entangled State)")
plt.legend()
plt.show()

# compute expectations from trained QBM
learned_expects = qbm.compute_expectation(ham_ops)

# target vs learnt expects
plt.figure(figsize=(8, 4))
bar_width = 0.35
x = np.arange(len(ham_ops))
plt.bar(x - bar_width/2, target_expects, width=bar_width, label='Target')
plt.bar(x + bar_width/2, learned_expects, width=bar_width, label='Learned')
plt.xticks(x, ['X⊗X', 'Y⊗Y', 'Z⊗Z'])
plt.title('Target vs Learned Expectation Values')
plt.ylabel('Expectation Value')
plt.legend()
plt.tight_layout()
plt.show()

# compare full density matrixes
eta_learned = qbm.get_density_matrix()

# fidelity as metric
fidelity = float(np.abs(qu.tr(eta @ eta_learned)))
print(f"Fidelity between target and learned state: {fidelity:.6f}")

rho_learned = eta_learned

# trace distance
def trace_norm(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    return np.sum(s)

trace_distance = 0.5 * trace_norm((eta - rho_learned).A)
print(f"Trace distance from target: {trace_distance:.6f}")

# plot target and learnt density matrices
def plot_density_matrix(rho, title="Density Matrix"):
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(np.abs(rho.A), cmap='viridis')
    plt.title(title)
    plt.colorbar(cax)
    plt.xlabel("Col")
    plt.ylabel("Row")
    plt.show()

plot_density_matrix(eta, "Target: Bell State")
plot_density_matrix(rho_learned, "QBM Learned State")

