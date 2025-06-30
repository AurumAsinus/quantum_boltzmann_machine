import quimb as qu
import quimb.tensor as qtn
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
import math
from training import QBM, train_qbm  


# Hamiltonian operators
ham_ops = [qu.ikron(qu.pauli('Z'), [2, 2], inds=0),  # Z on qubit 0
           qu.ikron(qu.pauli('X'), [2, 2], inds=1)]  # X on qubit 1

ham_ops = [
    qu.ikron(qu.pauli('X') @ qu.pauli('X'), [2, 2], inds=[0, 1]),  # XX
    qu.ikron(qu.pauli('Z') @ qu.pauli('Z'), [2, 2], inds=[0, 1])   # ZZ
]
ham_ops = [
    qu.ikron(qu.pauli('Z'), [2, 2], inds=[0, 1]),  # Z0Z1 (Ising interaction)
    qu.ikron(qu.pauli('X'), [2, 2], inds=[0]),     # X0 (transverse field)
    qu.ikron(qu.pauli('X'), [2, 2], inds=[1])      # X1 (transverse field)
]

ham_ops = [
    qu.ikron(qu.pauli('X'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('X'), [2, 2], inds=[1]),  # X0X1
    qu.ikron(qu.pauli('Y'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('Y'), [2, 2], inds=[1]),  # Y0Y1
    qu.ikron(qu.pauli('Z'), [2, 2], inds=[0]) @ qu.ikron(qu.pauli('Z'), [2, 2], inds=[1])   # Z0Z1
]

ham_ops = [
    qu.ikron(qu.pauli('X'), [2, 2, 2], inds=[0, 1]),  # X0X1
    qu.ikron(qu.pauli('X'), [2, 2, 2], inds=[1, 2]),  # X1X2
    qu.ikron(qu.pauli('Z'), [2, 2, 2], inds=[0]),     # Z0
    qu.ikron(qu.pauli('Z'), [2, 2, 2], inds=[1]),     # Z1
    qu.ikron(qu.pauli('Z'), [2, 2, 2], inds=[2])      # Z2
]
ham_ops = [
    qu.ikron(qu.pauli('X'), [2, 2], inds=[0]),  # X0
    qu.ikron(qu.pauli('X'), [2, 2], inds=[1]),  # X1
    qu.ikron(qu.pauli('Z'), [2, 2], inds=[0]),  # Z0
    qu.ikron(qu.pauli('Z'), [2, 2], inds=[1]),  # Z1
]


#   ground state calculation

# compute total Ham
H_total = sum(ham_ops)
print(f"\nH_total:\n{H_total}\n")

# compute ground state
eigvals, eigvecs = np.linalg.eigh(H_total)
ground_state = eigvecs[:, np.argmin(eigvals)]
ground_state = ground_state / np.linalg.norm(ground_state)
print(f"Ground state vector: {ground_state}")


# target state to approximate through Hamiltonian

# param initialization
coeffs = np.random.uniform(-1, 1, len(ham_ops))
# target 1
psi_target = qu.qu([0.707, 0, 0, 0.707]).reshape((4, 1))
# target 2
alpha = 1.0 
psi_target = qu.qu([np.exp(-alpha**2/2) * alpha**i / np.sqrt(math.factorial(i)) for i in range(4)]).reshape((4, 1))
# target 3
psi_target = qu.qu([1, 1, 1, 1, 0, 0, 0, 0])
# target 4
psi_target = qu.qu([1/np.sqrt(3), 0, 0, 0, 1/np.sqrt(3), 0, 0, 1/np.sqrt(3)])  # W state
# target 5
psi_target = qu.qu([ 1.930281e-17,  2.319206e-01, -1.032144e-01, -1.665335e-16,  2.319206e-01,  8.326673e-17,       
  0.000000e+00, -9.390274e-01])

# random target state
# coeffs = np.random.rand(2) + 1j * np.random.rand(2)
# coeffs /= np.linalg.norm(coeffs)  # normalize to from valid state
# psi_target_random = qu.qarray(coeffs)

#psi_target = psi_target_random

# target 6
target_output = [0, 1, 1, 0]
psi_target = np.array([np.sqrt(p) if p > 0 else 0 for p in target_output]) 
psi_target /= np.linalg.norm(psi_target)
# target 7
psi_target = qu.qu([0, 0.707, 0.707, 0]).reshape((4, 1))
#target_output = [0, 0.5, 0.5, 0]
print(f"Shape of psi_target: {psi_target.shape}") 
print(f"State (psi_target):\n {psi_target}")

print(f"----- TARGET ------ {psi_target}")

# target density matrix
eta = qu.outer(psi_target, psi_target)  # ρ = |ψ><ψ|

print("Target density matrix (eta):\n", eta)
print("Eigenvalues of eta:", qu.eigvalsh(eta))

print("eta shape:", eta.shape)
for i, op in enumerate(ham_ops):
    print(f"ham_ops[{i}] shape:", op.shape)

# target expects for each Ham op
target_expects = [qu.expec(eta, op) for op in ham_ops]
print(f"Target Expectations (out of loop): {target_expects}")

# eigenvalues of target
eta_evals = qu.eigvalsh(eta)
print(f"Eta Eigenvalues: {eta_evals}")

#   qbm training

# model initialization
qbm = QBM(ham_ops, coeffs)

qre_history = []

# training
trained_qbm, grad_history, qre_history = train_qbm(
    qbm,
    target_expects,
    learning_rate=0.1,
    epochs=100,
    eps=1e-6,
    sigma=0.01, # shot noise
    compute_qre=True,
    target_eta=eta,
    target_eta_ev=eta_evals
)


print("QRE History : ", qre_history)

plt.plot(grad_history, label="Max Gradient")
plt.xlabel("Epoch")
plt.ylabel("Gradient")
plt.title("Gradient Descent in QBM Training")
plt.legend()
plt.show()

plt.plot(qre_history, label="Quantum Relative Entropy")
plt.xlabel("Epoch")
plt.ylabel("QRE")
plt.title("Quantum Relative Entropy during Training")
plt.legend()
plt.show()

