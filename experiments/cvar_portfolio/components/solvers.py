"""
Wraps quantum solver methods for QUBO/Ising optimization problems.

Supported:
- VQE
- SamplingVQE (CVaR-compatible)
- QAOA

Each solver is returned in a form compatible with Qiskit's MinimumEigenOptimizer.
This module connects:
  • the ansatz
  • the classical optimizer
  • the sampling backend
"""

from qiskit_algorithms import VQE, SamplingVQE, QAOA
from qiskit.primitives import Sampler, Estimator
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def get_solver(method="vqe", ansatz=None, optimizer=None, backend="sampler", aggregation=None, reps=None, callback=None):
    """
    Returns a MinimumEigenOptimizer wrapped around the selected quantum solver.

    Args:
        method (str): Solver type. Options: "vqe", "samplingvqe", "qaoa"
        ansatz (QuantumCircuit): Parameterized quantum circuit
        optimizer: Classical optimizer instance
        backend (str): Either "sampler" or "estimator" — relevant for Qiskit 1.x
        aggregation (float or None): CVaR alpha (0 < alpha ≤ 1). Only used with SamplingVQE.
        reps (int): Number of p-layers for QAOA (if not encoded in ansatz)

    Returns:
        MinimumEigenOptimizer: Ready-to-use optimization object
    """
    method = method.lower()

    if method == "vqe":
        if backend == "estimator":
            solver = VQE(ansatz=ansatz, optimizer=optimizer, estimator=Estimator(), callback=callback)
        else:
            raise ValueError("VQE in Qiskit 1.x requires the 'estimator' backend.")
        return MinimumEigenOptimizer(solver)

    elif method == "samplingvqe":
        if backend != "sampler":
            raise ValueError("SamplingVQE requires the 'sampler' backend.")
        solver = SamplingVQE(ansatz=ansatz, optimizer=optimizer, sampler=Sampler(),
                             aggregation=aggregation, callback=callback)
        return MinimumEigenOptimizer(solver)

    elif method == "qaoa":
        if backend == "sampler":
            solver = QAOA(ansatz=ansatz, optimizer=optimizer, sampler=Sampler(), reps=reps, callback=callback)
        else:
            solver = QAOA(ansatz=ansatz, optimizer=optimizer, estimator=Estimator(), reps=reps, callback=callback)
        return MinimumEigenOptimizer(solver)

    else:
        raise ValueError(f"Unsupported solver type: {method}")
