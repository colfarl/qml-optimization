# experiments/cvar_portfolio/runner.py

from qiskit_optimization.converters import LinearEqualityToPenalty
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

from experiments.cvar_portfolio.components.ansatz import get_ansatz
from experiments.cvar_portfolio.components.optimizers import get_optimizer
from experiments.cvar_portfolio.components.solvers import get_solver



def run_experiment(qp, penalty=10, alphas=[1.0, 0.5, 0.25], ansatz_name="RealAmplitudes",
                   optimizer_name="COBYLA", maxiter=100, reps=1, seed=42):
    """
    Runs a CVaR-optimized VQE experiment over different alpha levels.

    Args:
        qp (QuadraticProgram): Qiskit-formatted portfolio optimization problem.
        penalty (float): Penalty for equality constraint enforcement.
        alphas (list): List of confidence levels (alpha) for CVaR objective.
        ansatz_name (str): Name of ansatz circuit to use.
        optimizer_name (str): Classical optimizer name.
        maxiter (int): Maximum optimizer iterations.
        reps (int): Repetitions (layers) in ansatz.
        seed (int): Seed for reproducibility.

    Returns:
        results (dict): alpha → OptimizationResult
        history (dict): alpha → Objective history
    """
    np.random.seed(seed)

    # Convert QP to unconstrained Ising problem
    converter = LinearEqualityToPenalty(penalty=penalty)
    qp_penalized = converter.convert(qp)
    _, offset = qp_penalized.to_ising()

    num_qubits = qp.get_num_binary_vars()
    ansatz = get_ansatz(ansatz_name, num_qubits, reps=reps)
    optimizer = get_optimizer(optimizer_name, maxiter=maxiter)

    results = {}
    history = {alpha: [] for alpha in alphas}

    def callback(i, params, obj, stddev, alpha=None):
        history[alpha].append(np.real_if_close(-(obj + offset)))

    for alpha in alphas:
        solver = get_solver("samplingvqe",
                            ansatz=ansatz,
                            optimizer=optimizer,
                            backend="sampler",
                            aggregation=alpha,
                            callback=lambda i, p, o, s, a=alpha: callback(i, p, o, s, a))
        result = solver.solve(qp_penalized)
        results[alpha] = result
        print(f"[alpha={alpha:.2f}] fval = {result.fval:.4f} → solution = {result.variables_dict}")


    return results, history
