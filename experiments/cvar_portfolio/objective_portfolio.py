# This file contains adaptations from IBM Qiskit CVaR Optimization Tutorial
# https://qiskit-community.github.io/qiskit-optimization/tutorials/08_cvar_optimization.html
# Original copyright (c) IBM 2017, 2024.
# Modified by Colin Farley (2025) for research use at HARP Research.

from qiskit_algorithms import NumPyMinimumEigensolver, SamplingVQE, QAOA
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer, CplexOptimizer
from qiskit_optimization.converters import LinearEqualityToPenalty
import numpy as np

from docplex.mp.model import Model

def portfolio_to_qp(mu, sigma, risk_factor=0.5, budget_percent=2):

    """
    Constructs a quadratic program for portfolio optimization.

    This function formulates the classical mean-variance portfolio optimization problem,
    where the objective is to maximize expected return while minimizing risk 
    (modeled as covariance between asset returns). A budget constraint limits the number 
    of assets selected. The resulting model is returned as a Qiskit QuadraticProgram object, 
    suitable for classical or quantum solution methods.

    Args:
        mu (np.ndarray): Array of expected returns for each asset.
        sigma (np.ndarray): Covariance matrix representing asset risk correlations.
        risk_factor (float): Trade-off parameter (lambda) that scales the risk penalty term.
        budget_percent (int): Denominator for budget fraction. E.g., 2 means select n/2 assets.

    Returns:
        QuadraticProgram: A Qiskit optimization object representing the portfolio problem.
    """

    # Initialize docplex model
    mdl = Model("portfolio_optimization")

    # Derive necessary variables
    n = len(mu)
    budget = n // budget_percent 
    q = risk_factor             

    # Formulate Problem:     
    x = mdl.binary_var_list(range(n), name="x")                                                 # create list of binary variables
    objective = mdl.sum([mu[i] * x[i] for i in range(n)])                                       # pair reward (expected return) with given asset
    objective -= q * mdl.sum([sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n)])   # associate penalty for covariance

    # Create docplex model
    mdl.maximize(objective)
    mdl.add_constraint(mdl.sum(x[i] for i in range(n)) == budget)

    # Convert docplex model to quadratic program
    return from_docplex_mp(mdl)

def portfolio_add_penalty(qp, penalty):
    """
    Converts a constrained QuadraticProgram to an unconstrained QuadraticProgram.

    This is required for variational quantum algorithms (e.g., SamplingVQE),
    which cannot natively handle constraints. The equality constraint is converted
    to a penalty term in the objective function.

    Args:
        qp (QuadraticProgram): The constrained portfolio optimization problem.
        penalty (float): Penalty multiplier to enforce the budget constraint.

    Returns:
        qp (Quadratic Program): The unconstrained equivalent with the penalty added to the equation
    """

    linear2penalty = LinearEqualityToPenalty(penalty=penalty)
    qp = linear2penalty.convert(qp)
    return qp

def classic_solve(qp):
    """
    Solves a constrained QuadraticProgram classically using an exact eigensolver.

    This is useful for benchmarking quantum variational results against the true
    optimal solution.

    Args:
        qp (QuadraticProgram): The constrained portfolio optimization problem.

    Returns:
        OptimizationResult: Solution containing optimal value, variable assignment, and status.
    """
    return MinimumEigenOptimizer(NumPyMinimumEigensolver()).solve(qp)

def classic_relaxed_solve(qp):
    return CplexOptimizer().solve(qp)

def solve_with_sampling_vqe(qp, sampler, ansatz, optimizer, offset, alphas, initial_point=None):
    """
    Solves an unconstrained portfolio optimization problem using SamplingVQE with optional CVaR.

    This function supports multiple confidence levels (alpha) and tracks optimization convergence
    for each run. It can be used for both standard VQE (alpha=1.0) and CVaR-VQE (alpha < 1.0).

    Args:
        qp (QuadraticProgram): Unconstrained QUBO for portfolio optimization.
        sampler (BaseSampler): Qiskit Sampler primitive (e.g., Sampler()).
        ansatz (QuantumCircuit): Parameterized quantum circuit used to prepare trial states.
        optimizer (Optimizer | Minimizer): Classical optimizer for variational parameters.
        offset (float): Constant offset from Ising transformation (added back to computed values).
        alphas (List[float]): List of confidence levels (CVaR α-values) to evaluate.
        initial_point (np.ndarray | None): Optional initial parameter vector for the ansatz.

    Returns:
        Tuple[dict, dict]:
            - results: Dictionary mapping alpha -> OptimizationResult
            - objectives: Dictionary mapping alpha -> list of intermediate objective values per iteration
    """
    results = {}
    objectives = {alpha: [] for alpha in alphas}

    def make_callback(alpha):
        return lambda i, params, obj, stddev: objectives[alpha].append(np.real_if_close(-(obj + offset)))

    for alpha in alphas:
        vqe = SamplingVQE(
            sampler=sampler,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=initial_point,
            aggregation=alpha,
            callback=make_callback(alpha),
        )
        opt_alg = MinimumEigenOptimizer(vqe)
        results[alpha] = opt_alg.solve(qp)

    return results, objectives


def solve_with_qaoa(qp, sampler, optimizer, offset, alphas, reps=1, initial_point=None, initial_state=None, mixer=None):
    """
    Solves an unconstrained portfolio optimization problem using QAOA with optional CVaR.

    This function supports varying circuit depths (reps), confidence levels (alpha), and
    allows for warm-started and mixer-constrained variants of QAOA.

    Args:
        qp (QuadraticProgram): Unconstrained QUBO for portfolio optimization.
        sampler (BaseSampler): Qiskit Sampler primitive (e.g., Sampler()).
        optimizer (Optimizer | Minimizer): Classical optimizer for variational parameters.
        offset (float): Constant offset from Ising transformation (added back to computed values).
        alphas (List[float]): List of confidence levels (CVaR α-values) to evaluate.
        reps (int): Number of QAOA layers (circuit depth). Default is 1.
        initial_point (np.ndarray | None): Optional initial parameter vector [β₀, γ₀, ..., βₚ₋₁, γₚ₋₁].
        initial_state (QuantumCircuit | None): Optional circuit to prepend before QAOA layers.
        mixer (QuantumCircuit | BaseOperator | None): Custom mixer for constrained optimization.

    Returns:
        Tuple[dict, dict]:
            - results: Dictionary mapping alpha -> OptimizationResult
            - objectives: Dictionary mapping alpha -> list of intermediate objective values per iteration
    """

    results = {}
    objectives = {alpha: [] for alpha in alphas}

    def make_callback(alpha):
        return lambda i, params, obj, stddev: objectives[alpha].append(np.real_if_close(-(obj + offset)))

    for alpha in alphas:
        qaoa = QAOA(
            sampler=sampler,
            optimizer=optimizer,
            reps=reps,
            initial_state=initial_state,
            mixer=mixer,
            initial_point=initial_point,
            aggregation=alpha,
            callback=make_callback(alpha),  # ← safe binding
        )
        opt_alg = MinimumEigenOptimizer(qaoa)
        results[alpha] = opt_alg.solve(qp)

    return results, objectives


def format_qaoa_samples(samples, max_len: int = 10):
    """
    Filter and format QAOA samples for display (works for sampling VQE as well).

    Filters only feasible samples (sum(x) == 3), then sorts them by descending
    objective value and formats them with bitstring, value, and sampling probability.

    Args:
        samples (List[SolutionSample]): Samples returned by MinimumEigenOptimizer.
        max_len (int): Number of top entries to display.

    Returns:
        List[str]: Formatted summary strings for the top samples.
    """
    qaoa_res = []
    for s in samples:
        if sum(s.x) == 3:
            qaoa_res.append(("".join([str(int(_)) for _ in s.x]), s.fval, s.probability))

    res = sorted(qaoa_res, key=lambda x: -x[1])[0:max_len]

    return [(_[0] + f": value: {_[1]:.3f}, probability: {1e2*_[2]:.1f}%") for _ in res]