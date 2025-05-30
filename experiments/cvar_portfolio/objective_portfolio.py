# This file contains adaptations from IBM Qiskit CVaR Optimization Tutorial
# https://qiskit-community.github.io/qiskit-optimization/tutorials/08_cvar_optimization.html
# Original copyright (c) IBM 2017, 2024.
# Modified by Colin Farley (2025) for research use at HARP Research.

from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import LinearEqualityToPenalty

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