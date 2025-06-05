from qiskit_optimization.problems.variable import VarType
import copy
from qiskit.circuit import Parameter
from qiskit import QuantumCircuit
import numpy as np

def relax_problem(problem):
    """
    Converts a binary QuadraticProgram into a continuous relaxation.

    This function transforms all binary decision variables into continuous variables
    bounded between 0 and 1. This is useful for generating a relaxed version of 
    the original combinatorial optimization problem, which can then be solved using
    classical convex solvers. The resulting solution can serve as a warm start 
    (soft guidance) for quantum optimization algorithms like QAOA.

    Args:
        problem (QuadraticProgram): The binary (combinatorial) optimization problem.

    Returns:
        QuadraticProgram: A relaxed version of the input problem with continuous variables.
    """
    relaxed_problem = copy.deepcopy(problem)
    for variable in relaxed_problem.variables:
        variable.vartype = VarType.CONTINUOUS

    return relaxed_problem

def generate_mixer_and_initial_state(c_stars, sigma):
    beta = Parameter("β")
    thetas = [2 * np.arcsin(np.sqrt(c_star)) for c_star in c_stars]

    init_qc = QuantumCircuit(len(sigma))
    for idx, theta in enumerate(thetas):
        init_qc.ry(theta, idx)

    ws_mixer = QuantumCircuit(len(sigma))
    for idx, theta in enumerate(thetas):
        ws_mixer.ry(-theta, idx)
        ws_mixer.rz(-2 * beta, idx)
        ws_mixer.ry(theta, idx)

    return init_qc, ws_mixer

