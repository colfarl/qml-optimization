"""
Provides flexible access to Qiskit's supported classical optimizers
used in variational quantum algorithms.

Each optimizer is returned as a Qiskit optimizer object, ready to plug into VQE, QAOA, etc.

Supported:
- COBYLA
- SPSA
- NELDER_MEAD
- L_BFGS_B
- SLSQP
- ADAM
- GradientDescent
- QNGBFGS (NaturalGradient)

You can extend this file with wrappers for external optimizers (e.g., SciPy, Nevergrad).
"""

from qiskit_algorithms.optimizers import (
    COBYLA,
    SPSA,
    NELDER_MEAD,
    L_BFGS_B,
    SLSQP,
    ADAM,
    GradientDescent
)

def get_optimizer(name="COBYLA", maxiter=100, **kwargs):
    """
    Return a classical optimizer by name.

    Args:
        name (str): Name of the optimizer. Options include:
                    - "COBYLA", "SPSA", "NELDER_MEAD", "L_BFGS_B", "SLSQP"
                    - "ADAM", "GradientDescent", "QNGBFGS"
        maxiter (int): Maximum number of iterations (where applicable).
        kwargs: Additional keyword args for specific optimizers.

    Returns:
        Optimizer: An instance of the selected Qiskit optimizer.
    """
    name = name.upper()

    if name == "COBYLA":
        return COBYLA(maxiter=maxiter, **kwargs)

    elif name == "SPSA":
        return SPSA(maxiter=maxiter, **kwargs)

    elif name == "NELDER_MEAD":
        return NELDER_MEAD(maxiter=maxiter, **kwargs)

    elif name == "L_BFGS_B":
        return L_BFGS_B(maxiter=maxiter, **kwargs)

    elif name == "SLSQP":
        return SLSQP(maxiter=maxiter, **kwargs)

    elif name == "ADAM":
        return ADAM(maxiter=maxiter, **kwargs)

    elif name == "GRADIENTDESCENT":
        return GradientDescent(maxiter=maxiter, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
