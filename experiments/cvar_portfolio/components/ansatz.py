"""
Provides flexible access to various parameterized ansätze used in variational quantum algorithms.
Each ansatz is returned as a Qiskit QuantumCircuit object.

Supported:
- RealAmplitudes
- TwoLocal
- EfficientSU2 (legacy)
- WarmStartQAOAAnsatz (if needed in future)
- Customizable TwoLocal blocks

You can extend this file to support problem-specific or research ansatz designs.
"""

from qiskit.circuit.library import RealAmplitudes, TwoLocal, EfficientSU2


def get_ansatz(name, num_qubits, **kwargs):
    """
    Return a parameterized quantum ansatz circuit by name.

    Args:
        name (str): Name of the ansatz. Options include:
                    - "RealAmplitudes"
                    - "TwoLocal"
                    - "EfficientSU2"
                    - "WarmStartQAOA" (requires pre-solved QP and initial point)
        num_qubits (int): Number of qubits / variables in the problem.
        kwargs: Additional arguments specific to the ansatz, e.g.:
            - reps (int): Number of repetitions/layers (default: 1 or 2)
            - entanglement (str): Pattern for entanglement (e.g. "linear", "full")
            - initial_point (np.ndarray): For WarmStartQAOA
            - qubo (QuadraticProgram): For WarmStartQAOA

    Returns:
        QuantumCircuit: The constructed parameterized ansatz.
    """
    name = name.lower()

    if name == "realamplitudes":
        reps = kwargs.get("reps", 1)
        return RealAmplitudes(num_qubits, reps=reps)

    elif name == "twolocal":
        reps = kwargs.get("reps", 2)
        rotation_blocks = kwargs.get("rotation_blocks", "ry")
        entanglement_blocks = kwargs.get("entanglement_blocks", "cz")
        entanglement = kwargs.get("entanglement", "linear")
        return TwoLocal(num_qubits, rotation_blocks=rotation_blocks,
                        entanglement_blocks=entanglement_blocks,
                        entanglement=entanglement, reps=reps)

    elif name == "efficientsu2":
        reps = kwargs.get("reps", 1)
        return EfficientSU2(num_qubits, reps=reps)

    else:
        raise ValueError(f"Unsupported ansatz type: {name}")
