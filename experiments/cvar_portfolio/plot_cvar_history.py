import matplotlib.pyplot as plt

def plot_cvar_convergence(objectives, opt_val=None, maxiter=None, title=None):
    """
    Plot optimization convergence traces for different alpha values.

    Args:
        objectives (dict): Keys = alpha values, values = list of objective values per iteration.
        opt_val (float): Known optimal value to show as reference (e.g., classical benchmark).
        maxiter (int): Max number of iterations (used for x-axis limits).
        title (str): Optional plot title.
    """
    plt.figure(figsize=(10, 5))
    
    if opt_val is not None and maxiter is not None:
        plt.plot([0, maxiter], [opt_val, opt_val], "r--", linewidth=2, label="classical optimum")

    for alpha, trace in objectives.items():
        plt.plot(trace, label=f"α = {alpha:.2f}", linewidth=2)

    plt.legend(loc="lower right", fontsize=14)
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Objective Value", fontsize=14)
    if title:
        plt.title(title, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if maxiter:
        plt.xlim(0, maxiter)
    plt.grid(True)
    plt.tight_layout()
    plt.show()