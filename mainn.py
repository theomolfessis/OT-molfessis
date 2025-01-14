import numpy as np
import matplotlib.pyplot as plt
from generate_samples import generate_samples
from discrete_OT import discrete_OT
from discrete_cost import discrete_cost
from gaussian_OT import gaussian_OT
from comparing_plot import comparing_plot



import numpy as np
import matplotlib.pyplot as plt
from generate_samples import generate_samples
from discrete_OT import discrete_OT
from discrete_cost import discrete_cost
from gaussian_OT import gaussian_OT
from comparing_plot import comparing_plot

def plot_cost_and_eigenvalues_evolution(n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, mean2,plot_transfer_idx = -1 ):
    costs_proj = []
    
    # Generate data and eigenvalues
    X, Y, S, Sbis, eigenvalues_S, eigenvalues_Sbis = generate_samples(d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, n, mean2)

    # Discrete OT for full dimensional space
    T_high, _ = discrete_OT(X, Y)
    cost_high = discrete_cost(T_high, X, Y)

    # Gaussian OT cost
    A_real, cost_real = gaussian_OT(S, Sbis, np.zeros(d), mean2)

    for k in range(1, d + 1):
        X_proj = X[:, :k]
        Y_proj = Y[:, :k]

        T_proj, _ = discrete_OT(X_proj, Y_proj)
        cost_proj = discrete_cost(T_proj, X, Y)
        costs_proj.append(cost_proj)

        # Comparing plot should be called meaningfully, possibly not in every iteration
        if k == plot_transfer_idx:  # Possibly call on final iteration or conditionally
            comparing_plot(np.zeros(d), S, mean2, Sbis, X, Y, T_high, T_proj)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting costs
    ks = np.arange(1, d + 1)
    ax1.plot(ks, costs_proj, 'r-', label='Projected Cost', linewidth=2)
    ax1.axhline(y=cost_real, color='b', linestyle='--', linewidth=2, label='Gaussian OT Cost')
    ax1.axhline(y=cost_high, color='k', linestyle='--', linewidth=2, label='High-Dimensional Discrete OT Cost')
    ax1.set_xlabel('Dimension of the Projection Subspace and descending rank of eigenvalue')
    ax1.set_ylabel('Cost')
    ax1.tick_params(axis='y', labelcolor='tab:red')
   
    ax1.legend()
    ax1.grid(True)

    # Eigenvalues plot on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(ks, eigenvalues_S, 'bo-', label='Eigenvalues $\Sigma_1$', markersize=10)
    ax2.plot(ks, eigenvalues_Sbis, 'go-', label='Eigenvalues $\Sigma_2$', markersize=10)

    # Adjust y-axis scale for eigenvalues to ensure visibility
    max_eigenvalue = max(max(eigenvalues_S), max(eigenvalues_Sbis))
    ax2.set_ylim(0, max_eigenvalue * 1.1)  # Extend upper limit for clarity
    ax2.set_ylabel('Eigenvalues')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # Legend
    #lines, labels = ax1.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Evolution of OT Costs and Eigenvalues as k Varies, dim ' +str(d) + ' , with ' +str(n) + ' points' )
    plt.show()




# Example usage configurations
configs = [
    (20, 3, 200, 50, 0.5, 0.7, "geometric", "geometric",2),
    (100, 5, 300, 200, 0.8, 0.9, "geometric", "geometric",3),
    (1000, 4, 150, 400, 25, 0.4, "linear", "geometric",2),
    (100, 10, 1000 , 400, 90 , 35, "linear", "linear",5),
    (500, 20, 1500, 1000, 0.4, 0.35, "geometric", "geometric",8)
]
mean2 = 10 * np.ones(5)  # Mean for the target Gaussian, adjust as necessary per d

for config in configs:
    n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, plot_idx = config
    plot_cost_and_eigenvalues_evolution(n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, mean2[:d],plot_idx)