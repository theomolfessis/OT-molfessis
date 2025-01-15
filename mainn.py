import numpy as np
import matplotlib.pyplot as plt
from generate_samples import generate_samples
from discrete_OT import discrete_OT
from discrete_cost import discrete_cost
from gaussian_OT import gaussian_OT
from comparing_plot import comparing_plot
from algo_one import subspace_gd
import torch
import time


import numpy as np
import matplotlib.pyplot as plt
from generate_samples import generate_samples
from discrete_OT import discrete_OT
from discrete_cost import discrete_cost
from gaussian_OT import gaussian_OT
from comparing_plot import comparing_plot

def plot_cost_and_eigenvalues_evolution(n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, mean2,max_dim, plot_transfer_idx=-1):
    print(f'starting algo for n={n}, d = {d}, lambda1={lambda1}, lambda2={lambda2} and decay of type ' +decay_type1 )
    costs_proj = []
    optimal_subspace_costs = []  # List to store the costs from the optimal subspace
    time_hist = []
    print("generating samples")
    # Generate data and eigenvalues
    X, Y, S, Sbis, eigenvalues_S, eigenvalues_Sbis = generate_samples(d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, n, np.zeros(d))
    print('samples generated')
    # Discrete OT for full dimensional space
    if max_dim == d+1: 
        print('computing full OT')
        stime = time.time()
        T_high, _ = discrete_OT(X, Y)
        endtime = time.time()
        print(f'time for computing the full OT : {endtime -stime} seconds')
        cost_high = discrete_cost(T_high, X, Y)
        

    # Gaussian OT cost
    print('computing gaussian OT')
    stime = time.time()
    A_real, cost_real = gaussian_OT(S, Sbis, np.zeros(d), mean2)
    endtime = time.time()
    print(f'time for computing the full OT between gaussians : {endtime -stime} seconds')



    for k in range(1, max_dim):
        X_proj = X[:, :k]
        Y_proj = Y[:, :k]
        print('done projecting, now discrete OT')
        stime = time.time()
        T_proj, _ = discrete_OT(X_proj, Y_proj)
        endtime = time.time()
        print(f'time for computing the discrete OT with naive subspace in dim {k}: {endtime -stime} seconds')
        cost_proj = discrete_cost(T_proj, X, Y)
        costs_proj.append(cost_proj)

        # Comparing plot should be called meaningfully, possibly not in every iteration
        if k == plot_transfer_idx:  # Possibly call on final iteration or conditionally
            comparing_plot(np.zeros(d), S, np.zeros(d), Sbis, X, Y, T_high, T_proj)
            # Save the figure with configuration parameters in the title
            figure_filename = f'fig/transports_d={d}_n={n}_lambda1={lambda1}_lambda2={lambda2}_k={k}.png'
            plt.savefig(figure_filename)
            plt.close()
        
        # Compute the optimal subspace and associated cost using algorithm_1
        # Set parameters for algorithm_1
        num_iterations = 401  # Number of iterations
        stime = time.time()
        V_MK, MK_cost = subspace_gd(torch.from_numpy(S), torch.from_numpy(Sbis),k, niter=num_iterations ,verbose=True)
        endtime = time.time()
        print(f'time for computing the MK subspace in dim {k}: {endtime -stime} seconds')
        V_k = V_MK.cpu().detach().numpy()[:,:k]
        

        # Compute transports and costs in the optimal subspace
        X_optimal = X @ V_k
        Y_optimal = Y @ V_k
        print(X_optimal.shape)
        stime =time.time()
        T_optimal, _ = discrete_OT(X_optimal, Y_optimal)
        endtime = time.time()
        print(f'time for computing the discrete OT with MK subspace in dim {k}: {endtime -stime} seconds')
        optimal_subspace_cost = discrete_cost(T_optimal, X, Y)
        optimal_subspace_costs.append(optimal_subspace_cost)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plotting costs
    ks = np.arange(1, max_dim )
    ax1.plot(ks, costs_proj, 'r-', label='Projected Cost', linewidth=2)
    ax1.plot(ks, optimal_subspace_costs , 'r--', label='Using algorithm 1', linewidth=2)  # Modified: plot optimal subspace cost
    ax1.axhline(y=cost_real, color='b', linestyle='--', linewidth=2, label='Gaussian OT Cost')
    if max_dim ==d+1: 
        ax1.axhline(y=cost_high, color='k', linestyle='--', linewidth=2, label='High-Dimensional Discrete OT Cost')
    ax1.set_xlabel('Dimension of the Projection Subspace and descending rank of eigenvalue')
    ax1.set_ylabel('Cost')
    ax1.tick_params(axis='y', labelcolor='tab:red')
   
    ax1.legend()
    ax1.grid(True)

    # Eigenvalues plot on secondary y-axis
    ks = np.arange(1, d +1)
    ax2 = ax1.twinx()
    ax2.plot(ks, eigenvalues_S, 'bo-', label='Eigenvalues $\Sigma_1$', markersize=10)
    ax2.plot(ks , eigenvalues_Sbis, 'go-', label='Eigenvalues $\Sigma_2$', markersize=10)

    # Adjust y-axis scale for eigenvalues to ensure visibility
    max_eigenvalue = max(max(eigenvalues_S), max(eigenvalues_Sbis))
    ax2.set_ylim(0, max_eigenvalue * 1.1)  # Extend upper limit for clarity
    ax2.set_ylabel('Eigenvalues')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.title('Evolution of OT Costs and Eigenvalues as k Varies, dim ' +str(d) + ' , with ' +str(n) + ' points' )
  
    # Save the figure with configuration parameters in the title
    figure_filename = f'fig/cost_and_eigenvalues_d={d}_n={n}_lambda1={lambda1}_lambda2={lambda2}.png'
    plt.savefig(figure_filename)
    plt.close()



# Example usage configurations
configs = [
    (300, 10, 300, 2000, 0.5, 0.7, "geometric", "geometric",7,-1),
    (300, 10, 2000, 300, 0.7, 0.5, "geometric", "geometric",7,-1),
    #(100, 16, 1000 , 400, 60 , 20, "linear", "linear",-1),
    #(100, 16, 400 , 1000, 20 , 60, "linear", "linear",-1)
    #(40, 10, 1500, 1000, 0.4, 0.35, "geometric", "geometric",8)
]
# Mean for the target Gaussian, adjust as necessary per d

for config in configs:
    n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, max_dim, plot_idx = config
    plot_cost_and_eigenvalues_evolution(n, d, lambda1, lambda2, decay1, decay2, decay_type1, decay_type2, np.zeros(d),max_dim, plot_idx)
