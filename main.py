from generate_samples import generate_samples
from discrete_OT import discrete_OT
from discrete_cost import discrete_cost
from plot_discrete_OT import plot_discrete_OT
from gaussian_OT import gaussian_OT
from comparing_plot import comparing_plot
import numpy as np

n = 20 #number of samples
d= 3 #dimension of the ambient space
k =2 #dimension of the projection subspace
lambda1 =200
lambda2 =50
decay1 = 0.2
decay2= 0.1
decay_type1 = "geometric"
decay_type2 = "geometric"
mean2 = 10* np.ones(d)


X,Y, S, Sbis = generate_samples(d,lambda1,lambda2,decay1,decay2,decay_type1,decay_type2,n,mean2)


X_proj = X[:,:k]
Y_proj = Y[:,:k]

T_proj,cost_mat_proj = discrete_OT(X_proj,Y_proj)
T_high,cost_mat_high = discrete_OT(X,Y)
cost_proj = discrete_cost(T_proj,X,Y)
cost_high= discrete_cost(T_high,X,Y)
A_real,cost_real = gaussian_OT(S,Sbis,np.zeros(d),mean2)

print(cost_real)
print(cost_high)
print(cost_proj)

comparing_plot(np.zeros(d),S,mean2,Sbis,X,Y,T_high,T_proj)



