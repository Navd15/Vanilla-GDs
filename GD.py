# %% [markdown]
# ### **Vailla implementation of Gradient Descent to minimize MSE from scratch**

# %%
import numpy as np
import math
from sklearn.datasets import make_regression

# %%
def _loss_fn(theta,y,x,idx=-1):
    """
    Defines a loss (h(theta.x)-y)*2 over single sample

    Parameters:
        theta (list or numpy array) 1-D: The weights for the function
        y (list or numpy array) 1-D: Truth values
        x (2-D list or numpy array): samples (x_i)
        idx (int) default=-1: index of theta. if other than -1 that means _loss_fn called for derivative. 
    Returns:
        float
    """
    if idx>=0:
        return (np.matmul(theta,x) - y) 
    return (np.matmul(theta,x) - y)**2

# %%
def _cost_fn(theta,y,x,theta_idx=-1)-> float:
    """
    Defines cost MSE over a dataset x

    Parameters:
        theta (list or numpy array) 1-D: The weights for the function
        y (list or numpy array) 1-D: Truth values
        x (2-D list or numpy array): samples (x_i)
        theta_idx (int) default=-1: index of theta. if other than -1 that means _cost_fn called for derivative. 
    Returns:
        summation (float): total cost over the dataset
    """
    rows=len(x)
    summation=0
    for i,j in enumerate(x):
        #  if greater than -1 then calculate the derivative wrt to theta[theta_idx]
         if theta_idx>=0:
            summation+=_loss_fn(theta,y[i],x[i],theta_idx)*x[i][theta_idx]
         else:
            summation+=_loss_fn(theta,y[i],x[i]) 
    if theta_idx>=0:
        return summation/rows
    # divide by 2 if not called for derivative
    return summation/(2*rows)

# %%
def min_fn(y,x,alpha):
    """
    minimize MSE cost and updates global theta

    Parameters:
        y (list or numpy array) 1-D: Truth values
        x (list or numpy array) 2-D: dataset.
        alpha (float): learning rate
    Returns:
        None
    """
    # initialize theta as zeros with dimensions x+1 for bias/intercept
    theta=[0]*(1+len(x[0]))
    # add x_0=1 to all the datapoints 
    x=np.hstack(([[1]]*len(x),x))
    change=math.inf
    # calculate th
    old_cost=_cost_fn(theta,y,x)
    
    latest_cost=math.inf
    # infinite loop
    while True:
        # if latest_cost==math.inf then its first time; dont check change
        if latest_cost!=math.inf:
            # as the cost is not changing very much we have reached a miniumum.
            if change<0.000001:
                break
            # update the old cost with latest cost calculated with new weights
            old_cost=latest_cost
        '''
        generate new theta into temp_theta 
        but we dont update until all the thetas are updated in a single loop
        '''
        temp_theta=[0]*len(theta)
        for i,j in enumerate(theta):
            temp_theta[i]=theta[i]- (alpha * _cost_fn(theta,y,x,i))
        # update theta when an epoch is done
        theta=temp_theta
        # calculate the new cost with new weights
        latest_cost=_cost_fn(theta,y,x)
        # calculate the change 
        change=old_cost-latest_cost
    return theta

# %%
x,y=make_regression(10,n_features=2,n_targets=1)

# %%
min_fn(y,x,0.0001)