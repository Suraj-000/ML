import numpy as np
import matplotlib.pyplot as plt 
import time


'''
closed form solution  for Linear Regression
'''
np.random.seed(42)

def sample_data(mean,varience,num_samples):
    return np.random.normal(mean,varience,num_samples)

true_parameters = np.array([3,1,2])
num_samples=1000000
epochs=1
learning_rate=0.001
batch_size=[1,80,8000,800000]

parameters=np.array([0,0,0])
x0=np.ones(num_samples)
x1=sample_data(3,2,num_samples)
x2=sample_data(-1,2,num_samples)
noise=sample_data(0,np.sqrt(2),num_samples)
features=np.column_stack((x0,x1,x2))

'''
change the commented line below to add noise to labels
'''
# labels=true_parameters.dot(features.T)+noise
labels=true_parameters.dot(features.T)

cov_matrix=features.T.dot(features)
inv_cov_matrix=np.linalg.inv(cov_matrix)
optimal_parameters=inv_cov_matrix.dot(features.T).dot(labels)
print("Optimal parameters:",optimal_parameters)
