import numpy as np
import matplotlib.pyplot as plt 
import sys
from helper_fns import standard_normalization,plot,plot_quad_boundary

# read data
features=np.loadtxt('Q4_data/q4x.dat')
str_labels=np.loadtxt('Q4_data/q4y.dat',dtype=str)

x1,x2=standard_normalization(features)
m=x1.shape
# alaska=1 canada=0
labels=[1 if i=='Alaska' else 0 for i in str_labels]
alaska_data=[]
canada_data=[]
for i,label in enumerate(labels):
    if label==1: alaska_data.append([x1[i],x2[i]])
    else:canada_data.append([x1[i],x2[i]])
alaska_data=np.array(alaska_data)
canada_data=np.array(canada_data)

alaska_mean=np.array([np.mean(alaska_data[:,0]),np.mean(alaska_data[:,1])])
canada_mean=np.array([np.mean(canada_data[:,0]),np.mean(canada_data[:,1])])
print(f'Alaska mean = {alaska_mean}')
print(f'Canada mean = {canada_mean}')

alaska_data_centered=alaska_data-alaska_mean
canada_data_centered=canada_data-canada_mean
data=np.append(alaska_data_centered,canada_data_centered,axis=0)

sigma=(data.T@data)/m

# question 1
print(f'shared covarience matrix: \n  {sigma}')

# question 2
# plotting data 
plot(alaska_data,canada_data)

# question 3
# computing parameters
inv_sigma=np.linalg.inv(sigma)
w=inv_sigma@(np.array(alaska_mean)-np.array(canada_mean))
print(f'learned parameters: {w}')
plot(alaska_data,canada_data,w)

# question 4
canada_sigma=(canada_data_centered.T@canada_data_centered)/canada_data_centered.shape[0]
alaska_sigma=(alaska_data_centered.T@alaska_data_centered)/alaska_data_centered.shape[0]

print(f'Canada covarience matrix: \n  {canada_sigma}')
print(f'Alaska covarience matrix: \n  {alaska_sigma}')

#question 5
inv_cs=np.linalg.inv(canada_sigma)
inv_as=np.linalg.inv(alaska_sigma)

a=inv_cs-inv_as
b=-2*(inv_cs@canada_mean - inv_as@alaska_mean)
c=(canada_mean.T@inv_cs@canada_mean - alaska_mean.T@inv_as@alaska_mean + 
   np.log(np.linalg.det(alaska_sigma)) -np.log(np.linalg.det(canada_sigma)))

plot_quad_boundary(alaska_data, canada_data, w, alaska_mean, canada_mean, canada_sigma, alaska_sigma)