import numpy as np
import matplotlib.pyplot as plt 
import sys

# read data
features=np.loadtxt('Q3_data/logisticX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q3_data/logisticY.csv',delimiter=',',skiprows=1)

print(features.shape,labels.shape)
# (99, 2) (99,)


def standard_normalization(features):
    m,_=features.shape
    x1=[features[i][0] for i in range(m)]
    x2=[features[i][1] for i in range(m)]
    x1_mean=np.mean(x1)
    x2_mean=np.mean(x2)
    x1_std=np.std(x1)
    x2_std=np.std(x2)
    x1=(x1-x1_mean)/x1_std
    x2=(x2-x2_mean)/x2_std
    return x1,x2

def sigmoid_fn(z):
    return 1/(1+np.exp(-z))

def get_hessian_and_first_derivative(features,labels,parameters):
    m,n=features.shape
    y_pred=sigmoid_fn(features@parameters)

    first_d=features.T@(y_pred-labels)
    diag_matrix=np.diag(y_pred*(1-y_pred))

    hessian=features.T@diag_matrix@features
    return hessian,first_d

def newtons_method(features,labels,parameters,num_iterations=500):
    m,n=features.shape
    for i in range(num_iterations):
        hessian,first_d=get_hessian_and_first_derivative(features,labels,parameters)
        parameters-=np.linalg.inv(hessian)@first_d
    return parameters

def plot_logistic_regression(features,labels,parameters):
    positive_points= labels==1.0
    negative_points=labels==0.0
    plt.scatter(features[positive_points,1],features[positive_points,2],c='b',marker='o',label='y=1')
    plt.scatter(features[negative_points,1],features[negative_points,2],c='r',marker='x',label='y=0')
    x1_min,x1_max = features[:,1].min()-0.5,features[:,1].max()+0.5
    x2_min,x2_max = features[:,2].min()-0.5,features[:,2].max()+0.5

    if abs(parameters[2]) > 1e-6:   
        x1_vals = np.array([x1_min,x1_max])
        x2_vals = -(parameters[0] + parameters[1]*x1_vals) / parameters[2]
        plt.plot(x1_vals,x2_vals,'k-',linewidth=2,label="Decision Boundary")
    else: 
        x1_boundary =-parameters[0]/parameters[1]
        plt.axvline(x=x1_boundary,color='k',linewidth=2,label="Decision Boundary")

    plt.xlim(x1_min,x1_max)
    plt.ylim(x2_min,x2_max)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title("logistic regression decision boundary")
    plt.show()

x1,x2=standard_normalization(features)
x0=np.ones((features.shape[0]))
features=np.column_stack((x0,x1,x2))
parameters=np.array([0,0,0],dtype=float)

parameters=newtons_method(features,labels,parameters,num_iterations=3)
print(f'parameters learned = {parameters}')
plot_logistic_regression(features,labels,parameters)
# [ 0.46722676  2.55770122 -2.78143761]