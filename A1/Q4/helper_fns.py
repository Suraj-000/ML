import numpy as np
import matplotlib.pyplot as plt 
import sys

def standard_normalization(features):
    m,_=features.shape
    x1=np.array([features[i][0] for i in range(m)])
    x2=np.array([features[i][1] for i in range(m)])
    x1_mean=np.mean(x1)
    x2_mean=np.mean(x2)
    x1_std=np.std(x1)
    x2_std=np.std(x2)
    x1=(x1-x1_mean)/x1_std
    x2=(x2-x2_mean)/x2_std
    return x1,x2

def plot(alaska,canada,parameters=[]):
    plt.figure(figsize=(8,8))
    plt.scatter(alaska[:,0],alaska[:,1],marker='o',c='b',label='alaska')
    plt.scatter(canada[:,0],canada[:,1],marker='x',c='r',label='canada')
    if len(parameters):
        x=np.linspace(-2.0,2.0,100)
        y=-(parameters[0]*x)/parameters[1]
        plt.plot(x,y,'k-',label='Decision Boundary')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("LDA:Canada vs Alaska")
    plt.legend()
    plt.grid(True)
    plt.savefig('data_plot.png')
    plt.show()

def plot_quad_boundary(alaska_data, canada_data, w, alaska_mean, canada_mean, canada_sigma, alaska_sigma):
    plt.figure(figsize=(8,8))
    plt.scatter(alaska_data[:,0],alaska_data[:,1],marker='o',c='b',label='alaska')
    plt.scatter(canada_data[:,0],canada_data[:,1],marker='x',c='r',label='canada')

    x_min,x_max=plt.xlim()
    y_min,y_max=plt.ylim()
    x=np.linspace(x_min,x_max,100)
    y=-(w[0]*x)/w[1]
    plt.plot(x,y,'k-',label='Decision Boundary')

    x_range=np.linspace(x_min,x_max,200)
    y_range=np.linspace(y_min,y_max,200)
    X,Y = np.meshgrid(x_range, y_range)
    inv_as=np.linalg.inv(alaska_sigma)
    inv_cs=np.linalg.inv(canada_sigma)
    p,q=X.shape
    Z=np.zeros_like(X)
    for i in range(p):
        for j in range(q):
            point=np.array([X[i,j],Y[i,j]])
            alaska_score=((point-alaska_mean).T@inv_as@(point-alaska_mean)+np.log(np.linalg.det(alaska_sigma)))
            canada_score=((point-canada_mean).T@inv_cs@(point-canada_mean)+np.log(np.linalg.det(canada_sigma)))
            Z[i,j]=alaska_score -canada_score

    plt.contour(X,Y,Z,levels=[0],c='p')
    plt.plot([],[],'purple',label='Quadratic Boundary')

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("LDA:Canada vs Alaska")
    plt.legend()
    plt.grid(True)
    plt.savefig('data_plot_quad.png')
    plt.show()