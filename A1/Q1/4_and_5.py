
import numpy as np
import matplotlib.pyplot as plt
from helper_fns import least_squares_loss,batch_gradient_descent

features=np.loadtxt('Q1_data/linearX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q1_data/linearY.csv',delimiter=',',skiprows=1)

def stopping_criteria_loss(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False 

def plot_contours(X,Y,Z,intercept_path,slope_path,lr):
    fig=plt.figure(figsize=(8, 6))
    cp=plt.contour(X,Y,Z,levels=30,cmap='viridis')
    plt.clabel(cp, inline=True, fontsize=8)

    plt.xlabel('Intercept')
    plt.ylabel('Slope')
    plt.title(f'Error Contours lr={lr}')
    plt.colorbar(cp,label='Loss')

    plt.pause(2)
    for i in range(len(intercept_path)):
        plt.plot(intercept_path[i],slope_path[i],'ro',markersize=6)
        plt.pause(0.1)
    plt.show()

parameters=[0,0]
learning_rates=[0.1,0.025,0.001,0.01]
lr=learning_rates[3]
parameters,data= batch_gradient_descent(parameters,labels,features,lr)
intercept_path=np.array(data)[:,0]
slope_path=np.array(data)[:,1]
intercept, slope=parameters

Intercept=np.linspace(intercept_path.min()-5,intercept_path.max()+5,50)
Slope=np.linspace(slope_path.min()-5,slope_path.max()+5,50)

X,Y=np.meshgrid(Intercept,Slope)
Z=np.zeros(X.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pred=X[i][j]+Y[i][j]*features
        Z[i][j]=least_squares_loss(pred,labels)

# Change step value to visualise more or less points on the contours
step=20
plot_contours(X,Y,Z,intercept_path[::step],slope_path[::step],lr)
