import numpy as np
import matplotlib.pyplot as plt
import sys
from helper_fns import least_squares_loss,batch_gradient_descent

features=np.loadtxt('Q1_data/linearX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q1_data/linearY.csv',delimiter=',',skiprows=1)

parameters,data=batch_gradient_descent([0,0],labels,features,0.1)

intercept, slope=parameters

intercept_path=np.array([d[0]for d in data])
slope_path=np.array([d[1] for d in data])
loss_path=np.array([d[2] for d in data])

intercept, slope=parameters

Intercept=np.linspace(intercept-30,intercept+30,100)
Slope=np.linspace(slope-30,slope+30,100)
X,Y=np.meshgrid(Intercept,Slope)

Z=np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pred=X[i][j]+Y[i][j]*features
        Z[i][j]=least_squares_loss(pred,labels)

fig=plt.figure(figsize=(8, 6))
ax=fig.add_subplot(projection='3d')
cp=ax.plot_surface(X,Y,Z,cmap='viridis',linewidth=0, edgecolor='none',alpha=0.8)

fig.colorbar(cp,ax=ax,shrink=0.6,aspect=10,label='Loss')

params_point , =ax.plot([],[],[],'ro',markersize=6, label='current loss')
plt.pause(3)
for i in range(len(intercept_path)):
    ax.plot([intercept_path[i]],[slope_path[i]],[loss_path[i]], marker='o',color='r',markersize=4,alpha=0.9)
    params_point.set_data([intercept_path[i]],[slope_path[i]])
    params_point.set_3d_properties([loss_path[i]])
    plt.pause(0.2)

ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Loss')
ax.set_title('Error Surface')    
ax.auto_scale_xyz(Intercept,Slope, Z)    
plt.show()