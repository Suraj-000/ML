import numpy as np
import matplotlib.pyplot as plt
from helper_fns import least_squares_loss,batch_gradient_descent

features=np.loadtxt('Q1_data/linearX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q1_data/linearY.csv',delimiter=',',skiprows=1)

parameters,data=batch_gradient_descent([0,0],labels,features,0.01)

intercept, slope=parameters

Intercept=np.linspace(intercept-5,intercept+5,50)
Slope=np.linspace(slope-5,slope+5,50)
X,Y=np.meshgrid(Intercept,Slope)

Z=np.zeros(X.shape)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pred=X[i][j]+Y[i][j]*features
        Z[i][j]=least_squares_loss(pred,labels)

fig=plt.figure(figsize=(8, 6))

ax=fig.add_subplot(projection='3d')
cp=ax.plot_surface(X,Y,Z,cmap='viridis',linewidth=0, edgecolor='none')
fig.colorbar(cp,ax=ax,shrink=0.6,aspect=10,label='Loss')

ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Loss')
ax.set_title('Error Surface')    
ax.auto_scale_xyz(Intercept,Slope, Z)    
plt.show()