import matplotlib.pyplot as plt
import numpy as np  
from helper_fns import scatter_plot2d,least_squares_loss
# read data from a csv file
features=np.loadtxt('Q1_data/linearX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q1_data/linearY.csv',delimiter=',',skiprows=1)

def stopping_criteria_loss(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False 

def batch_gradient_descent(parameters,labels,features,learning_rate,flag=False):
    i=0
    old_loss=np.mean(labels)
    scatter_plot2d(features,labels,parameters,i)
    while True:
        i+=1
        pred=parameters[0]+parameters[1]*features
        error=pred - labels 
        parameters[0] = parameters[0] - learning_rate * np.mean(error)
        parameters[1] = parameters[1] - learning_rate * np.mean(error * features)
        loss=least_squares_loss(pred,labels)
        if (stopping_criteria_loss(old_loss,loss) or (i>10000)): break
        old_loss=loss
        if flag:
            if i%100==0:
                scatter_plot2d(features,labels,parameters,i)
                print(f"Iteration {i}: parameters={parameters}, loss={loss}")
    scatter_plot2d(features,labels,parameters,i)
    print(f"Iteration {i}: parameters={parameters}, loss={loss}")
    return parameters

parameters=[0,0]
learning_rates=[0.1,0.01,0.025,0.001]
# to visualise the plots set visualise to True
batch_gradient_descent(parameters,labels,features,learning_rates[1],visualize=False)
