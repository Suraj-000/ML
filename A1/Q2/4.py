import numpy as np
import matplotlib.pyplot as plt 
from helper_fns import train_test_split
np.random.seed(42)

def sample_data(mean,varience,num_samples):
    return np.random.normal(mean,varience,num_samples)

def loss_function(labels,predictions):
    return np.mean((labels-predictions)**2)/2

def stocastic_gradient_descent(parameters,labels,features,learning_rate):
    for epoch in range(epochs):
        for i in range(features.shape[0]):
            pred=parameters[0]*features[i][0]+parameters[1]*features[i][1]+parameters[2]*features[i][2]
            error=pred-labels[i]
            parameters[0] = parameters[0] - learning_rate * error* features[i][0]
            parameters[1] = parameters[1] - learning_rate * error * features[i][1]
            parameters[2] = parameters[2] - learning_rate * error * features[i][2]
            if (i+1)%1==0:
                loss=np.mean((labels-pred)**2)
                print(f"Iteration {i}: parameters={parameters}, loss={loss}")
        print(f"Epoch {epoch+1}/{epochs} completed.")
    return parameters

true_parameters= [3,1,2]
num_samples=1000000
batch_size=80
epochs=5
learning_rate=0.001

parameters=[0,0,0]
x0=np.ones(num_samples)
x1=sample_data(3,2,num_samples)
x2=sample_data(-1,2,num_samples)
noise=sample_data(0,1,num_samples)
features=np.column_stack((x0,x1,x2))
labels=true_parameters[0] + true_parameters[1]*x1 + true_parameters[2]*x2 + noise
train_data,test_data,train_labels,test_labels=train_test_split(features,labels) 
print("Training data and labels shape:",train_data.shape,train_labels.shape)
print("Testing data and labels shape:",test_data.shape,test_labels.shape)

'''
these parameter are learned for batch size of 8000 with learning range 0.001
'''
parameters=[2.9906,1.0018,2.0006]
test_pred=test_data.dot(parameters)
test_loss=loss_function(test_labels,test_pred)
print("Test Loss:",test_loss)