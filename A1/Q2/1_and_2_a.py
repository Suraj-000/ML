import numpy as np
import matplotlib.pyplot as plt 

np.random.seed(42)

def sample_data(mean,varience,num_samples):
    return np.random.normal(mean,varience,num_samples)

def plot_data2d(features,labels):
    fig=plt.figure(figsize=(8,6))
    plt.scatter(features,labels)
    plt.xlabel("Features")
    plt.ylabel("Labels")
    plt.title("Normal Data")
    plt.show()

def train_test_split(features,labels,train_size=0.8):
    train_data=features[:int(len(features)*train_size)]
    test_data=features[int(len(features)*train_size):]
    train_labels=labels[:int(len(labels)*train_size)]
    test_labels=labels[int(len(labels)*train_size):]
    return train_data,test_data,train_labels,test_labels

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

true_parameters= [3,1,2]
num_samples=1000000
epochs=1
learning_rate=0.01

parameters=[0,0,0]
x0=np.ones(num_samples)
x1=sample_data(3,2,num_samples)
x2=sample_data(-1,2,num_samples)
noise=sample_data(0,np.sqrt(2),num_samples)
features=np.column_stack((x0,x1,x2))
labels=true_parameters[0] + true_parameters[1]*x1 + true_parameters[2]*x2 + noise
train_data,test_data,train_labels,test_labels=train_test_split(features,labels) 
print("Training data and labels shape:",train_data.shape,train_labels.shape)
print("Testing data and labels shape:",test_data.shape,test_labels.shape)

stocastic_gradient_descent(parameters,train_labels,train_data,learning_rate)