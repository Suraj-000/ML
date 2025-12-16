import numpy as np
import matplotlib.pyplot as plt 
import time
from helper_fns import plot_data2d

np.random.seed(42)

def sample_data(mean,varience,num_samples):
    return np.random.normal(mean,varience,num_samples)

def train_test_split(features,labels,train_size=0.8):
    train_data=features[:int(len(features)*train_size)]
    test_data=features[int(len(features)*train_size):]
    train_labels=labels[:int(len(labels)*train_size)]
    test_labels=labels[int(len(labels)*train_size):]
    return train_data,test_data,train_labels,test_labels

def process_in_batches(features,labels,batch_size):
    batch_data=[]
    for i in range(0,features.shape[0],batch_size):
        batch_data.append((features[i:i+batch_size], labels[i:i+batch_size]))
    return batch_data

def loss_function(labels,predictions):
    return np.mean((labels-predictions)**2)/2

def convergence_criteria(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False

def mini_batch_gradient_descent(parameters,batch_data,learning_rate):
    num_batches=len(batch_data)
    old_loss=0
    final_loss=0
    for epoch in range(epochs):
        old_loss=final_loss
        iteration_loss=0
        for batch_index in range(num_batches):
            features,labels=batch_data[batch_index]
            pred=features.dot(parameters)
            error=pred-labels
            parameters= parameters - learning_rate * (1/len(labels))*(features.T.dot(error))
            loss=loss_function(labels,pred)
            iteration_loss+=loss
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_index+1}/{num_batches}, parameters={parameters}, loss={loss}")
        final_loss=iteration_loss/num_batches
        if convergence_criteria(final_loss,old_loss):
            print("Convergence reached.")
            print(f"Epoch {epoch+1}/{epochs} completed. Parameters: {parameters}, Loss: {iteration_loss/num_batches}")
            break
        print()
    print(f"Final parameters: {parameters}, Final Loss: {final_loss}")

def generate_data(true_parameters,num_samples):
    parameters=np.array([0,0,0])
    x0=np.ones(num_samples)
    x1=sample_data(3,2,num_samples)
    x2=sample_data(-1,2,num_samples)
    noise=sample_data(0,np.sqrt(2),num_samples)
    features=np.column_stack((x0,x1,x2))
    labels = features@true_parameters + noise
    return features,labels,parameters


true_parameters = [3,1,2]
num_samples=1000000
epochs=15000
learning_rate=0.001
batch_size=[1,80,8000,800000]
train_size=0.8

features,labels,parameters=generate_data(true_parameters,num_samples)
train_data,test_data,train_labels,test_labels=train_test_split(features,labels,train_size) 
print("Training data and labels shape:",train_data.shape,train_labels.shape)
print("Testing data and labels shape:",test_data.shape,test_labels.shape)


'''
change batch size here
'''
print(f'Starting with batch size: {batch_size[2]}')
batch_data=process_in_batches(train_data,train_labels,batch_size[2])
start = time.perf_counter()
mini_batch_gradient_descent(parameters,batch_data,learning_rate)
end = time.perf_counter()     # end timer
print(f"Total execution time: {end - start:.4f} seconds")
