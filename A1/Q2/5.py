import numpy as np
import matplotlib.pyplot as plt 
import time
from helper_fns import train_test_split,plot_data2d
np.random.seed(42)

def sample_data(mean,varience,num_samples):
    return np.random.normal(mean,varience,num_samples)

def process_in_batches(features,labels,batch_size):
    batch_data=[]
    for i in range(0,features.shape[0],batch_size):
        batch_data.append((features[i:i+batch_size], labels[i:i+batch_size]))
    return batch_data

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

def convergence_criteria(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False

def mini_batch_gradient_descent(parameters,batch_data,learning_rate,params_trajectory=None):
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
            if params_trajectory is not None:
                params_trajectory.append(parameters.copy()) 
            loss=loss_function(labels,pred)
            iteration_loss+=loss
            final_loss=loss
            print(f"Epoch {epoch+1}/{epochs}, Batch {batch_index+1}/{num_batches}, parameters={parameters}, loss={loss}")
        final_loss=iteration_loss/num_batches
        if convergence_criteria(final_loss,old_loss):
            print("Convergence reached.")
            print(f"Epoch {epoch+1}/{epochs} completed. Parameters: {parameters}, Loss: {iteration_loss/num_batches}")
            break
        print()
    print(f"Final parameters: {parameters}, Final Loss: {final_loss}")

def plot_3d(params_trajectory):
    params_trajectory = np.array(params_trajectory)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(params_trajectory[:, 0], params_trajectory[:, 1], params_trajectory[:, 2], marker='o')
    ax.plot([true_parameters[0]], [true_parameters[1]], [true_parameters[2]], marker='x', markersize=10, color='r', label='True Parameters')
    ax.set_xlabel('Parameter 1')
    ax.set_ylabel('Parameter 2')
    ax.set_zlabel('Parameter 3')
    ax.set_title('Parameter Trajectory in 3D Space')
    plt.show()

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
batch_sizes=[1,80,8000,800000]
train_size=0.8


features,labels,parameters=generate_data(true_parameters,num_samples)
train_data,test_data,train_labels,test_labels=train_test_split(features,labels) 
print("Training data and labels shape:",train_data.shape,train_labels.shape)
print("Testing data and labels shape:",test_data.shape,test_labels.shape)

params_trajectory=[]
batch_size=batch_sizes[2]
'''
Change batch size here
'''
print(f'Starting with batch size: {batch_size}')
batch_data=process_in_batches(train_data,train_labels,batch_size)
start = time.perf_counter()
mini_batch_gradient_descent(parameters,batch_data,learning_rate,params_trajectory)
end = time.perf_counter()     # end timer
print(f"Total execution time: {end - start:.4f} seconds")
print('parameters trajectory length:', len(params_trajectory))
plot_3d(params_trajectory[::1000])  # Plot every 100th point to reduce clutter

