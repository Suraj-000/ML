import numpy as np  
from helper_fns import scatter_plot2d,least_squares_loss
# read data from a csv file
features=np.loadtxt('Q1_data/linearX.csv',delimiter=',',skiprows=1)
labels=np.loadtxt('Q1_data/linearY.csv',delimiter=',',skiprows=1)
x=np.ones((features.shape[0]))
features2=np.column_stack((x,features))

data=[]

def stopping_criteria_loss(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False 

def batch_gradient_descent(parameters,labels,features,learning_rate):
    i=0
    loss=least_squares_loss(np.zeros_like(labels),labels)
    old_loss=loss
    scatter_plot2d(features,labels,parameters,i)
    print(f"Iteration {i}: Parameters={parameters}, LMS Loss={loss}")
    while True:
        i+=1
        pred=parameters[0]+parameters[1]*features
        error=pred - labels 
        parameters[0] = parameters[0] - learning_rate * np.mean(error)
        parameters[1] = parameters[1] - learning_rate * np.mean(error * features)
        pred=parameters[0]+parameters[1]*features
        loss=least_squares_loss(pred,labels)
        if (stopping_criteria_loss(old_loss,loss) and i != 1): break
        old_loss=loss
        data.append([parameters[0],parameters[1],loss])
        # if i%100==0:
        scatter_plot2d(features,labels,parameters,i)
        print(f"Iteration {i}: Parameters={parameters}, LMS Loss={loss}")
    data.append([parameters[0],parameters[1],loss])
    scatter_plot2d(features,labels,parameters,i)
    print(f"Iteration {i}: Parameters={parameters}, LMS Loss={loss}")

parameters=[0,0]

def lipschitz_constant(features2):
    hessian=(features2.T@features2)/features2.shape[0]
    eigenvalues=np.linalg.eigvals(hessian)
    print(eigenvalues)
    return max(eigenvalues)

l=lipschitz_constant(features2)
print(f"Lipschitz constant: {l}")
batch_gradient_descent(parameters,labels,features,1/l)