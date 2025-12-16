import numpy as np  
import matplotlib.pyplot as plt

def least_squares_loss(predictions,labels):
    return np.mean((predictions - labels) ** 2)


def scatter_plot2d(features,labels,parameters,iteration=None):
    x,y=features,labels
    plt.figure(figsize=(8, 6))
    pred=parameters[0]+parameters[1]*x
    plt.plot(x, pred, color='red', label='Prediction Line')
    plt.scatter(x,y,color='blue',marker='o',s=10)
    plt.xlabel('Acidity')
    plt.ylabel('Density')
    plt.title(f'Density of wine based on its acidity. iteration={iteration}')
    # add loss values to the plot
    loss=least_squares_loss(pred,labels)
    plt.text(0.5, 0.9, f'Loss: {loss:.4f} Params: {parameters[0],parameters[1]}', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
    plt.show()  

def batch_gradient_descent(parameters,labels,features,learning_rate,flag=False):
    i=0 
    data=[]
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
        data.append([parameters[0],parameters[1],loss])
        if flag:
            if i%100==0:
                scatter_plot2d(features,labels,parameters,i)
                print(f"Iteration {i}: parameters={parameters}, loss={loss}")
    data.append([parameters[0],parameters[1],loss])
    scatter_plot2d(features,labels,parameters,i)
    print(f"Iteration {i}: parameters={parameters}, loss={loss}")
    return parameters,data

def stopping_criteria_loss(old_loss,new_loss,threshold=1e-6):
    difference=abs(old_loss-new_loss)
    if difference<threshold: return True
    return False 