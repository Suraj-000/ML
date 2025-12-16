import numpy as np
import matplotlib.pyplot as plt 

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
