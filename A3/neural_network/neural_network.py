
import os
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
import random
import cv2
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report,accuracy_score

class Logger:
    def __init__(self,file_path):
        self.terminal=sys.stdout
        self.log=open(file_path,"w",encoding="utf-8")

    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

original_stdout = sys.stdout

def logging(s='1',device='cpu'):
    global current_logger
    os.makedirs("logs",exist_ok=True)
    log_path = os.path.join('logs', f'output_{s}.txt')
    current_logger = Logger(log_path)
    sys.stdout = current_logger
    print("Logging started — all terminal output will also go to:", log_path)
    print("Using device:", device)

def print_star(n=100):
    print("*"*n)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def open_img_to_numpy(path):
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    arr=np.array(img,dtype=np.float32)/255.0
    return arr.flatten()

def data_loader(root_path):
    data,labels=[],[]
    class_names=sorted(os.listdir(root_path))
    for label,folder in enumerate(class_names):
        class_folder=os.path.join(root_path,folder)

        if not os.path.isdir(class_folder):continue
        img_files=sorted(os.listdir(class_folder))
        for file in img_files:
            img_path=os.path.join(class_folder,file)
            img=open_img_to_numpy(img_path)
            data.append(img)
            labels.append(label)
    X=np.array(data)
    y=np.array(labels)
    return X,y,class_names

class NeuralNetwork:

    def __init__(self,hps):

        self.input_dim=hps['input_dim']
        self.output_dim=hps['output_dim']
        self.hidden_layers=hps['hidden_layers']
        self.learning_rate=hps['learning_rate']
        self.batch_size=hps['batch_size']
        self.epochs=hps['epochs']
        self.seed=hps['seed']
        self.delta=hps['delta']
        self.patience=hps['patience']
        self.activation_type=hps['activation']

        set_seed(self.seed)

        self.network_layers=[self.input_dim]+self.hidden_layers+[self.output_dim]
        self.num_layers=len(self.network_layers)

        self.weights,self.biases=self.init_weights_bias(self.num_layers,self.network_layers)

    def init_weights_bias(self,num_layers,network_layers):
        weights,bias=[],[]
        for i in range(num_layers-1):
            if self.activation_type=='relu':
                w=np.random.randn(network_layers[i],network_layers[i+1])*np.sqrt(2.0/network_layers[i])
            else:
                w=np.random.randn(network_layers[i],network_layers[i+1])*np.sqrt(1.0/network_layers[i])
            b=np.zeros((1,network_layers[i+1]))
            weights.append(w)
            bias.append(b)
        return weights,bias

    def relu(self,z):
        return np.maximum(0,z)
    
    def diff_relu(self,z):
        grad=np.where(z>0,1.0,0.0)
        grad[z==0]=0.5
        return grad

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        
    def diff_sigmoid(self,p):
        return p*(1-p)
    
    def softmax(self,scores):
        shifted=scores-np.max(scores,axis=1,keepdims=True)
        exp_vals=np.exp(np.clip(shifted,-50,50))
        probabilities=exp_vals/np.sum(exp_vals,axis=1,keepdims=True)
        return probabilities

    def cross_entropy_loss(self,labels,preds):
        predicted_probs = np.clip(preds,1e-15,1 - 1e-15)
        loss =-np.mean(np.sum(labels*np.log(predicted_probs), axis=1))
        return loss
    
    def forward(self,x):

        self.z=[]
        self.activation=[x]

        for i in range(len(self.weights)-1):
            z=np.dot(self.activation[-1],self.weights[i])+self.biases[i]
            if self.activation_type=='relu':activation=self.relu(z)
            else:activation=self.sigmoid(z)
            self.z.append(z)
            self.activation.append(activation)

        z_out=np.dot(self.activation[-1],self.weights[-1])+self.biases[-1]
        a_out=self.softmax(z_out)
        self.z.append(z_out)
        self.activation.append(a_out)
        return a_out
    
    def back_prop(self,x,y):
        m=x.shape[0]
        num_layers=len(self.weights)

        grad_W = [None] * num_layers
        grad_b = [None] * num_layers
        
        delta=self.activation[-1]-y

        grad_W[-1]=np.dot(self.activation[-2].T, delta) / m
        grad_b[-1] = np.sum(delta,axis=0, keepdims=True) / m

        for i in range(num_layers - 2, -1, -1):
            if self.activation_type=='relu' :delta = np.dot(delta, self.weights[i+1].T)*self.diff_relu(self.z[i])
            else: delta = np.dot(delta, self.weights[i+1].T) * self.diff_sigmoid(self.activation[i+1])

            grad_W[i] = np.dot(self.activation[i].T, delta) / m
            grad_b[i] = np.sum(delta, axis=0, keepdims=True) / m
        
        return grad_W, grad_b
    
    def update_params(self,weight_grads,bias_grads):
        for i in range(len(self.weights)):
            self.weights[i]-=self.learning_rate*weight_grads[i]
            self.biases[i]-=self.learning_rate*bias_grads[i]

    def one_hot_encode(self,y):
        n_samples=len(y)
        y_one_hot=np.zeros((n_samples, self.output_dim))
        y_one_hot[np.arange(n_samples), y] = 1
        return y_one_hot

    def fit(self,x_train,y_train,x_test,y_test):
        n_samples = x_train.shape[0]
        y_one_hot = self.one_hot_encode(y_train)

        train_loss_history=[]
        test_loss_history=[]
        best_loss=np.inf
        self.f1_scores_test=[]
        self.f1_scores_train=[]

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = x_train[indices]
            y_shuffled = y_one_hot[indices]
            
            no_progress=0
            epoch_loss = 0
            n_batches = 0
            for i in range(0,n_samples,self.batch_size):
                batch_end=min(i+self.batch_size,n_samples)
                X_batch=X_shuffled[i:batch_end]
                y_batch=y_shuffled[i:batch_end]
        
                y_pred=self.forward(X_batch)

                batch_loss=self.cross_entropy_loss(y_batch,y_pred)
                epoch_loss+=batch_loss
                n_batches+=1

                weight_grads,bias_grads=self.back_prop(X_batch,y_batch)
                self.update_params(weight_grads,bias_grads)

            avg_epoch_loss=epoch_loss/n_batches
            train_loss_history.append(avg_epoch_loss)

            pred_train=self.predict(x_train)
            pred_test=self.predict(x_test)

            f1_train=f1_score(y_train, pred_train, average='macro',zero_division=0)
            f1_test=f1_score(y_test, pred_test, average='macro',zero_division=0)

            self.f1_scores_test.append(f1_test)
            self.f1_scores_train.append(f1_train)

            train_acc=accuracy_score(y_train,pred_train)
            test_acc=accuracy_score(y_test,pred_test)
            
            if (best_loss-avg_epoch_loss)<=self.delta:
                best_loss=avg_epoch_loss
                no_progress=0
            else: no_progress+=1

            if (epoch + 1) % 1 == 0:
                print(f'--> '
                    f'Epoch {epoch+1}/{self.epochs} | '
                    f'Avg epoch Loss: {avg_epoch_loss:.4f} | '
                    f'Train Acc: {train_acc:.4f} | '
                    f'Test Acc: {test_acc:.4f} '
                    )
            if no_progress>=self.patience:
                print(f'-->Early stopping at epoch {epoch+1}| '
                      f'(no change in loss) | ')
                break

    def get_f1_20(self):
        return self.f1_scores_train,self.f1_scores_test

    def predict_proba(self, X):
        return self.forward(X)

    def predict(self,x):
        probs=self.predict_proba(x)
        return np.argmax(probs, axis=1)
    
    def get_acc(self,x,y):
        preds=self.predict(x)
        return np.mean(preds==y)
    
    def get_classification_report(self,x,y):
        preds=self.predict(x)
        return classification_report(y,preds)
    
    def save_weights(self,file_path="model_weights_f.npz"):
        w=np.empty(len(self.weights),dtype=object)
        b=np.empty(len(self.biases),dtype=object)
        for i in range(len(self.weights)):
            w[i]=self.weights[i]
            b[i]=self.biases[i]
        np.savez_compressed(file_path,weights=w,biases=b)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path="model_weights_f.npz"):
        data=np.load(file_path, allow_pickle=True)
        self.weights=[w for w in data["weights"]]
        self.biases=[b for b in data["biases"]]
        print(f"Model weights loaded from {file_path}")

    def load_for_transfer(self, pretrained_path, new_output_dim):
        data=np.load(pretrained_path, allow_pickle=True)
        pretrained_w=[w for w in data["weights"]]
        pretrained_b=[b for b in data["biases"]]
        for i in range(len(self.weights) - 1):
            self.weights[i]=pretrained_w[i]
            self.biases[i]=pretrained_b[i]

        in_dim=self.weights[-2].shape[1]
        self.weights[-1]=np.random.randn(in_dim, new_output_dim)*np.sqrt(2.0 / in_dim)
        self.biases[-1]=np.zeros((1, new_output_dim))
