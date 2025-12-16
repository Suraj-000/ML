import sys,os
import time
import pandas as pd
import csv
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

os.makedirs("logs",exist_ok=True)
log_path=os.path.join('logs','output.txt')
class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")
        self.log.write("\n\n--- Run started at: " + str(datetime.now()) + " ---\n")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
sys.stdout = Logger(log_path)
print("Logging started — all terminal output will also go to:", log_path)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from decision_tree import DecisionTree

def print_star(n=100):
    print("*"*n)

def load_data(path):
    ''' 
    path: path to csv file
    '''
    df=pd.read_csv(path)
    if 'result' not in df.columns: 
        print(f'column named results not in csv file')
        sys.exit()
    y=df['result']
    x=df.drop('result',axis=1)
    label_encode={}
    for att in x.columns:
        if x[att].dtype=='object':
            le=LabelEncoder()
            x[att]=le.fit_transform(x[att])
            label_encode[att]=le
    return x,y

def plot(max_depths,train_acc,test_acc,val_acc,flag):
    os.makedirs("../plots", exist_ok=True)
    plt.figure(figsize=(10,8))
    plt.plot(max_depths,train_acc,marker='o',label='Train Accuracy')
    plt.plot(max_depths,test_acc,marker='s',label='Test Accuracy')
    plt.plot(max_depths,val_acc,marker='x',label='Validation Accuracy')
    for i, depth in enumerate(max_depths):
        plt.text(depth, train_acc[i] + 0.005, f"{train_acc[i]:.3f}", ha='center', color='blue', fontsize=9,bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        plt.text(depth, val_acc[i] - 0.015, f"{val_acc[i]:.3f}", ha='center',va='bottom', color='green', fontsize=9,bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
        plt.text(depth, test_acc[i] + 0.015, f"{test_acc[i]:.3f}", ha='center', va='top', color='orange', fontsize=9,bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    plt.xlabel('Maximum Depth of Tree')
    plt.ylabel('Accuracy')
    if flag==1:plt.title('Decision Tree Accuracy vs Pruning parameter')
    else: plt.title('Decision Tree Accuracy vs Maximum Depth')
    plt.legend()
    plt.grid(True)
    plot_path=None
    if flag==1:plot_path=os.path.join("../plots",f"Q5_accuracy_vs_pruning_param.png")
    else:plot_path=os.path.join("../plots",f"Q5_accuracy_vs_max_depth.png")
    plt.savefig(plot_path)
    # plt.show()
    plt.close()
    print(f'image saved in plots folder')

def save_result_in_csv(path,y_test_pred,header='result'):
    data={header:y_test_pred}
    df=pd.DataFrame(data)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path,index=False)
    print(f'Data saved to :{path}')

def main():
    print_star()
    start_time=time.time()
    print(f"Q5. Decision tree construction Using sklearn libraries")
    _,train_path,test_path,val_path,output_path=sys.argv

    x_train,y_train=load_data(train_path)
    print(f'Number of training samples: {x_train.shape[0]}')
    x_test,y_test=load_data(test_path)
    print(f'Number of testing samples: {x_test.shape[0]}')
    x_val,y_val=load_data(val_path)
    print(f'Number of validation samples: {x_val.shape[0]}')

    train_acc,test_acc,val_acc=[],[],[]
    max_depths=[15,25,35,45]
    criterion='entropy'
    one_test_pred_data=[]
    one_val_acc=[]
    for d in max_depths:
        print_star(80)
        print(f'Implementing decison tree for max depth {d}')
        
        dtree=DecisionTreeClassifier(criterion=criterion,max_depth=d,random_state=42)
        dtree.fit(x_train,y_train)

        train_pred=dtree.predict(x_train)
        test_pred=dtree.predict(x_test)
        val_pred=dtree.predict(x_val)

        accuracy_train=np.mean(train_pred==y_train)
        accuracy_test=np.mean(test_pred==y_test)
        accuracy_val=np.mean(val_pred==y_val)
        print(f"train accuracy: {accuracy_train:.4f}, Test accuracy: {accuracy_test:.4f}, validation accuracy: {accuracy_val:.4f}")
    
        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)
        val_acc.append(accuracy_val)

        one_test_pred_data.append(test_pred)
        one_val_acc.append(accuracy_val)
        print_star(80)
        # sys.exit()
    plot(max_depths,train_acc,test_acc,val_acc,0)

    print_star()
    print(f'(ii) DT classifier with default depth and varing ccp_alpha')
    train_acc,test_acc,val_acc=[],[],[]
    ccp_alphas=[0.0,0.0001,0.0003,0.0005]
    two_test_pred_data=[]
    two_val_acc=[]
    criterion='entropy'
    for d in ccp_alphas:
        print_star(80)
        print(f'Implementing decison tree for pruning parameter {d}')
        
        # dtree=DecisionTree(max_depth=d)
        dtree=DecisionTreeClassifier(criterion=criterion,ccp_alpha=d,random_state=42)
        dtree.fit(x_train,y_train)

        train_pred=dtree.predict(x_train)
        test_pred=dtree.predict(x_test)
        val_pred=dtree.predict(x_val)

        accuracy_train=np.mean(train_pred==y_train)
        accuracy_test=np.mean(test_pred==y_test)
        accuracy_val=np.mean(val_pred==y_val)
        print(f"train accuracy: {accuracy_train:.4f}, Test accuracy: {accuracy_test:.4f}, validation accuracy: {accuracy_val:.4f}")

        two_test_pred_data.append(test_pred)
        two_val_acc.append(accuracy_val)

        train_acc.append(accuracy_train)
        test_acc.append(accuracy_test)
        val_acc.append(accuracy_val)
        print_star(80)
        # sys.exit()
    plot(ccp_alphas,train_acc,test_acc,val_acc,1)

    max_idx1=np.argmax(one_val_acc)
    max_idx2=np.argmax(two_val_acc)
    a,b=np.max(one_val_acc),np.max(two_val_acc)
    print(f'model with max depth {max_depths[max_idx1]} with val acc {a}')
    print(f'model with cpp alpha {ccp_alphas[max_idx2]} with val acc {b}')

    if a>b: save_result_in_csv(output_path,one_test_pred_data[max_idx1])
    else: save_result_in_csv(output_path,two_test_pred_data[max_idx2])
    print(f'csv file saved')


    end_time=time.time()
    print(f'total execution time = {(end_time-start_time):.4f}')
    print_star()

if __name__=='__main__':
    main()