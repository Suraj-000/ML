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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

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
    return x,y.values

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
    if flag==1:plot_path=os.path.join("../plots",f"Q6_accuracy_vs_pruning_param.png")
    else:plot_path=os.path.join("../plots",f"Q6_accuracy_vs_max_depth.png")
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
    print(f"Q6. Random Forest Construction Using sklearn libraries")
    _,train_path,test_path,val_path,output_path=sys.argv
    x_train,y_train=load_data(train_path)
    print(f'Number of training samples: {x_train.shape[0]}')
    x_test,y_test=load_data(test_path)
    print(f'Number of testing samples: {x_test.shape[0]}')
    x_val,y_val=load_data(val_path)
    print(f'Number of validation samples: {x_val.shape[0]}')


    best_val_acc=0
    best_params=None
    results=[]
    best_model=None
    bootstrap=True
    criterion='entropy'

    param_grid = {
        'n_estimators':[50,150,250,350],
        'max_features':[0.1,0.3,0.5,0.7,0.9],
        'min_samples_split':[2,4,6,8,10],
        'criterion':['entropy'],
        'bootstrap':[True]
    }

    rf=RandomForestClassifier(oob_score=True,random_state=42,n_jobs=-1)

    grid_search=GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=3,
        scoring='accuracy'
    )

    grid_search.fit(x_train,y_train)

    best_rf=grid_search.best_estimator_
    print("best parameters are : ",grid_search.best_params_)

    train_acc=accuracy_score(y_train,best_rf.predict(x_train))
    test_acc=accuracy_score(y_test,best_rf.predict(x_test))
    val_pred=best_rf.predict(x_val)
    save_result_in_csv(output_path,val_pred)
    val_acc=accuracy_score(y_val,val_pred)
    oob_acc=best_rf.oob_score_

    print(f'train acc: {train_acc:.4f}')
    print(f'test acc: {test_acc:.4f}')
    print(f'val acc: {val_acc:.4f}')
    print(f'OOB acc: {oob_acc:.4f}')

    end_time=time.time()
    print(f'total execution time = {(end_time-start_time):.4f}')
    print_star()

if __name__=='__main__':
    main()